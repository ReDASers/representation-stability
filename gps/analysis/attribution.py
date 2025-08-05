"""
Embedding utilities for adversarial text analysis.

This module provides functions for computing text embeddings, analyzing word importance
through various attribution methods (gradients, integrated gradients, attention rollout, random), and
measuring sensitivity through perturbation-based approaches. It supports transformer
models and handles batch processing for efficiency.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
)
import random
import numpy as np

from ..core.sensitivity import compute_embedding_distance
from ..utils.text_processing import (
    map_subtokens_to_words,
    split_into_words,
    tokenize_texts,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_mean_pooled_representation(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model,
    layers_to_average: int = 1
) -> torch.Tensor:
    """
    Compute mean-pooled embeddings from model hidden states.
    
    Parameters
    ----------
    input_ids : torch.Tensor
        Token IDs tensor of shape [batch_size, seq_length].
    attention_mask : torch.Tensor
        Attention mask tensor of shape [batch_size, seq_length].
    model : Any
        Transformer model with hidden states output.
    layers_to_average : int, default 1
        Number of layers to average from the end.
        
    Returns
    -------
    torch.Tensor
        Mean-pooled embeddings of shape [batch_size, hidden_size].
    """
    # Move inputs to the model's device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Select hidden states to use
        if layers_to_average == 1:
            hidden_states = outputs.hidden_states[-1]
        else:
            # Average multiple layers from the end
            hidden_states = torch.stack(outputs.hidden_states[-layers_to_average:])
            hidden_states = torch.mean(hidden_states, dim=0)
        
        # Apply attention mask to zero out pad tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        
        # Normalize by the number of non-pad tokens
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings


@torch.no_grad()
def precompute_full_embeddings(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
    batch_size: int = 16,
    layers_to_average: int = 1,
) -> torch.Tensor:
    """
    Compute mean-pooled embeddings for texts in mini-batches.
    
    Parameters
    ----------
    texts : List[str]
        Input texts to embed.
    model : AutoModelForSequenceClassification
        Sequence classification model.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    max_length : int, optional
        Maximum sequence length for tokenization.
    batch_size : int, default 16
        Processing batch size.
    layers_to_average : int, default 1
        Number of model layers to average.
        
    Returns
    -------
    torch.Tensor
        Embeddings tensor of shape [num_texts, hidden_size].
    """
    if not texts:
        return torch.empty((0, model.config.hidden_size), device=next(model.parameters()).device)

    # Ensure model is on the correct device
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Process texts in batches
    all_embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings", leave=False):
        # Extract current batch
        batch = texts[start : start + batch_size]
        
        # Tokenize the batch
        enc = tokenize_texts(batch, tokenizer, max_length)
        for key in enc:
            if isinstance(enc[key], torch.Tensor):
                enc[key] = enc[key].to(device)
        
        # Compute embeddings for the batch
        emb = get_mean_pooled_representation(
            enc["input_ids"], 
            enc["attention_mask"], 
            model, 
            layers_to_average=layers_to_average
        )
        all_embs.append(emb)
    
    return torch.cat(all_embs, dim=0).to(device)

@torch.no_grad()
def identify_salient_words(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
) -> Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]:
    """
    Identify words by gradient-based saliency scores.

    Parameters
    ----------
    texts : List[str]
        Input texts to analyze.
    model : AutoModelForSequenceClassification
        Sequence classification model.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    max_length : int, optional
        Maximum sequence length for tokenization.
        
    Returns
    -------
    Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]
        Tuple containing:
        - Words with scores and indices, sorted by saliency
        - Raw word scores in original order
    """
    if not texts:
        return ([], [])

    device = next(model.parameters()).device
    model = model.to(device)

    # Process all texts in a single pass to get words and token mappings
    all_words = []
    all_token_to_word = []
    valid_indices = []
    for idx, text in enumerate(texts):
        words, token_to_word = map_subtokens_to_words(
            text,
            tokenizer,
            return_token_to_word=True,
            max_length=max_length,
        )
        if words:
            all_words.append(words)
            all_token_to_word.append(token_to_word)
            valid_indices.append(idx)

    if not valid_indices:
        return ([[] for _ in texts], [[] for _ in texts])

    valid_texts = [texts[i] for i in valid_indices]

    # Tokenize all valid texts at once
    inputs = tokenizer(
        valid_texts,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length
    ).to(device)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    B, T = input_ids.shape

    # Compute gradients for the entire batch
    with torch.enable_grad():
        # Single batch forward pass
        emb = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        
        # Forward pass
        out = model(inputs_embeds=emb, attention_mask=attention_mask)
        logits = out.logits
        
        # Get predicted classes
        preds = logits.argmax(dim=1)
        
        # Extract scores for predicted classes
        target_scores = logits.gather(1, preds.unsqueeze(-1)).squeeze(-1)
        
        # Single backward pass for entire batch
        model.zero_grad()
        target_scores.sum().backward()
        
        if emb.grad is None:
            print("Gradients for embeddings are None after backward pass. Check model architecture or hooks.")
            return ([[] for _ in texts], [[] for _ in texts])
            
        # Compute token importances as L2 norm of gradients
        grads = emb.grad.norm(dim=2)  # [B, T]

    # Process gradients into word scores
    batch_tuples: List[List[Tuple[str, float, int]]] = []
    batch_word_scores: List[List[float]] = []

    for b_idx in range(B):
        words = all_words[b_idx]
        token_to_word = all_token_to_word[b_idx]
        current_grads = grads[b_idx]

        # Aggregate token-level gradients to word-level scores
        word_scores = [0.0] * len(words)
        for tok_idx, word_idx in token_to_word.items():
            if tok_idx < current_grads.size(0):
                word_scores[word_idx] += current_grads[tok_idx].item()

        # Create sorted tuple list (word, score, original_index)
        tuples = [(w, s, i) for i, (w, s) in enumerate(zip(words, word_scores))]
        tuples.sort(key=lambda t: t[1], reverse=True)
            
        batch_tuples.append(tuples)
        batch_word_scores.append(word_scores)

    # Map the results back to the original order
    final_batch_tuples = [[] for _ in texts]
    final_batch_word_scores = [[] for _ in texts]
    processed_idx = 0
    for orig_idx in range(len(texts)):
        if orig_idx in valid_indices:
            final_batch_tuples[orig_idx] = batch_tuples[processed_idx]
            final_batch_word_scores[orig_idx] = batch_word_scores[processed_idx]
            processed_idx += 1

    return (final_batch_tuples, final_batch_word_scores)

@torch.enable_grad()
def identify_salient_words_integrated_gradients(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
    steps: int = 25,
    baseline_type: str = "zero",
) -> Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]:
    """
    Identify words using integrated gradient-based saliency.
    
    Integrated gradients aggregate gradients along a path from baseline to input,
    satisfying completeness and sensitivity properties.

    Parameters
    ----------
    texts : List[str]
        Input texts to analyze.
    model : AutoModelForSequenceClassification
        Sequence classification model.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    max_length : int, optional
        Maximum sequence length for tokenization.
    steps : int, default 25
        Number of steps for integrated gradients computation.
    baseline_type : str, default "zero"
        Baseline type: "zero", "random", or "pad".
        
    Returns
    -------
    Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]
        Tuple containing:
        - Words with scores and indices, sorted by absolute saliency
        - Raw word scores in original order
    """
    if not texts:
        return ([], [])
        
    device = next(model.parameters()).device
    model = model.to(device)

    # Process all texts to get words and token mappings
    all_words = []
    all_token_to_word = []
    valid_indices = []
    for idx, text in enumerate(texts):
        words, token_to_word = map_subtokens_to_words(
            text,
            tokenizer,
            return_token_to_word=True,
            max_length=max_length,
        )
        if words:
            all_words.append(words)
            all_token_to_word.append(token_to_word)
            valid_indices.append(idx)

    if not valid_indices:
        return ([[] for _ in texts], [[] for _ in texts])

    valid_texts = [texts[i] for i in valid_indices]

    # Tokenize all valid texts at once
    inputs = tokenizer(
        valid_texts,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length
    ).to(device)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    B, T = input_ids.shape

    # Get original embeddings and predicted classes in a single pass
    word_embeddings = model.get_input_embeddings()
    original_embeddings = word_embeddings(input_ids).detach()
    
    # Create baseline embeddings based on baseline_type
    if baseline_type == "zero":
        baseline_embeddings = torch.zeros_like(original_embeddings)
    elif baseline_type == "pad":
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        baseline_ids = torch.full_like(input_ids, pad_id)
        baseline_embeddings = word_embeddings(baseline_ids).detach()
    else:  # Random baseline
        baseline_embeddings = torch.randn_like(original_embeddings) * 0.1
    
    # Get predicted classes for all inputs
    with torch.no_grad():
        outputs = model(inputs_embeds=original_embeddings, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)  # [B]
    
    # Compute integrated gradients steps
    alphas = torch.linspace(0, 1, steps, device=device)
    grads = []
    
    # For each interpolation step
    for alpha in alphas:
        # Create interpolated embeddings for all inputs
        interpolated_embeddings = baseline_embeddings + alpha * (original_embeddings - baseline_embeddings)
        interpolated_embeddings.requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs_embeds=interpolated_embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get scores for predicted classes
        target_scores = logits.gather(1, preds.unsqueeze(-1)).squeeze(-1)
        
        # Backward pass
        model.zero_grad()
        target_scores.sum().backward()
        
        if interpolated_embeddings.grad is not None:
            grads.append(interpolated_embeddings.grad.clone())
        else:
            print(f"IG Step {alpha}: Gradients are None for interpolated embeddings.")
            grads.append(torch.zeros_like(original_embeddings))
    
    # Stack gradients from all steps
    grads = torch.stack(grads)
    
    # Compute average gradients across steps
    avg_grad = (grads[:-1] + grads[1:]).mean(dim=0) / 2
    
    # Compute integrated gradients
    integrated_gradients = (original_embeddings - baseline_embeddings) * avg_grad
    
    # Sum over embedding dimension to get token importance
    token_importance = integrated_gradients.sum(dim=2)
    token_importance_np = token_importance.detach().cpu().numpy()
    
    # Process token importance into word scores
    batch_tuples: List[List[Tuple[str, float, int]]] = []
    batch_word_scores: List[List[float]] = []

    for b_idx in range(B):
        words = all_words[b_idx]
        token_to_word = all_token_to_word[b_idx]
        current_token_importance = token_importance_np[b_idx]

        # Aggregate token-level importance to word-level scores
        word_scores = [0.0] * len(words)
        for tok_idx, word_idx in token_to_word.items():
            if tok_idx < len(current_token_importance):
                word_scores[word_idx] += current_token_importance[tok_idx]
    
        # Create sorted tuple list (word, score, original_index)
        tuples = [(w, s, i) for i, (w, s) in enumerate(zip(words, word_scores))]
        tuples.sort(key=lambda t: abs(t[1]), reverse=True)
        
        batch_tuples.append(tuples)
        batch_word_scores.append(word_scores)
    
    # Map results back to original order
    final_batch_tuples = [[] for _ in texts]
    final_batch_word_scores = [[] for _ in texts]
    processed_idx = 0
    for orig_idx in range(len(texts)):
        if orig_idx in valid_indices:
            final_batch_tuples[orig_idx] = batch_tuples[processed_idx]
            final_batch_word_scores[orig_idx] = batch_word_scores[processed_idx]
            processed_idx += 1
            
    return (final_batch_tuples, final_batch_word_scores)

@torch.enable_grad()
def compute_gradient_attention(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
    layer_aggregation: str = "last",
) -> Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]:
    """
    Compute Gradient × Attention importance scores using model hooks.

    Multiplies attention weights by their gradients, using hooks to capture
    internal attention tensors and compute gradients relative to model loss.

    Parameters
    ----------
    texts : List[str]
        Input texts to analyze.
    model : AutoModelForSequenceClassification
        Sequence classification model.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    max_length : int, optional
        Maximum sequence length for tokenization.
    layer_aggregation : str, default "last"
        Layer aggregation method: "last" or "all".
        
    Returns
    -------
    Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]
        Tuple containing:
        - Words with scores and indices, sorted by importance
        - Raw word scores in original order
    """
    if not texts:
        return ([], [])

    device = next(model.parameters()).device
    model = model.to(device)

    # Prepare word mappings for valid texts
    all_words = []
    all_word_to_tokens = []
    valid_indices = []
    for idx, text in enumerate(texts):
        words, word_to_tokens = map_subtokens_to_words(
            text, tokenizer, max_length=max_length
        )
        if words:
            all_words.append(words)
            all_word_to_tokens.append(word_to_tokens)
            valid_indices.append(idx)

    if not valid_indices:
        return ([[] for _ in texts], [[] for _ in texts])

    valid_texts = [texts[i] for i in valid_indices]
    
    # Tokenize all valid texts at once
    enc = tokenizer(
        valid_texts, 
        return_tensors="pt", 
        padding=True,
        truncation=max_length is not None, 
        max_length=max_length
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    B, T = input_ids.shape

    # Determine attention module patterns based on model architecture
    model_type = getattr(model.config, "model_type", "bert")
    if "roberta" in model_type:
        target_module_substrings = ["attention.self"] 
        layer_idx_pattern = r'\.layers\.(\d+)\.'
    elif "deberta" in model_type:
        target_module_substrings = ["attention.self", "attention.output", "disentangled_attention"]
        layer_idx_pattern = r'\.layers\.(\d+)\.'
    elif "modernbert" in model_type:
        target_module_substrings = [".attn"]
        layer_idx_pattern = r'\.layers\.(\d+)\.'
    else:
        target_module_substrings = ["attention.self"]
        layer_idx_pattern = r'\.layers\.(\d+)\.'
    fallback_layer_idx_pattern = r'\.(\d+)\.'

    # Store attention tensors with layer indices
    attention_probs_list: List[Tuple[int, torch.Tensor]] = [] 

    # Helper function to extract layer index from module name
    import re
    def get_layer_idx(name: str) -> int:
        match = re.search(layer_idx_pattern, name)
        if match: return int(match.group(1))
        match = re.search(fallback_layer_idx_pattern, name)
        if match: return int(match.group(1))
        return -1

    # Hook function to capture attention outputs
    def hook_fn(module, input, output, module_name: str):
        attn_probs = None
        if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].ndim == 4:
             attn_probs = output[1]
        elif isinstance(output, torch.Tensor) and output.ndim == 4:
             attn_probs = output

        if attn_probs is not None:
             attn_probs.retain_grad()
             layer_idx = get_layer_idx(module_name)
             attention_probs_list.append((layer_idx, attn_probs))
             #print(f"Hooked attention from {module_name} (Layer {layer_idx}), shape: {attn_probs.shape}")

    # Register hooks on attention modules
    hooks = []
    all_target_patterns = set(target_module_substrings + 
                           [".attn", ".attention", "attention.output", "self_attention", 
                            "multihead_attention", "mha", "self_attn"])
    
    registered_hook_names = set()
    for name, module in model.named_modules():
         name_lower = name.lower()
         if any(sub in name_lower for sub in all_target_patterns) and \
            "dropout" not in name_lower and \
            name not in registered_hook_names:
              hook_handle = module.register_forward_hook(
                  lambda m, i, o, n=name: hook_fn(m, i, o, module_name=n)
              )
              hooks.append(hook_handle)
              registered_hook_names.add(name)
              #print(f"Registered attention hook for: {name}")

    if not hooks:
         print(f"Could not register hooks for attention patterns. Trying direct attention output.")
         direct_attention = True
    else:
         direct_attention = False

    # List to store attention × gradient products
    attn_grads = []
    
    try:
        with torch.enable_grad():
            # Forward pass with attention outputs
            outputs = model(**enc, output_attentions=True)
            logits = outputs.logits
            
            # Get predicted classes for all texts
            preds = logits.argmax(dim=1)
            
            # Get scores for predicted classes
            target_scores = logits.gather(1, preds.unsqueeze(-1)).squeeze(-1)
            
            # Backward pass
            model.zero_grad()
            target_scores.sum().backward(retain_graph=True)
            
            # Check if hooks captured attention and gradients
            hook_count = len(attention_probs_list)
            hook_with_grads = sum(1 for _, attn in attention_probs_list if attn.grad is not None)
            
            if hook_with_grads == 0:
                print(f"None of the {hook_count} hooked attention tensors received gradients. Check model architecture.")
                
                # Fall back to direct attention outputs if needed
                if direct_attention or hook_count == 0:
                    if hasattr(outputs, "attentions") and outputs.attentions:
                        print("Attempting to use model's direct attention outputs as fallback")
                        attentions = outputs.attentions
                        fake_grad = torch.ones_like(attentions[0])
                        for layer_idx, layer_attn in enumerate(attentions):
                            attn_grads.append((layer_idx, layer_attn * fake_grad))
            else:
                pass
                #print(f"{hook_with_grads} out of {hook_count} hooked attention tensors received gradients.")

        # Process hooked attention tensors with gradients
        if not direct_attention:
            for layer_idx, attn_probs in attention_probs_list:
                if attn_probs.grad is not None:
                    if attn_probs.grad.shape == attn_probs.shape:
                        grad_x_attn = attn_probs * attn_probs.grad
                        attn_grads.append((layer_idx, grad_x_attn))
                    else:
                        print(f"Layer {layer_idx}: Mismatched shape between attention ({attn_probs.shape}) and gradient ({attn_probs.grad.shape}). Skipping.")
                else:
                    print(f"Layer {layer_idx}: No gradient for attention tensor.")

        # Check if we have attention gradients to process
        if not attn_grads:
            print("No gradients were captured for attention weights via hooks and no fallback available.")
            return ([[] for _ in texts], [[] for _ in texts])

        # Sort by layer index and select tensors
        attn_grads.sort(key=lambda x: x[0])
        sorted_attn_grad_tensors = [t for _, t in attn_grads]

        # Aggregate across layers
        if layer_aggregation == "last" and len(sorted_attn_grad_tensors) > 0:
            combined_attn = sorted_attn_grad_tensors[-1]
        else:
            if not sorted_attn_grad_tensors:
                print("No attention gradients available for aggregation.")
                return ([[] for _ in texts], [[] for _ in texts])
            device = sorted_attn_grad_tensors[0].device
            tensors_to_stack = [t.to(device) for t in sorted_attn_grad_tensors]
            combined_attn = torch.stack(tensors_to_stack).mean(dim=0)
        
        # Average across attention heads
        combined_attn = combined_attn.mean(dim=1)
        
        # Sum across sequence dimension to get token importance
        token_importance = combined_attn.sum(dim=1).detach().cpu().numpy()
        
        # Process token importance into word scores
        batch_tuples: List[List[Tuple[str, float, int]]] = []
        batch_word_scores: List[List[float]] = []

        for b_idx in range(B):
            words = all_words[b_idx]
            word_to_tokens = all_word_to_tokens[b_idx]
            current_token_importance = token_importance[b_idx]

            # Aggregate token-level importance to word-level scores
            word_scores = [0.0] * len(words)
            for w_idx, word in enumerate(words):
                tok_idxs = word_to_tokens.get(w_idx, [])
                valid_tok_idxs = [idx for idx in tok_idxs if idx < len(current_token_importance)]

                if not valid_tok_idxs:
                    score = 0.0
                else:
                    values = current_token_importance[valid_tok_idxs]
                    if values.size == 0:
                         score = 0.0
                    else:
                        score = float(np.sum(values))
                word_scores[w_idx] = score
            
            # Create sorted tuple list (word, score, original_index)
            tuples = [(w, s, i) for i, (w, s) in enumerate(zip(words, word_scores))]
            tuples.sort(key=lambda t: t[1], reverse=True)
            
            batch_tuples.append(tuples)
            batch_word_scores.append(word_scores)

        # Map results back to original order
        final_batch_tuples = [[] for _ in texts]
        final_batch_word_scores = [[] for _ in texts]
        processed_idx = 0
        for orig_idx in range(len(texts)):
            if orig_idx in valid_indices:
                final_batch_tuples[orig_idx] = batch_tuples[processed_idx]
                final_batch_word_scores[orig_idx] = batch_word_scores[processed_idx]
                processed_idx += 1

        return (final_batch_tuples, final_batch_word_scores)
    
    except Exception as e:
        print(f"Error during gradient attention computation: {e}", exc_info=True)
        return ([[] for _ in texts], [[] for _ in texts])
    finally:
        for hook in hooks:
            hook.remove()

@torch.no_grad()
def extract_attention_importance(                    
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
) -> Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]:
    """
    Rank words by attention importance using attention rollout.
    
    Parameters
    ----------
    texts : List[str]
        Input texts to analyze.
    model : AutoModelForSequenceClassification
        Sequence classification model.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    max_length : int, optional
        Maximum sequence length for tokenization.
        
    Returns
    -------
    Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]
        Tuple containing:
        - Words with scores and indices, sorted by attention importance
        - Raw word scores in original order
    """
    if not texts:
        return ([], [])

    device = next(model.parameters()).device
    model = model.to(device)

    # Prepare word mappings for valid texts
    all_words = []
    all_word_to_tokens = []
    valid_indices = []
    for idx, text in enumerate(texts):
        words, word_to_tokens = map_subtokens_to_words(
            text,
            tokenizer,
            max_length=max_length,
        )
        if words:
            all_words.append(words)
            all_word_to_tokens.append(word_to_tokens)
            valid_indices.append(idx)

    if not valid_indices:
        return ([[] for _ in texts], [[] for _ in texts])

    valid_texts = [texts[i] for i in valid_indices]

    # Tokenize all valid texts at once
    enc = tokenizer(
        valid_texts,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
    ).to(device)

    # Get attention weights from model (all layers, all heads)
    attn = torch.stack(model(**enc, output_attentions=True).attentions)
    L, B, H, T, _ = attn.shape

    # Average heads, add residual, row-normalize for all samples at once
    attn = attn.mean(dim=2)  # Average across heads: [L, B, T, T]
    
    # Create identity matrix for residual connections
    eye = torch.eye(T, dtype=attn.dtype, device=attn.device)
    
    # Add residual connections (identity matrix) to each layer and sample
    attn = 0.5 * attn + 0.5 * eye.unsqueeze(0).unsqueeze(0)
    
    # Normalize rows to ensure valid attention weights
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Rollout: recursive matrix multiplication across layers
    # Use last 4 layers for rollout
    num_layers_for_rollout = 4
    actual_num_layers = attn.shape[0]
    start_rollout_idx = max(0, actual_num_layers - num_layers_for_rollout)
    
    layers_to_rollout = attn[start_rollout_idx:]

    if layers_to_rollout.shape[0] == 0:
        print("No layers available for attention rollout. Returning zero importance.")
        # Fallback: return empty/zero results
        final_batch_tuples = [[("", 0.0, 0)] * len(all_words[i]) if all_words[i] else [] for i in range(len(all_words))]
        final_batch_word_scores = [[0.0] * len(all_words[i]) if all_words[i] else [] for i in range(len(all_words))]
        # Map results back to original order
        final_tuples_mapped = [[] for _ in texts]
        final_scores_mapped = [[] for _ in texts]
        processed_idx = 0
        for orig_idx in range(len(texts)):
            if orig_idx in valid_indices:
                final_tuples_mapped[orig_idx] = final_batch_tuples[processed_idx]
                final_scores_mapped[orig_idx] = final_batch_word_scores[processed_idx]
                processed_idx += 1
        return (final_tuples_mapped, final_scores_mapped)

    roll = layers_to_rollout[0]  # Start with the first layer
    
    # Iteratively multiply with each successive layer
    for l_idx in range(1, layers_to_rollout.shape[0]):
        roll = torch.bmm(layers_to_rollout[l_idx], roll)

    # Sum attention across sequence dimension to get token importance
    tok_importance = roll.sum(dim=1)

    # Process token importance into word scores
    batch_tuples: List[List[Tuple[str, float, int]]] = []
    batch_word_scores: List[List[float]] = []
    
    for b_idx in range(B):
        words = all_words[b_idx]
        word_to_tokens = all_word_to_tokens[b_idx]
        current_tok_importance = tok_importance[b_idx]

        # Aggregate token-level importance to word-level scores
        word_scores = [0.0] * len(words)
        for w_idx, word in enumerate(words):
            tok_idxs = word_to_tokens.get(w_idx, [])
            valid_tok_idxs = [ti for ti in tok_idxs if ti < T]

            if not valid_tok_idxs:
                score = 0.0
            else:
                vals = current_tok_importance[valid_tok_idxs]
                if vals.numel() == 0:
                    score = 0.0
                else:
                    score = vals.sum().item()
            word_scores[w_idx] = score
            
        # Create sorted tuple list (word, score, original_index)
        tuples = [(word, score, w_idx) for w_idx, (word, score) in enumerate(zip(words, word_scores))]
        tuples.sort(key=lambda t: t[1], reverse=True)
        
        batch_tuples.append(tuples)
        batch_word_scores.append(word_scores)

    # Map results back to original order
    final_batch_tuples = [[] for _ in texts]
    final_batch_word_scores = [[] for _ in texts]
    processed_idx = 0
    for orig_idx in range(len(texts)):
        if orig_idx in valid_indices:
            final_batch_tuples[orig_idx] = batch_tuples[processed_idx]
            final_batch_word_scores[orig_idx] = batch_word_scores[processed_idx]
            processed_idx += 1

    return (final_batch_tuples, final_batch_word_scores)

def random_word_importance(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
    seed: int | None = None,
) -> Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]:
    """
    Generate random word importance scores as a baseline.
    
    Serves as a control for comparing against sophisticated importance metrics
    like attention and gradient-based saliency.

    Parameters
    ----------
    texts : List[str]
        Input texts to analyze.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    max_length : int, optional
        Maximum sequence length for tokenization.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    Tuple[List[List[Tuple[str, float, int]]], List[List[float]]]
        Tuple containing:
        - Words with uniform scores (1.0) and indices
        - Raw word scores (always 1.0) for all words
    """
    if not texts:
        return ([], [])
        
    # Set random seed for reproducibility
    current_seed = seed if seed is not None else 42
    random.seed(current_seed)
    
    # Process all texts to get words
    all_words = []
    valid_indices = []
    for idx, text in enumerate(texts):
        words = split_into_words(text)
        if words:
            all_words.append(words)
            valid_indices.append(idx)
        else:
            all_words.append([])
            
    # Initialize result structures
    batch_tuples = []
    batch_raw_scores = []
    
    # Process each text's words
    for i, words in enumerate(all_words):
        # Assign uniform importance score of 1.0 to all words
        raw_scores = [1.0] * len(words)
        
        # Create tuples of (word, score, index)
        if not words:
            tuples = []
        else:
            tuples = [(w, 1.0, i) for i, w in enumerate(words)]
        
        batch_tuples.append(tuples)
        batch_raw_scores.append(raw_scores)
    
    return batch_tuples, batch_raw_scores

@torch.no_grad()
def compute_word_sensitivity_batch(
    texts: List[str],
    full_embs: torch.Tensor,
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    selected_indices: List[List[int]],
    *,
    max_length: int | None = None,
    distance_metric: str = "cosine",
    layers_to_average: int = 1,
    perturbation_batch_size: int = 32,
) -> List[List[float]]:
    """
    Compute word sensitivity maps through masking perturbations.

    Takes texts and pre-selected word indices, generates perturbed versions by masking,
    computes embeddings, and calculates distances to measure sensitivity.

    Parameters
    ----------
    texts : List[str]
        Input texts to analyze.
    full_embs : torch.Tensor
        Pre-computed embeddings for the original texts.
    model : AutoModelForSequenceClassification
        Sequence classification model.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    selected_indices : List[List[int]]
        Word indices to perturb for each text.
    max_length : int, optional
        Maximum sequence length for tokenization.
    distance_metric : str, default "cosine"
        Distance metric for measuring embedding changes.
    layers_to_average : int, default 1
        Number of hidden layers to average for embeddings.
    perturbation_batch_size : int, default 32
        Batch size for processing perturbed texts.
        
    Returns
    -------
    List[List[float]]
        Sensitivity values (embedding distances) per word for each text.
    """
    if not texts:
        return []
        
    batch_size = len(texts)
    
    # Ensure data is on the correct device
    device = next(model.parameters()).device
    model = model.to(device)
    full_embs = full_embs.to(device)

    # Validate mask token is available
    mask_token = getattr(tokenizer, "mask_token", None)
    if mask_token is None:
        raise ValueError("Tokenizer lacks mask_token which is required for masking strategy")

    # Split texts into words
    all_words = [split_into_words(t) for t in texts]
    
    # Initialize sensitivity maps with zeros
    sens_maps = [[0.0] * len(w) if w else [0.0] for w in all_words]

    # Generate perturbed texts for all selected indices
    pert_texts, map_back = [], []
    for b_idx, words in enumerate(all_words):
        indices_to_perturb = selected_indices[b_idx]
        for pos in indices_to_perturb:
            if pos >= len(words):
                print(f"Sample {b_idx}: Index {pos} out of bounds for words list (len {len(words)}). Skipping.")
                continue
                
            # Create perturbed version by masking
            new_words = words.copy()
            new_words[pos] = mask_token
                
            # Join the words back into text
            pert_text = " ".join(new_words)
            if not pert_text.strip():
                pert_text = tokenizer.pad_token if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token else "[PAD]"
                print(f"Sample {b_idx}, Pos {pos}: Text became empty after perturbation, using pad token.")

            pert_texts.append(pert_text)
            map_back.append((b_idx, pos))

    if not pert_texts:
        print("No valid perturbations generated based on selected_indices. Returning zero sensitivity maps.")
        return sens_maps

    # Process perturbations in batches
    all_dist: List[float] = []
    for start in range(0, len(pert_texts), perturbation_batch_size):
        sub_pert_texts = pert_texts[start : start + perturbation_batch_size]
        sub_map_back = map_back[start : start + perturbation_batch_size]
        
        # Tokenize perturbed texts
        enc = tokenize_texts(sub_pert_texts, tokenizer, max_length)
        
        # Compute embeddings for perturbed texts
        pert_emb = get_mean_pooled_representation(
            enc["input_ids"], 
            enc["attention_mask"], 
            model, 
            layers_to_average=layers_to_average
        )
        
        # Get original embeddings for comparison
        orig_indices_in_batch = [orig_batch_i for orig_batch_i, _ in sub_map_back]
        orig_emb = full_embs[orig_indices_in_batch]
        
        # Ensure embeddings are on the correct device
        orig_emb = orig_emb.to(device)
        pert_emb = pert_emb.to(device)

        # Compute distance between original and perturbed embeddings
        try:
            dist = compute_embedding_distance(
                orig_emb, 
                pert_emb, 
                distance_metric=distance_metric, 
            )
            all_dist.extend(dist.cpu().tolist())
        except Exception as exc: 
            print("Distance computation failed: %s", exc)
            all_dist.extend([float("nan")] * len(sub_pert_texts)) 

    # Map distances back to their positions in the sensitivity maps
    for (b_idx, pos), d in zip(map_back, all_dist):
        if pos < len(sens_maps[b_idx]):
            sens_maps[b_idx][pos] = d
        else:
             print(f"Sample {b_idx}: Position {pos} out of bounds for sensitivity map (len {len(sens_maps[b_idx])}). Discarding.")

    return sens_maps
