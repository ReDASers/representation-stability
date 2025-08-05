from typing import List,  Dict, Any, Callable
import re
import string

import torch
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    BatchEncoding,
)

# Constants for text processing
WORD_LIKE_PATTERN = re.compile(
    r"[A-Za-z\u00C0-\u017F0-9]+(?:[.\-'][A-Za-z\u00C0-\u017F0-9]+)*"
)
"""Pattern to identify word-like tokens: letters (including accents), digits, and internal punctuation."""

PUNCT_TO_STRIP = "".join(c for c in string.punctuation if c not in {".", "-", "'"})
"""Characters to strip from the beginning and end of token candidates."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_into_words(text: str) -> List[str]:
    """
    Extract word-like tokens from text, preserving internal punctuation.

    Parameters
    ----------
    text : str
        Input string to split into words.

    Returns
    -------
    List[str]
        Word-like substrings extracted from the input text.
    """
    if not text:
        return []

    # Extract candidate substrings and their positions
    candidates = (
        (m.group().strip(PUNCT_TO_STRIP), m.start())
        for m in WORD_LIKE_PATTERN.finditer(text)
    )
    
    # Keep order and drop empty results
    words_with_pos = [(w, pos) for w, pos in candidates if w]
    words_with_pos.sort(key=lambda t: t[1])
    
    return [w for w, _ in words_with_pos]

def _get_word_maps_for_batch(
    original_texts: List[str],
    batch_encoding: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
    max_length: int | None,
) -> List[Dict[str, Any]]:
    """
    Processes a batch of tokenized texts to map tokens to words for each text,
    using pre-computed batch_encoding (including offset_mapping).
    
    Returns a list of dictionaries, each containing:
    - words: List of words for the text
    - token_to_word: Dictionary mapping token indices to word indices
    - original_idx_in_batch: The original index of this text in the batch
    """
    processed_data_for_batch = []

    for i in range(len(original_texts)):
        text = original_texts[i]
        
        # Get words for current text
        current_words = split_into_words(text)
        if not current_words:
            print(f"Text idx {i} in batch ('{text[:50]}...') produced no words. Skipping mapping.")
            continue
            
        try:
            # Get offset mapping for current text in batch
            current_offsets = batch_encoding["offset_mapping"][i].tolist()
            current_input_ids_tensor = batch_encoding["input_ids"][i]
            # Filter out padding tokens before converting to tokens
            current_tokens = tokenizer.convert_ids_to_tokens(current_input_ids_tensor.tolist())
    
            # First pass: Find all word spans directly using character-level search
            word_spans = []
            curr_pos = 0
            for word in current_words:
                word_len = len(word)
                # Find the word in the text
                while curr_pos < len(text):
                    if (curr_pos + word_len <= len(text) and 
                        text[curr_pos:curr_pos+word_len] == word and
                        (curr_pos == 0 or not text[curr_pos-1].isalnum()) and
                        (curr_pos+word_len == len(text) or not text[curr_pos+word_len].isalnum())):
                        word_spans.append((curr_pos, curr_pos + word_len))
                        curr_pos += word_len
                        break
                    curr_pos += 1
                else:
                    # Word not found
                    word_spans.append((-1, -1))
            
            # Create token-to-word mapping using word spans and token offsets
            token_to_word_map = {}
            
            for t_idx, (start_offset, end_offset) in enumerate(current_offsets):
                if start_offset == end_offset == 0:
                    continue
                
                # Skip if token is a special token
                token_str = current_tokens[t_idx]
                if token_str in tokenizer.all_special_tokens:
                    continue
                
                # Find which word this token belongs to by checking overlap with word spans
                for w_idx, (w_start, w_end) in enumerate(word_spans):
                    if w_start == -1:  # Skip invalid word spans
                        continue
                    
                    # Check if token span overlaps with word span
                    # Token is within word boundary or overlaps
                    if (start_offset >= w_start and end_offset <= w_end) or \
                       (start_offset < w_end and end_offset > w_start and min(end_offset, w_end) - max(start_offset, w_start) > 0):
                        token_to_word_map[t_idx] = w_idx
                        break
                        
        except (IndexError, KeyError) as e:
            print(f"Error accessing batch_encoding for text index {i} in _get_word_maps_for_batch: {e}")
            continue
        
        # Only add if there are words and at least some token mapping was found
        if current_words and (not token_to_word_map):
            print(f"Text idx {i} ('{text[:50]}...') resulted in an empty token_to_word map despite having words. May indicate all tokens were special or mapping failed.")
            
        processed_data_for_batch.append({
            "words": current_words,
            "token_to_word": token_to_word_map,
            "original_idx_in_batch": i  # Index in the original texts list
        })
        
    return processed_data_for_batch

@torch.enable_grad() 
def compute_awi_vanilla_gradient(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
    batch_encoding: BatchEncoding = None,
    processed_text_data: List[Dict[str, Any]] = None,
) -> List[torch.Tensor]:
    """
    Compute Adaptive Word Importance (AWI) using Vanilla Gradients.
    Optimized to use single batch tokenization and output one importance score per word.
    
    Args:
        texts: List of text strings to compute AWI for
        model: Classification model to use
        tokenizer: Tokenizer for the model
        max_length: Maximum token length for tokenization
        batch_encoding: Pre-computed batch encoding (optional, for reuse across methods)
        processed_text_data: Pre-computed word maps (optional, for reuse across methods)
        
    Returns:
        List of 1D tensors, each containing importance scores for words in the corresponding text.
        The importance scores measure how much each word contributes to the predicted class.
    """
    if not texts:
        return []
        
    def vanilla_gradient_func(original_embeddings, attention_mask, class_idx):
        """
        Compute gradients of the predicted class score with respect to input.
        This function handles individual text samples for compatibility.
        
        Args:
            original_embeddings: Input embeddings [1, seq_len, emb_dim]
            attention_mask: Attention mask [1, seq_len]
            class_idx: Class index to compute gradients for
            
        Returns:
            Token saliencies [1, seq_len]
        """
        # Clone embeddings for the current class gradient computation
        current_input_embeddings = original_embeddings.clone().requires_grad_(True)

        # Forward pass with current embeddings
        outputs = model(inputs_embeds=current_input_embeddings, attention_mask=attention_mask)
        logits = outputs.logits  # (B_valid, num_classes)

        # Score for the current class_idx for all items in the valid batch
        class_specific_scores = logits[:, class_idx]  # (B_valid,)

        model.zero_grad()
        if current_input_embeddings.grad is not None:
            current_input_embeddings.grad.zero_()

        # Compute gradients of scores w.r.t. current_input_embeddings
        class_specific_scores.sum().backward() 

        if current_input_embeddings.grad is None:
            return None
            
        # Norm of gradients across embedding dimension
        token_saliencies = current_input_embeddings.grad.norm(dim=2)
        
        return token_saliencies
    
    return _compute_awi_base(
        texts,
        model,
        tokenizer,
        "VG",
        vanilla_gradient_func,
        max_length=max_length,
        batch_encoding=batch_encoding,
        processed_text_data=processed_text_data
    )

@torch.enable_grad()
def compute_awi_integrated_gradients(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
    steps: int = 10, # Number of steps for IG
    baseline_type: str = "zero", # "zero", "pad", "random"
    batch_encoding: BatchEncoding = None,
    processed_text_data: List[Dict[str, Any]] = None,
) -> List[torch.Tensor]:
    """
    Compute Adaptive Word Importance (AWI) using Integrated Gradients.
    Optimized to use single batch tokenization.
    
    Args:
        texts: List of text strings to compute AWI for
        model: Classification model to use
        tokenizer: Tokenizer for the model
        max_length: Maximum token length for tokenization
        steps: Number of steps for Integrated Gradients approximation
        baseline_type: Type of baseline to use ("zero", "pad", "random")
        batch_encoding: Pre-computed batch encoding (optional, for reuse across methods)
        processed_text_data: Pre-computed word maps (optional, for reuse across methods)
        
    Returns:
        List of 1D tensors, each containing importance scores for words in the corresponding text.
        The importance scores measure how much each word contributes to the predicted class.
    """
    if not texts:
        return []

    model_device = next(model.parameters()).device
    
    # Handle steps parameter first to avoid errors
    if steps <= 0:
        print("IG steps must be positive. Will use steps=1")
        steps = 1

    def ig_gradient_func(original_embeddings, attention_mask, class_idx):
        """
        Efficient implementation of Integrated Gradients using batched computation.
        Instead of computing gradients for each step individually, this batches
        all steps together for parallel processing.
        
        Args:
            original_embeddings: Input embeddings [B, seq_len, emb_dim]
            attention_mask: Attention mask [B, seq_len]
            class_idx: Class indices tensor [B]
            
        Returns:
            Token saliencies [B, seq_len]
        """
        B, seq_len, emb_dim = original_embeddings.shape
        device = original_embeddings.device
        
        word_embedding_layer = model.get_input_embeddings()
        
        if baseline_type == "zero":
            baseline_embeddings = torch.zeros_like(original_embeddings)
        elif baseline_type == "pad":
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
                tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
            baseline_input_ids = torch.full_like(
                torch.zeros((B, seq_len), dtype=torch.long, device=device), 
                pad_token_id
            )
            baseline_embeddings = word_embedding_layer(baseline_input_ids).detach()
        elif baseline_type == "random":
            baseline_embeddings = torch.randn_like(original_embeddings) * 0.01
        else:
            print(f"Unknown baseline_type '{baseline_type}'. Defaulting to 'zero'.")
            baseline_embeddings = torch.zeros_like(original_embeddings)
            
        alphas = torch.linspace(0.0, 1.0, steps, device=device)
        
        accumulated_gradients = torch.zeros_like(original_embeddings)
        
        for alpha_idx, alpha in enumerate(alphas[:-1]):  # Exclude last point which is just the original
            # Compute midpoint alpha for Riemann approximation
            mid_alpha = (alphas[alpha_idx] + alphas[alpha_idx+1]) / 2
            
            # Compute interpolated embeddings for all inputs
            interpolated_embeddings = baseline_embeddings + mid_alpha * (original_embeddings - baseline_embeddings)
            interpolated_embeddings.requires_grad_(True)
            
            # Forward pass
            outputs = model(inputs_embeds=interpolated_embeddings, attention_mask=attention_mask)
            logits = outputs.logits  # [B, num_classes]
            
            # Get scores for target classes
            batch_indices = torch.arange(B, device=device)
            target_scores = logits[batch_indices, class_idx]
            
            # Backward pass
            model.zero_grad()
            target_scores.sum().backward()
            
            if interpolated_embeddings.grad is None:
                print(f"IG gradient is None at alpha {mid_alpha}. Using zeros.")
                continue
                
            # Accumulate gradients (weighted by step size)
            step_size = alphas[alpha_idx+1] - alphas[alpha_idx]
            accumulated_gradients.add_(interpolated_embeddings.grad * step_size)
            
            # Clear gradients for next iteration
            interpolated_embeddings.grad = None
        
        # Compute attributions
        attributions = (original_embeddings - baseline_embeddings) * accumulated_gradients
        
        # Sum over embedding dimension
        token_saliencies = attributions.sum(dim=2)  # [B, seq_len]
        
        return token_saliencies
    
    return _compute_awi_base(
        texts,
        model,
        tokenizer,
        "IG",
        ig_gradient_func,
        max_length=max_length,
        batch_encoding=batch_encoding,
        processed_text_data=processed_text_data
    )

@torch.enable_grad()
def compute_awi_guided_backpropagation(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
    batch_encoding: BatchEncoding = None,
    processed_text_data: List[Dict[str, Any]] = None,
) -> List[torch.Tensor]:
    """
    Compute Adaptive Word Importance (AWI) using Guided Backpropagation (GBP).
    Optimized to use single batch tokenization and output one importance score per word.
    
    Args:
        texts: List of text strings to compute AWI for
        model: Classification model to use
        tokenizer: Tokenizer for the model
        max_length: Maximum token length for tokenization
        batch_encoding: Pre-computed batch encoding (optional, for reuse across methods)
        processed_text_data: Pre-computed word maps (optional, for reuse across methods)
        
    Returns:
        List of 1D tensors, each containing importance scores for words in the corresponding text.
        The importance scores measure how much each word contributes to the predicted class.
    """
    if not texts:
        return []
    
    def guided_backprop_func(original_embeddings, attention_mask, class_idx):
        """
        Compute gradients of the predicted class score with respect to input,
        keeping only positive gradients (guided backpropagation).
        
        Args:
            original_embeddings: Input embeddings [1, seq_len, emb_dim]
            attention_mask: Attention mask [1, seq_len]
            class_idx: Class index to compute gradients for
            
        Returns:
            Token saliencies [1, seq_len]
        """
        current_input_embeddings = original_embeddings.clone().requires_grad_(True)

        # Forward pass with current embeddings
        outputs = model(inputs_embeds=current_input_embeddings, attention_mask=attention_mask)
        logits = outputs.logits  # (B_valid, num_classes)

        # Score for the current class_idx for all items in the valid batch
        class_specific_scores = logits[:, class_idx]  # (B_valid,)

        model.zero_grad() # Clear gradients from model parameters
        if current_input_embeddings.grad is not None:
            current_input_embeddings.grad.zero_()

        # Compute gradients of scores w.r.t. current_input_embeddings
        class_specific_scores.sum().backward() 

        if current_input_embeddings.grad is None:
            return None
            
        # Apply guided backprop: only keep positive gradients
        positive_gradients = torch.clamp(current_input_embeddings.grad, min=0.0)
        token_saliencies = positive_gradients.norm(dim=2)  # Norm over embedding dimension
        
        return token_saliencies
    
    return _compute_awi_base(
        texts,
        model,
        tokenizer,
        "GBP",
        guided_backprop_func,
        max_length=max_length,
        batch_encoding=batch_encoding,
        processed_text_data=processed_text_data
    )

@torch.enable_grad()
def compute_awi_lrp(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    *,
    max_length: int | None = None,
    batch_encoding: BatchEncoding = None,
    processed_text_data: List[Dict[str, Any]] = None,
) -> List[torch.Tensor]:
    """
    Compute Adaptive Word Importance (AWI) using Layerwise Relevance Propagation (LRP).
    Optimized to use single batch tokenization and output one importance score per word.
    
    Args:
        texts: List of text strings to compute AWI for
        model: Classification model to use
        tokenizer: Tokenizer for the model
        max_length: Maximum token length for tokenization
        batch_encoding: Pre-computed batch encoding (optional, for reuse across methods)
        processed_text_data: Pre-computed word maps (optional, for reuse across methods)
        
    Returns:
        List of 1D tensors, each containing importance scores for words in the corresponding text.
        The importance scores measure how much each word contributes to the predicted class.
    """
    if not texts:
        return []
    
    def lrp_func(original_embeddings, attention_mask, class_idx):
        """
        Compute LRP attributions for the predicted class score.
        
        Args:
            original_embeddings: Input embeddings [1, seq_len, emb_dim]
            attention_mask: Attention mask [1, seq_len]
            class_idx: Class index to compute gradients for
            
        Returns:
            Token saliencies [1, seq_len]
        """
        current_input_embeddings = original_embeddings.clone().requires_grad_(True)
        
        # Reference embeddings
        reference_embeddings = torch.zeros_like(original_embeddings)
        embedding_differences = current_input_embeddings - reference_embeddings

        # Forward pass with current embeddings
        outputs = model(inputs_embeds=current_input_embeddings, attention_mask=attention_mask)
        logits = outputs.logits  # (B_valid, num_classes)

        # Score for the current class_idx
        class_specific_scores = logits[:, class_idx]

        model.zero_grad()
        # For LRP, we need to retain graph if there are shared components
        class_specific_scores.sum().backward(retain_graph=True)

        if current_input_embeddings.grad is None:
            return None
            
        # Compute LRP attributions
        lrp_attributions = embedding_differences * current_input_embeddings.grad
        token_saliencies = lrp_attributions.sum(dim=2) 
        
        return token_saliencies
    
    return _compute_awi_base(
        texts,
        model,
        tokenizer,
        "LRP",
        lrp_func,
        max_length=max_length,
        batch_encoding=batch_encoding,
        processed_text_data=processed_text_data
    )

def _compute_awi_base(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    awi_method_name: str,
    gradient_func: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    *,
    max_length: int | None = None,
    batch_encoding: BatchEncoding = None,
    processed_text_data: List[Dict[str, Any]] = None,
) -> List[torch.Tensor]:
    """
    Base helper function for AWI computation methods to handle common operations.
    
    Args:
        texts: List of text strings to compute AWI for
        model: Classification model to use
        tokenizer: Tokenizer for the model
        awi_method_name: Name of the AWI method (for logging)
        gradient_func: Function that computes the token saliencies for each class
                       Takes (input_embeddings, attention_mask, class_idx) and returns token_saliencies
        max_length: Maximum token length for tokenization
        batch_encoding: Pre-computed batch encoding (optional, for reuse across methods)
        processed_text_data: Pre-computed word maps (optional, for reuse across methods)
        
    Returns:
        List of 1D tensors, where each tensor contains one importance score per word
    """
    if not texts:
        return []

    model_device = next(model.parameters()).device
    model.eval()

    if batch_encoding is None:
        batch_encoding = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_offsets_mapping=True,
        )
    
    if processed_text_data is None:
        processed_text_data = _get_word_maps_for_batch(
            texts,
            batch_encoding,
            tokenizer,
            max_length=max_length
        )
    
    final_awi_result_list: List[torch.Tensor] = [
        torch.empty(0, device=model_device) for _ in range(len(texts))
    ]

    # Filter out texts that didn't yield words or mappings
    valid_processed_data = []
    valid_input_indices_in_batch_encoding = []

    for ptd in processed_text_data:
        if ptd["words"] and ptd["token_to_word"]:
            valid_processed_data.append(ptd)
            valid_input_indices_in_batch_encoding.append(ptd["original_idx_in_batch"])
    
    if not valid_processed_data:
        print(f"{awi_method_name} AWI: No valid texts found after word mapping for this batch.")
        return final_awi_result_list  # All will be empty tensors

    # Filter batch_encoding tensors to only include valid texts
    input_ids = batch_encoding["input_ids"][valid_input_indices_in_batch_encoding].to(model_device)
    attention_mask = batch_encoding["attention_mask"][valid_input_indices_in_batch_encoding].to(model_device)
    
    B_valid = input_ids.shape[0]  # Batch size of effectively processed texts

    # Initialize AWI scores for valid texts - 1D tensors (one score per word)
    current_batch_awi_scores = [
        torch.zeros(len(ptd["words"]), device=model_device) 
        for ptd in valid_processed_data
    ]

    # Get word embeddings layer and compute input embeddings once
    word_embedding_layer = model.get_input_embeddings()
    original_input_embeddings = word_embedding_layer(input_ids).detach()
    
    try:
        # First, get the predicted classes for all texts in a single forward pass
        with torch.no_grad():
            outputs = model(inputs_embeds=original_input_embeddings, attention_mask=attention_mask)
            logits = outputs.logits
            pred_classes = torch.argmax(logits, dim=1)  # [B_valid]
        
        def batch_gradient_adapter(embeddings, attn_mask, pred_class_indices):
            """
            Adapter that routes each input in the batch to its predicted class when computing gradients
            
            Args:
                embeddings: Input embeddings for the batch [B, seq_len, emb_dim]
                attn_mask: Attention mask for the batch [B, seq_len]
                pred_class_indices: Tensor of predicted class indices [B]
            
            Returns:
                Token saliencies [B, seq_len]
            """
            embeddings_with_grad = embeddings.clone().requires_grad_(True)
            
            # Forward pass with these embeddings
            batch_outputs = model(inputs_embeds=embeddings_with_grad, attention_mask=attn_mask)
            batch_logits = batch_outputs.logits  # [B, num_classes]
            
            # Create a one-hot like target that selects the predicted class for each item in batch
            batch_size = pred_class_indices.size(0)
            batch_indices = torch.arange(batch_size, device=pred_class_indices.device)
            
            # Extract the predicted class logits
            selected_logits = batch_logits[batch_indices, pred_class_indices]  # [B]
            
            # Backward pass to compute gradients
            model.zero_grad()
            selected_logits.sum().backward()
            
            if embeddings_with_grad.grad is None:
                print(f"{awi_method_name} AWI: Gradient is None after backward pass. Using zeros.")
                return torch.zeros_like(embeddings[:, :, 0])
            

            if awi_method_name == "VG":
                # For vanilla gradient, take the norm of gradients
                saliencies = embeddings_with_grad.grad.norm(dim=2)  # [B, seq_len]
            elif awi_method_name == "GBP":
                # For guided backprop, only keep positive gradients
                positive_gradients = torch.clamp(embeddings_with_grad.grad, min=0.0)
                saliencies = positive_gradients.norm(dim=2)  # [B, seq_len]
            elif awi_method_name == "LRP":
                # For LRP, compute attributions as input * gradient
                embeddings_diff = embeddings_with_grad - torch.zeros_like(embeddings_with_grad)
                lrp_attributions = embeddings_diff * embeddings_with_grad.grad
                saliencies = lrp_attributions.sum(dim=2)  # [B, seq_len]
            elif awi_method_name == "IG":
                # Integrated Gradients is handled specially in its own function
                # This branch shouldn't be reached for IG
                print(f"{awi_method_name} AWI: Using batch_gradient_adapter which is not ideal for IG method.")
                saliencies = embeddings_with_grad.grad.norm(dim=2)  # [B, seq_len] 
            else:
                # Default to norm of gradients
                saliencies = embeddings_with_grad.grad.norm(dim=2)  # [B, seq_len]
                
            return saliencies
            
        # Compute token saliencies in batch mode
        if awi_method_name == "IG":
            token_saliencies_list = []
            for b_idx in range(B_valid):
                try:
                    # Extract single sample
                    single_embedding = original_input_embeddings[b_idx:b_idx+1]
                    single_attn_mask = attention_mask[b_idx:b_idx+1]
                    single_class = pred_classes[b_idx].item()
                    
                    # Call the original gradient function
                    single_saliencies = gradient_func(single_embedding, single_attn_mask, single_class)
                    token_saliencies_list.append(single_saliencies)
                except Exception as e:
                    print(f"{awi_method_name} AWI: Error in IG computation for sample {b_idx}: {e}")
                    # Add empty tensor as placeholder
                    token_saliencies_list.append(torch.zeros((1, input_ids.shape[1]), device=model_device))
            
            # Stack results into a batch tensor
            if token_saliencies_list:
                token_saliencies = torch.cat(token_saliencies_list, dim=0)  # [B_valid, seq_len]
            else:
                token_saliencies = torch.zeros((B_valid, input_ids.shape[1]), device=model_device)
        else:
            # For other methods, use the batch adapter
            try:
                token_saliencies = batch_gradient_adapter(original_input_embeddings, attention_mask, pred_classes)
            except Exception as e:
                print(f"{awi_method_name} AWI: Error in batch gradient computation: {e}")
                token_saliencies = torch.zeros((B_valid, input_ids.shape[1]), device=model_device)
        
        # Distribute token saliencies to word scores for each text
        for b_idx_valid in range(B_valid):
            current_text_data = valid_processed_data[b_idx_valid]
            token_to_word_map = current_text_data["token_to_word"]
            
            current_token_saliencies = token_saliencies[b_idx_valid]
            
            # Aggregate token saliencies to word saliencies
            for token_idx_in_seq, word_idx_in_text in token_to_word_map.items():
                if token_idx_in_seq < current_token_saliencies.size(0):
                    # Add absolute importance value to the corresponding word
                    current_batch_awi_scores[b_idx_valid][word_idx_in_text] += abs(current_token_saliencies[token_idx_in_seq].item())
                    
    except Exception as e:
        print(f"{awi_method_name} AWI: Error during computation: {e}")
        # Ensure we still have valid outputs even after errors
        for b_idx_valid in range(B_valid):
            current_batch_awi_scores[b_idx_valid] = torch.full_like(
                current_batch_awi_scores[b_idx_valid], float('nan')
            )

    # Place the computed AWI scores into the final_awi_result_list at their original positions
    for i, awi_scores_for_valid_text in enumerate(current_batch_awi_scores):
        original_text_idx = valid_processed_data[i]["original_idx_in_batch"]
        final_awi_result_list[original_text_idx] = awi_scores_for_valid_text
    
    # Clean up any large tensors
    del original_input_embeddings, input_ids, attention_mask
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return final_awi_result_list 