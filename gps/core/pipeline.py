"""Adversarial Detection Experiment Helpers.

This module provides utility functions and classes for running adversarial detection
experiments. It handles data preparation, sensitivity map computation, feature extraction,
and detector training/evaluation.

Key Components:
    - SensitivityResults: Dataclass for storing sensitivity computation results
    - Data preparation and preprocessing functions
    - Sensitivity map computation with various word selection strategies
    - Feature extraction from sensitivity maps
    - Detector training and evaluation pipelines
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification

from .sensitivity import (
    extract_perturbed_word_positions,
    parse_attack_logs,
)
from ..analysis.attribution import (
    extract_attention_importance,
    identify_salient_words,
    identify_salient_words_integrated_gradients,
    compute_gradient_attention,
    random_word_importance,
    split_into_words,
    compute_word_sensitivity_batch,
    precompute_full_embeddings
)
from ..training.detector import (
    train_sensitivity_based_detector,
    evaluate_sensitivity_based_detector
)
from ..utils.text_processing import preprocess_texts_for_model_limit

from dataclasses import dataclass, field

@dataclass
class SensitivityResults:
    """Container for sensitivity map computation results.
    
    Stores the outputs from word-level sensitivity analysis including
    sensitivity maps, word rankings, and timing information.
    
    Attributes:
        maps: Sensitivity values for each word in each text
        selected_indices: Indices of words selected for perturbation
        word_ranks: Word rankings by importance score (index, score) pairs
        raw_importance_maps: Raw importance scores before sensitivity computation
        computation_time_total: Total computation time in seconds
        computation_time_per_sample: Average time per sample in seconds
    """
    maps: List[List[float]] = field(default_factory=list)
    selected_indices: List[List[int]] = field(default_factory=list)
    word_ranks: List[List[Tuple[int, float]]] = field(default_factory=list)
    raw_importance_maps: List[List[float]] = field(default_factory=list)
    computation_time_total: float = 0.0
    computation_time_per_sample: float = 0.0

# =============================================================================
# Global Configuration
# =============================================================================

# Set global random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_attack_log_for_positions(
    log_path: str | None,
    texts: List[str],
    log_prefix: str = "Data"
) -> List[List[int]]:
    """Extract perturbed word positions from attack log files.
    
    Parses attack log files to identify which word positions were modified
    during adversarial attacks. Handles mismatches between log entries and
    text count gracefully.
    
    Args:
        log_path: Path to attack log file, None if no logs available
        texts: Original texts corresponding to log entries
        log_prefix: Prefix for informational logging messages
        
    Returns:
        List of word position lists, one per input text. Empty lists
        are returned for texts without corresponding log entries.
        
    Note:
        If log_path is None or file is missing, returns empty position
        lists for all texts.
    """
    # Initialize empty position lists for all texts
    perturbed_positions: List[List[int]] = [[] for _ in range(len(texts))]

    if not log_path:
        print(f"No attack log path provided for {log_prefix}. Skipping perturbation position extraction.")
        return perturbed_positions

    try:
        logs = parse_attack_logs(log_path)
        num_logs = len(logs)
        num_texts = len(texts)
        
        # Warn if there's a mismatch between texts and log entries
        if num_logs != num_texts:
             print(f"WARNING: Mismatch between number of {log_prefix} texts ({num_texts}) and log entries ({num_logs}). Perturbation info might be incomplete.")

        # Extract positions for each available log entry
        for i in range(min(num_texts, num_logs)):
            orig_bracketed = logs["original_text"].iloc[i]
            perturbed_positions[i] = extract_perturbed_word_positions(orig_bracketed)
        
        processed_count = min(num_texts, num_logs)
        print(f"Loaded perturbation info for {processed_count} {log_prefix.lower()} examples from {log_path}")

    except FileNotFoundError:
        print(f"ERROR: Attack log file not found: {log_path}. Cannot load perturbation positions.")
    except Exception as exc:
        print(f"ERROR: Error parsing {log_prefix.lower()} attack logs from {log_path}: {exc}")
        perturbed_positions = [[] for _ in range(len(texts))]

    return perturbed_positions

def _prepare_experiment_data(
    calibration_data: Dataset,
    test_data: Dataset,
    model: AutoModelForSequenceClassification,
    tokenizer,
    max_length: int,
    cal_attack_logs_path: str | None = None,
    test_attack_logs_path: str | None = None,
):
    """Prepare and preprocess datasets for adversarial detection experiments.
    
    Extracts text pairs, handles truncation to model limits, and parses
    attack logs to extract perturbed word positions. Ensures data consistency
    between original/adversarial text pairs.
    
    Args:
        calibration_data: Dataset containing calibration text pairs
        test_data: Dataset containing test text pairs
        model: Model used for determining truncation limits
        tokenizer: Tokenizer for text preprocessing
        max_length: Maximum sequence length for model input
        cal_attack_logs_path: Optional path to calibration attack logs
        test_attack_logs_path: Optional path to test attack logs
        
    Returns:
        Tuple containing:
            - cal_orig_texts: Processed calibration original texts
            - cal_adv_texts: Processed calibration adversarial texts
            - test_orig_texts: Processed test original texts
            - test_adv_texts: Processed test adversarial texts
            - cal_perturbed_positions: Perturbed positions for calibration
            - test_perturbed_positions: Perturbed positions for test
    """
    # Extract text columns from datasets
    cal_orig_texts = calibration_data["original_text"]
    cal_adv_texts = calibration_data["adversarial_text"]
    test_orig_texts = test_data["original_text"]
    test_adv_texts = test_data["adversarial_text"]

    # Convert to DataFrames for consistent handling
    cal_df = pd.DataFrame({"original_text": cal_orig_texts, "adversarial_text": cal_adv_texts})
    test_df = pd.DataFrame({"original_text": test_orig_texts, "adversarial_text": test_adv_texts})

    # Convert back to lists for further processing
    cal_orig_texts = cal_df["original_text"].tolist()
    cal_adv_texts = cal_df["adversarial_text"].tolist()
    test_orig_texts = test_df["original_text"].tolist()
    test_adv_texts = test_df["adversarial_text"].tolist()

    # Truncate texts to model's maximum sequence length
    cal_orig_texts, cal_orig_trunc = preprocess_texts_for_model_limit(cal_orig_texts, model, tokenizer, max_length)
    cal_adv_texts, cal_adv_trunc = preprocess_texts_for_model_limit(cal_adv_texts, model, tokenizer, max_length)
    test_orig_texts, test_orig_trunc = preprocess_texts_for_model_limit(test_orig_texts, model, tokenizer, max_length)
    test_adv_texts, test_adv_trunc = preprocess_texts_for_model_limit(test_adv_texts, model, tokenizer, max_length)

    # Parse attack logs to identify which words were perturbed
    print("Parsing attack logs to extract perturbed word positions...")

    cal_perturbed_positions = _parse_attack_log_for_positions(
        log_path=cal_attack_logs_path,
        texts=cal_orig_texts,
        log_prefix="Calibration"
    )

    test_perturbed_positions = _parse_attack_log_for_positions(
        log_path=test_attack_logs_path,
        texts=test_orig_texts,
        log_prefix="Test"
    )

    return (
        cal_orig_texts,
        cal_adv_texts,
        test_orig_texts,
        test_adv_texts,
        cal_perturbed_positions,
        test_perturbed_positions,
    )

def _compute_embeddings(
    cal_orig_texts: List[str],
    cal_adv_texts: List[str],
    test_orig_texts: List[str],
    test_adv_texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer,
    max_length: int,
    batch_size: int = 16,
    layers_to_average: int = 1,
):
    """Compute contextualized embeddings for all text datasets.
    
    Generates embeddings from the specified model layers for both original
    and adversarial texts across calibration and test sets. Uses batched
    processing for memory efficiency.
    
    Args:
        cal_orig_texts: Original calibration texts
        cal_adv_texts: Adversarial calibration texts
        test_orig_texts: Original test texts
        test_adv_texts: Adversarial test texts
        model: Transformer model for embedding computation
        tokenizer: Tokenizer for text preprocessing
        max_length: Maximum sequence length for model input
        batch_size: Batch size for processing (default: 16)
        layers_to_average: Number of top layers to average (default: 1)
        
    Returns:
        Tuple containing embeddings for:
            (cal_orig_emb, cal_adv_emb, test_orig_emb, test_adv_emb)
    """
    device = next(model.parameters()).device
    print(f"Computing embeddings using device: {device}")
    print("Computing full embeddings...")

    # Compute embeddings for calibration original texts
    cal_orig_emb = precompute_full_embeddings(
        cal_orig_texts,
        model,
        tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        layers_to_average=layers_to_average,
    )
    # Compute embeddings for calibration adversarial texts
    cal_adv_emb = precompute_full_embeddings(
        cal_adv_texts,
        model,
        tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        layers_to_average=layers_to_average,
    )
    # Compute embeddings for test original texts
    test_orig_emb = precompute_full_embeddings(
        test_orig_texts,
        model,
        tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        layers_to_average=layers_to_average,
    )
    # Compute embeddings for test adversarial texts
    test_adv_emb = precompute_full_embeddings(
        test_adv_texts,
        model,
        tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        layers_to_average=layers_to_average,
    )

    return cal_orig_emb, cal_adv_emb, test_orig_emb, test_adv_emb

def _compute_sensitivity_for_dataset(
    texts: List[str],
    full_embs: torch.Tensor,
    model: AutoModelForSequenceClassification,
    tokenizer,
    distance_metric: str,
    word_selection_strategy: str = "all",
    strategy_params: Dict[str, Any] = None,
    dataset_name: str = "Dataset",
    device: torch.device = None,
    max_length: int = None,
    main_batch_size: int = 16,
    perturb_batch_size: int = 64,
    layers_to_average: int = 1,
    use_importance_channel: bool = False
) -> SensitivityResults:
    """Compute word-level sensitivity maps for a single dataset.

    Performs a two-step process:
    1. Calculate word importance using the specified selection strategy
    2. Compute sensitivity by perturbing selected words and measuring embedding distance
    
    Processing is done in batches for memory efficiency.

    Args:
        texts: Input texts to analyze
        full_embs: Pre-computed embeddings for the texts
        model: Transformer model for importance and sensitivity computation
        tokenizer: Tokenizer for text preprocessing
        distance_metric: Distance metric for measuring embedding changes
        word_selection_strategy: Method for selecting important words
            ("attention", "saliency", "gradient_attention", "integrated_gradients", "random", "all")
        strategy_params: Parameters specific to the selection strategy
        dataset_name: Descriptive name for logging
        device: Computation device, auto-detected if None
        max_length: Maximum sequence length for model input
        main_batch_size: Batch size for main processing
        perturb_batch_size: Batch size for perturbation operations
        layers_to_average: Number of model layers to average for embeddings
        use_importance_channel: Whether to store raw importance scores

    Returns:
        SensitivityResults containing sensitivity maps, word rankings,
        selected indices, and timing information.
    """
    results = SensitivityResults()
    
    # Early return if no texts provided
    if not texts:
        print(f"No texts provided for {dataset_name}. Skipping sensitivity computation.")
        return results

    # Set up computation device
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    full_embs = full_embs.to(device)

    start_time = time.time()

    # Process texts in batches for memory efficiency
    for i in tqdm(range(0, len(texts), main_batch_size),
                desc=f"{dataset_name} batches ({word_selection_strategy}, {distance_metric})",
                leave=False):
        batch_end = min(i + main_batch_size, len(texts))
        batch_texts = texts[i:batch_end]
        batch_embs = full_embs[i:batch_end]

        if not batch_texts:
            continue

        # Step 1: Compute word importance/selection for the current batch
        batch_tuples: List[List[Tuple[str, float, int]]] = [[] for _ in batch_texts]
        batch_raw_scores: List[List[float]] = [[] for _ in batch_texts]

        current_strategy_params = strategy_params or {}

        # Apply the appropriate word selection strategy
        if word_selection_strategy == "attention":
            batch_tuples, batch_raw_scores = extract_attention_importance(
                batch_texts, model, tokenizer,
            )
        elif word_selection_strategy == "saliency":
            batch_tuples, batch_raw_scores = identify_salient_words(
                batch_texts, model, tokenizer,
            )
        elif word_selection_strategy == "integrated_gradients":
            steps = current_strategy_params.get("steps", 100)
            baseline_type = current_strategy_params.get("baseline_type", "zero")
            batch_tuples, batch_raw_scores = identify_salient_words_integrated_gradients(
                batch_texts, model, tokenizer,
                steps=steps, baseline_type=baseline_type
            )
        elif word_selection_strategy == "gradient_attention":
            batch_tuples, batch_raw_scores = compute_gradient_attention(
                batch_texts, model, tokenizer,
            )
        elif word_selection_strategy == "random":
            seed = current_strategy_params.get("seed")
            batch_tuples, batch_raw_scores = random_word_importance(
                batch_texts, tokenizer,
                max_length=max_length, seed=seed
            )
        elif word_selection_strategy == "all":
            # Select all words with zero importance scores
            for idx, text in enumerate(batch_texts):
                words = split_into_words(text)
                batch_tuples[idx] = [(w, 0.0, pos) for pos, w in enumerate(words)]
                batch_raw_scores[idx] = [0.0] * len(words)
        else:
            print(f"ERROR: Unknown word_selection_strategy: {word_selection_strategy}")
            continue

        # Step 2: Derive selected indices and rank lists from importance results
        batch_selected_indices: List[List[int]] = []
        batch_rank_lists: List[List[Tuple[int, float]]] = []
        
        # Extract top_n parameter for word selection strategies
        top_n = current_strategy_params.get("top_n") if word_selection_strategy != "all" else None
        
        # Set random seed for reproducible random selection
        if word_selection_strategy == "random":
            seed = current_strategy_params.get("seed")
            current_seed = seed if seed is not None else 42
            random.seed(current_seed)

        # Process each text in the batch to extract indices and rankings
        for i, full_tuples_for_text in enumerate(batch_tuples):
            # Create rank list (index, score) - already sorted by importance score
            rank_list_for_text = [(idx, score) for _, score, idx in full_tuples_for_text]
            batch_rank_lists.append(rank_list_for_text)

            # Determine which word indices to select for perturbation
            selected_indices_for_text = []
            if word_selection_strategy == "all":
                # Select all available words
                selected_indices_for_text = [idx for _, _, idx in full_tuples_for_text]
            elif word_selection_strategy == "random":
                # Randomly select top_n words
                num_words = len(full_tuples_for_text)
                if top_n is not None and top_n < num_words:
                     random.seed(current_seed)
                     selected_indices_for_text = sorted(random.sample(range(num_words), top_n))
                else:
                     selected_indices_for_text = list(range(num_words))
            else:
                # For importance-based strategies (attention, saliency, gradients)
                if top_n is not None:
                    # Select top_n most important words
                    selected_indices_for_text = [idx for _, _, idx in full_tuples_for_text[:top_n]]
                else:
                    # Select all words if no top_n specified
                    selected_indices_for_text = [idx for _, _, idx in full_tuples_for_text]
            
            # Sort indices for consistent processing order
            batch_selected_indices.append(sorted(selected_indices_for_text))

        # Step 3: Compute sensitivity maps using selected word indices
        maps_batch = compute_word_sensitivity_batch(
            texts=batch_texts,
            full_embs=batch_embs,
            model=model,
            tokenizer=tokenizer,
            selected_indices=batch_selected_indices,
            max_length=max_length,
            distance_metric=distance_metric,
            layers_to_average=layers_to_average,
            perturbation_batch_size=perturb_batch_size
        )

        # Collect and store results for this batch
        results.maps.extend(maps_batch)
        results.selected_indices.extend(batch_selected_indices)
        results.word_ranks.extend(batch_rank_lists)

        # Store raw importance scores if requested
        if use_importance_channel:
             results.raw_importance_maps.extend(batch_raw_scores)

    # Record timing information
    end_time = time.time()
    results.computation_time_total = end_time - start_time
    results.computation_time_per_sample = results.computation_time_total / len(texts) if texts else 0
    print(f"{dataset_name} sensitivity computation: {results.computation_time_total:.2f}s total ({results.computation_time_per_sample:.4f}s/sample)")

    return results

def _compute_sensitivity_maps(
    cal_orig_texts: List[str],
    cal_adv_texts: List[str],
    test_orig_texts: List[str],
    test_adv_texts: List[str],
    cal_orig_emb: torch.Tensor,
    cal_adv_emb: torch.Tensor,
    test_orig_emb: torch.Tensor,
    test_adv_emb: torch.Tensor,
    model: AutoModelForSequenceClassification,
    tokenizer,
    distance_metric: str,
    # Word selection strategy flags
    use_attention: bool = False,
    use_saliency: bool = False,
    use_random: bool = False,
    use_gradient_attention: bool = False,
    use_integrated_gradients: bool = False,
    # Strategy-specific parameters
    attention_top_n: int = 5,
    gradient_attention_top_n: int = 5,
    integrated_gradients_top_n: int = 5,
    integrated_gradients_steps: int = 100,
    integrated_gradients_baseline_type: str = "zero",
    saliency_top_n: int = 5,
    random_top_n: int = 5,
    random_seed: int = 42,
    # Processing parameters
    max_length: int = None,
    main_batch_size: int = 16,
    perturb_batch_size: int = 64,
    layers_to_average: int = 1,
    use_importance_channel: bool = False,
) -> Tuple[SensitivityResults, SensitivityResults, SensitivityResults, SensitivityResults, str]:
    """Compute word-level sensitivity maps for all text datasets.

    Orchestrates sensitivity map computation across calibration and test datasets
    for both original and adversarial texts. Automatically selects the appropriate
    word selection strategy based on enabled flags with priority ordering.

    Strategy Priority (if multiple enabled):
        gradient_attention > integrated_gradients > attention > saliency > random

    Args:
        cal_orig_texts: Original calibration texts
        cal_adv_texts: Adversarial calibration texts
        test_orig_texts: Original test texts
        test_adv_texts: Adversarial test texts
        cal_orig_emb: Pre-computed embeddings for calibration original texts
        cal_adv_emb: Pre-computed embeddings for calibration adversarial texts
        test_orig_emb: Pre-computed embeddings for test original texts
        test_adv_emb: Pre-computed embeddings for test adversarial texts
        model: Transformer model for sensitivity computation
        tokenizer: Tokenizer for text preprocessing
        distance_metric: Distance metric for measuring embedding changes
        use_attention: Enable attention-based word selection
        use_saliency: Enable gradient-based saliency word selection
        use_random: Enable random word selection
        use_gradient_attention: Enable gradient×attention word selection
        use_integrated_gradients: Enable integrated gradients word selection
        attention_top_n: Number of top attention words to select
        gradient_attention_top_n: Number of top gradient×attention words
        integrated_gradients_top_n: Number of top integrated gradient words
        integrated_gradients_steps: Integration steps for integrated gradients
        integrated_gradients_baseline_type: Baseline type ("zero" or "pad")
        saliency_top_n: Number of top saliency words to select
        random_top_n: Number of random words to select
        random_seed: Random seed for reproducible random selection
        max_length: Maximum sequence length for model input
        main_batch_size: Batch size for main processing
        perturb_batch_size: Batch size for perturbation operations
        layers_to_average: Number of model layers to average for embeddings
        use_importance_channel: Whether to store raw importance scores

    Returns:
        Tuple containing:
            - cal_orig_results: Sensitivity results for calibration original
            - cal_adv_results: Sensitivity results for calibration adversarial
            - test_orig_results: Sensitivity results for test original
            - test_adv_results: Sensitivity results for test adversarial
            - word_selection_strategy: The strategy that was actually used
    """

    # Determine word selection strategy based on priority order
    enabled_strategies = [name for name, flag in [
        ("gradient_attention", use_gradient_attention),
        ("integrated_gradients", use_integrated_gradients),
        ("attention", use_attention), 
        ("saliency", use_saliency), 
        ("random", use_random)
    ] if flag]
    
    # Warn if multiple strategies are enabled
    if len(enabled_strategies) > 1:
        print(
            f"WARNING: Multiple word selection strategies enabled: {', '.join(enabled_strategies)}. "
            f"Using priority order: gradient_attention > integrated_gradients > attention > saliency > random. "
            f"Selected strategy: '{enabled_strategies[0]}'"
        )

    # Set default strategy and parameters
    word_selection_strategy = "all"  # Default: analyze all words
    strategy_params = {}
    
    # Configure strategy based on priority order
    if use_gradient_attention:  # Highest priority
        word_selection_strategy = "gradient_attention"
        strategy_params = {'top_n': gradient_attention_top_n}
        print(f"Using gradient_attention strategy with top_n={gradient_attention_top_n}")
    elif use_integrated_gradients:  # Second priority
        word_selection_strategy = "integrated_gradients"
        strategy_params = {
            'top_n': integrated_gradients_top_n,
            'steps': integrated_gradients_steps,
            'baseline_type': integrated_gradients_baseline_type
        }
        print(f"Using integrated_gradients strategy: top_n={integrated_gradients_top_n}, "
              f"steps={integrated_gradients_steps}, baseline='{integrated_gradients_baseline_type}'")
    elif use_attention:  # Third priority
        word_selection_strategy = "attention"
        strategy_params = {'top_n': attention_top_n}
        print(f"Using attention strategy with top_n={attention_top_n}")
    elif use_saliency:
        word_selection_strategy = "saliency"
        strategy_params = {'top_n': saliency_top_n}
        print(f"Using saliency strategy with top_n={saliency_top_n}")
    elif use_random:
        word_selection_strategy = "random"
        strategy_params = {'top_n': random_top_n, 'seed': random_seed}
        print(f"Using random strategy with top_n={random_top_n}, seed={random_seed}")
    else:
        print("No word selection strategy specified. Using 'all' strategy (analyzing all words)")

    print(f"Computing word-level sensitivity maps using '{word_selection_strategy}' strategy with {distance_metric} distance...")

    # Compute sensitivity maps for each dataset
    cal_orig_results = _compute_sensitivity_for_dataset(
        texts=cal_orig_texts,
        full_embs=cal_orig_emb,
        model=model,
        tokenizer=tokenizer,
        distance_metric=distance_metric,
        word_selection_strategy=word_selection_strategy,
        strategy_params=strategy_params,
        dataset_name="Calibration Original",
        max_length=max_length,
        main_batch_size=main_batch_size,
        perturb_batch_size=perturb_batch_size,
        layers_to_average=layers_to_average,
        use_importance_channel=use_importance_channel
    )

    cal_adv_results = _compute_sensitivity_for_dataset(
        texts=cal_adv_texts,
        full_embs=cal_adv_emb,
        model=model,
        tokenizer=tokenizer,
        distance_metric=distance_metric,
        word_selection_strategy=word_selection_strategy,
        strategy_params=strategy_params,
        dataset_name="Calibration Adversarial",
        max_length=max_length,
        main_batch_size=main_batch_size,
        perturb_batch_size=perturb_batch_size,
        layers_to_average=layers_to_average,
        use_importance_channel=use_importance_channel
    )

    test_orig_results = _compute_sensitivity_for_dataset(
        texts=test_orig_texts,
        full_embs=test_orig_emb,
        model=model,
        tokenizer=tokenizer,
        distance_metric=distance_metric,
        word_selection_strategy=word_selection_strategy,
        strategy_params=strategy_params,
        dataset_name="Test Original",
        max_length=max_length,
        main_batch_size=main_batch_size,
        perturb_batch_size=perturb_batch_size,
        layers_to_average=layers_to_average,
        use_importance_channel=use_importance_channel
    )

    test_adv_results = _compute_sensitivity_for_dataset(
        texts=test_adv_texts,
        full_embs=test_adv_emb,
        model=model,
        tokenizer=tokenizer,
        distance_metric=distance_metric,
        word_selection_strategy=word_selection_strategy,
        strategy_params=strategy_params,
        dataset_name="Test Adversarial",
        max_length=max_length,
        main_batch_size=main_batch_size,
        perturb_batch_size=perturb_batch_size,
        layers_to_average=layers_to_average,
        use_importance_channel=use_importance_channel
    )

    return (
        cal_orig_results,
        cal_adv_results,
        test_orig_results,
        test_adv_results,
        word_selection_strategy
    )

def _extract_and_process_features(
    cal_orig_sensitivity: SensitivityResults,
    cal_adv_sensitivity: SensitivityResults,
    test_orig_sensitivity: SensitivityResults,
    test_adv_sensitivity: SensitivityResults,
    cal_orig_emb: torch.Tensor,
    cal_adv_emb: torch.Tensor,
    test_orig_emb: torch.Tensor,
    test_adv_emb: torch.Tensor,
    cal_orig_imp_maps: List[List[float]],
    cal_adv_imp_maps: List[List[float]],
    test_orig_imp_maps: List[List[float]],
    test_adv_imp_maps: List[List[float]],
    use_importance_channel: bool = False,
    filter_importance_scores: bool = True,
    # Strategy flags and parameters for determining top_n
    use_gradient_attention: bool = False,
    use_integrated_gradients: bool = False,
    use_attention: bool = False,
    use_saliency: bool = False,
    use_random: bool = False,
    gradient_attention_top_n: int = 10,
    integrated_gradients_top_n: int = 10,
    attention_top_n: int = 10,
    saliency_top_n: int = 10,
    random_top_n: int = 10,
):
    """Extract variable-length sequence features from sensitivity maps.
    
    Converts word-level sensitivity maps into feature sequences suitable for
    neural network training. Supports both single-channel (sensitivity only)
    and dual-channel (sensitivity + importance) feature extraction.
    
    Preserves original sequence lengths for use with variable-length sequence
    models (e.g., RNNs with padding/masking).

    Args:
        cal_orig_sensitivity: Sensitivity results for calibration original texts
        cal_adv_sensitivity: Sensitivity results for calibration adversarial texts
        test_orig_sensitivity: Sensitivity results for test original texts
        test_adv_sensitivity: Sensitivity results for test adversarial texts
        cal_orig_emb: Embeddings for calibration original texts (unused)
        cal_adv_emb: Embeddings for calibration adversarial texts (unused)
        test_orig_emb: Embeddings for test original texts (unused)
        test_adv_emb: Embeddings for test adversarial texts (unused)
        cal_orig_imp_maps: Raw importance scores for calibration original texts
        cal_adv_imp_maps: Raw importance scores for calibration adversarial texts
        test_orig_imp_maps: Raw importance scores for test original texts
        test_adv_imp_maps: Raw importance scores for test adversarial texts
        use_importance_channel: Whether to include importance as second channel
        filter_importance_scores: Whether to filter importance to top_n positions
        use_gradient_attention: Whether gradient×attention strategy is active
        use_integrated_gradients: Whether integrated gradients strategy is active
        use_attention: Whether attention strategy is active
        use_saliency: Whether saliency strategy is active
        use_random: Whether random strategy is active
        gradient_attention_top_n: Top N words for gradient×attention
        integrated_gradients_top_n: Top N words for integrated gradients
        attention_top_n: Top N words for attention
        saliency_top_n: Top N words for saliency
        random_top_n: Top N words for random selection

    Returns:
        Tuple of feature lists for (cal_orig, cal_adv, test_orig, test_adv).
        
        Single-channel mode: Each feature is a list of sensitivity values
        Dual-channel mode: Each feature is a numpy array of shape [seq_len, 2]
        where channel 0 is sensitivity and channel 1 is importance.
    """
    print("Extracting sequence features from word-level sensitivity maps...")
    
    # Determine top_n parameter based on active selection strategy
    top_n = None
    if use_gradient_attention:
        top_n = gradient_attention_top_n
        print(f"Extracting top {top_n} gradient×attention-based features")
    elif use_integrated_gradients:
        top_n = integrated_gradients_top_n
        print(f"Extracting top {top_n} integrated-gradients-based features")
    elif use_attention:
        top_n = attention_top_n
        print(f"Extracting top {top_n} attention-based features")
    elif use_saliency:
        top_n = saliency_top_n
        print(f"Extracting top {top_n} saliency-based features")
    elif use_random:
        top_n = random_top_n
        print(f"Extracting top {top_n} randomly-selected features")
    else:
        print("Extracting all available sensitivity features (no selection strategy)")
                
    # Extract features in single-channel or dual-channel mode
    if use_importance_channel:
        try:
            from .sensitivity import extract_sequential_features
        except ImportError:
            print("ERROR: extract_sequential_features not found - check import path")
            return [], [], [], []
            
        if filter_importance_scores:
            print(f"Mode: 2-channel (Sensitivity + Filtered Importance), top_n={top_n}")
        else:
            print(f"Mode: 2-channel (Sensitivity + Full Importance), sensitivity top_n={top_n}")
            
        cal_orig_features = extract_sequential_features(
            cal_orig_sensitivity.maps, cal_orig_imp_maps, top_n=top_n, 
            filter_importance_scores=filter_importance_scores
        )
        cal_adv_features = extract_sequential_features(
            cal_adv_sensitivity.maps, cal_adv_imp_maps, top_n=top_n,
            filter_importance_scores=filter_importance_scores
        )
        test_orig_features = extract_sequential_features(
            test_orig_sensitivity.maps, test_orig_imp_maps, top_n=top_n,
            filter_importance_scores=filter_importance_scores
        )
        test_adv_features = extract_sequential_features(
            test_adv_sensitivity.maps, test_adv_imp_maps, top_n=top_n,
            filter_importance_scores=filter_importance_scores
        )
        
        if filter_importance_scores:
            print("Both sensitivity and importance filtered to top_n positions")
        else:
            print("Sensitivity filtered to top_n, importance uses all positions")
    else:
        try:
            from .sensitivity import extract_raw_sensitivity_values
        except ImportError:
            print("ERROR: extract_raw_sensitivity_values not found - check import path")
            return [], [], [], []
            
        print(f"Mode: 1-channel (Sensitivity only), top_n={top_n}")
        cal_orig_features = extract_raw_sensitivity_values(cal_orig_sensitivity.maps, top_n=top_n)
        cal_adv_features = extract_raw_sensitivity_values(cal_adv_sensitivity.maps, top_n=top_n)
        test_orig_features = extract_raw_sensitivity_values(test_orig_sensitivity.maps, top_n=top_n)
        test_adv_features = extract_raw_sensitivity_values(test_adv_sensitivity.maps, top_n=top_n)
    
    # Log sequence length statistics
    cal_orig_lengths = [f.shape[0] for f in cal_orig_features]
    cal_adv_lengths = [f.shape[0] for f in cal_adv_features]
    test_orig_lengths = [f.shape[0] for f in test_orig_features]
    test_adv_lengths = [f.shape[0] for f in test_adv_features]
    
    def format_stats(lengths, name):
        if lengths:
            return f"{name}: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}"
        return f"{name}: no sequences"
    
    print(format_stats(cal_orig_lengths, "Cal-orig lengths"))
    print(format_stats(cal_adv_lengths, "Cal-adv lengths"))
    print(format_stats(test_orig_lengths, "Test-orig lengths"))
    print(format_stats(test_adv_lengths, "Test-adv lengths"))
    
    # Calculate and log sparsity statistics
    if use_importance_channel and cal_orig_features and isinstance(cal_orig_features[0], np.ndarray):
        # Dual-channel: count non-zeros in sensitivity channel (channel 0)
        cal_orig_nonzeros = sum(np.count_nonzero(f[:, 0]) for f in cal_orig_features)
        cal_orig_total = sum(f.shape[0] for f in cal_orig_features)
        cal_adv_nonzeros = sum(np.count_nonzero(f[:, 0]) for f in cal_adv_features)
        cal_adv_total = sum(f.shape[0] for f in cal_adv_features)
    else:
        # Single-channel: count non-zero values directly
        cal_orig_nonzeros = sum(sum(1 for val in seq if val != 0) for seq in cal_orig_features)
        cal_orig_total = sum(len(seq) for seq in cal_orig_features)
        cal_adv_nonzeros = sum(sum(1 for val in seq if val != 0) for seq in cal_adv_features)
        cal_adv_total = sum(len(seq) for seq in cal_adv_features)
    
    nonzero_orig_pct = cal_orig_nonzeros / cal_orig_total if cal_orig_total > 0 else 0
    nonzero_adv_pct = cal_adv_nonzeros / cal_adv_total if cal_adv_total > 0 else 0
    print(f"Feature sparsity: {nonzero_orig_pct:.2%} non-zero (original), {nonzero_adv_pct:.2%} non-zero (adversarial)")
    
    # Report maximum sequence length across all datasets
    all_lengths = cal_orig_lengths + cal_adv_lengths + test_orig_lengths + test_adv_lengths
    max_seq_len = max(all_lengths) if all_lengths else 0
    print(f"Variable-length sequences: maximum length = {max_seq_len}")
    
    print("Feature extraction complete.")
    return cal_orig_features, cal_adv_features, test_orig_features, test_adv_features

def _train_evaluate_detectors(
    cal_orig_features: np.ndarray,
    cal_adv_features: np.ndarray,
    test_orig_features: np.ndarray,
    test_adv_features: np.ndarray,
    cal_orig_sensitivity: SensitivityResults, 
    cal_adv_sensitivity: SensitivityResults,  
    test_orig_sensitivity: SensitivityResults, 
    test_adv_sensitivity: SensitivityResults, 
    output_dir: str,
    distance_metric: str,
    detection_methods: List[str] = None,
    random_seed: int = None,
):
    """Train and evaluate adversarial detection models.
    
    Uses extracted sequence features to train neural network detectors and
    evaluates their performance on test data. Aggregates timing information
    from sensitivity computation stages.

    Args:
        cal_orig_features: Feature sequences for calibration original texts
        cal_adv_features: Feature sequences for calibration adversarial texts
        test_orig_features: Feature sequences for test original texts
        test_adv_features: Feature sequences for test adversarial texts
        cal_orig_sensitivity: Sensitivity computation results for calibration original
        cal_adv_sensitivity: Sensitivity computation results for calibration adversarial
        test_orig_sensitivity: Sensitivity computation results for test original
        test_adv_sensitivity: Sensitivity computation results for test adversarial
        output_dir: Directory to save detection results
        distance_metric: Distance metric used in sensitivity computation
        detection_methods: List of detection models to train (default: ["bilstm"])
        random_seed: Random seed for reproducible training (default: uses global SEED)

    Returns:
        Tuple containing:
            - results_df: DataFrame with detection performance metrics
            - first_result: Dictionary with metrics from the first detection method
    """
    # Configure random seed for reproducible training
    if random_seed is None:
        random_seed = SEED
    print(f"Training with random seed: {random_seed}")
    
    # Set default detection methods if none provided
    if detection_methods is None:
        detection_methods = ["bilstm"]
    print(f"Detection methods: {', '.join(detection_methods)}")
    
    # Storage for all detection results
    all_results = []
    
    # Aggregate timing information from sensitivity computation stages
    total_cal_time = cal_orig_sensitivity.computation_time_total + cal_adv_sensitivity.computation_time_total
    total_test_time = test_orig_sensitivity.computation_time_total + test_adv_sensitivity.computation_time_total
    
    # Calculate per-sample timing statistics
    num_cal_samples = len(cal_orig_features) + len(cal_adv_features)
    num_test_samples = len(test_orig_features) + len(test_adv_features)
    avg_cal_time_per_sample = total_cal_time / num_cal_samples if num_cal_samples > 0 else 0
    avg_test_time_per_sample = total_test_time / num_test_samples if num_test_samples > 0 else 0

    timing_info = {
        "cal_time_total_s": total_cal_time,
        "test_time_total_s": total_test_time,
        "cal_time_per_sample_s": avg_cal_time_per_sample,
        "test_time_per_sample_s": avg_test_time_per_sample,
    }

    # Train and evaluate each detection method
    for method in detection_methods:
        print(f"\nTraining {method} detector...")
        
        # Train detector model using calibration data
        detector_models = train_sensitivity_based_detector(
            cal_orig_features, 
            cal_adv_features, 
            method=method,
            val_split=0.1,
            random_state=random_seed
        )
        
        # Evaluate detector on test data
        print(f"Evaluating {method} detector...")
        metrics = evaluate_sensitivity_based_detector(
            test_orig_features, 
            test_adv_features, 
            detector_models
        )
        
        # Add metadata to metrics
        metrics.update({
            "detection_method": method,
            "distance_metric": distance_metric,
            "analysis_type": "sequence",
            "random_seed": random_seed,
        })
        
        # Include timing information from sensitivity computation
        metrics.update(timing_info)
        
        # Store results
        all_results.append(metrics)
        
        # Log key performance metrics
        print(f"Performance metrics for {method}:")
        for metric_name in ["accuracy", "precision", "recall", "f1", "auc"]:
            if metric_name in metrics:
                print(f"  {metric_name.capitalize()}: {metrics[metric_name]:.4f}")
    
    # Create results DataFrame and save to file
    results_df = pd.DataFrame(all_results)
    
    results_file = "detection_results_sequence.csv"
    results_path = os.path.join(output_dir, results_file)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Return results and first method's metrics for compatibility
    first_result = all_results[0] if all_results else None
    return results_df, first_result