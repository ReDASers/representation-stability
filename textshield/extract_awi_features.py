import os
import argparse
import json
import time
from typing import List, Tuple, Dict, Any, Optional, Callable
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer

from awi_utils import (
    compute_awi_vanilla_gradient,
    compute_awi_integrated_gradients,
    compute_awi_guided_backpropagation,
    compute_awi_lrp,
    _get_word_maps_for_batch
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global caches for tokenization and word mapping
tokenization_cache: Dict[str, Any] = {}
word_mapping_cache: Dict[str, Any] = {}

timing_metrics = {
    "dataset": [],
    "model": [],
    "attack": [],
    "method": [],
    "num_samples": [],
    "total_time": [],
    "time_per_sample": []
}

DATASETS = ["ag_news", "imdb", "yelp"]
MODEL_BASES = ["roberta", "deberta"] # Base names, will be combined with dataset suffix
ATTACKS = ["bert-attack", "deepwordbug", "textfooler"]
ALL_AWI_METHODS = ["vanilla_gradient", "integrated_gradients", "guided_backpropagation", "lrp"]

def get_cache_key(texts: List[str], max_length: Optional[int] = None) -> str:
    """Generate a cache key for a list of texts and max_length"""
    if not texts:
        return "empty_texts"
    first_text_sample = texts[0][:50] if texts[0] else ""
    last_text_sample = texts[-1][:50] if texts[-1] else ""
    return f"{hash(first_text_sample)}_{hash(last_text_sample)}_{len(texts)}_{max_length}"

def get_cached_tokenization(texts: List[str], tokenizer: PreTrainedTokenizer, 
                           max_length: Optional[int]) -> Tuple[Any, Any]:
    """Get tokenization and word mappings from cache or compute them"""
    global tokenization_cache, word_mapping_cache
    
    if not texts:
        return None, []
    
    cache_key = get_cache_key(texts, max_length)
    
    if cache_key not in tokenization_cache:
        batch_encoding = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        tokenization_cache[cache_key] = batch_encoding
        
        # Compute word mappings at the same time to avoid redundant processing
        word_maps = _get_word_maps_for_batch(
            texts,
            batch_encoding,
            tokenizer,
            max_length=max_length
        )
        word_mapping_cache[cache_key] = word_maps
    else:
        pass
    
    return tokenization_cache[cache_key], word_mapping_cache[cache_key]

def clear_caches():
    """Clear all caches to free memory"""
    global tokenization_cache, word_mapping_cache
    tokenization_cache.clear()
    word_mapping_cache.clear()
    print("Cleared tokenization and word mapping caches")

def load_model_and_tokenizer(model_path: str) -> Tuple[Optional[AutoModelForSequenceClassification], Optional[PreTrainedTokenizer]]:
    """Loads a Hugging Face model and tokenizer from the given path."""
    try:
        print(f"Loading model and tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        print(f"Successfully loaded model and tokenizer from {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer from {model_path}: {e}")
        return None, None


def load_text_data(data_dir: str, attack_name: str, dataset_type: str) -> Tuple[List[str], List[str]]:
    """
    Loads original and adversarial texts from the main data CSV.
    dataset_type: "calibration" or "test".
    The attack_name parameter is kept for consistency with the calling loop, but not directly used for file naming here.
    """
    original_texts: List[str] = []
    adversarial_texts: List[str] = []

    data_file = os.path.join(data_dir, f"{dataset_type}_data.csv")
    print(f"Attempting to load data from: {data_file} (attack_name '{attack_name}' is noted but file structure uses {dataset_type}_data.csv)")

    try:
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            
            if "original_text" in df.columns:
                original_texts = df["original_text"].astype(str).tolist()
                print(f"Loaded {len(original_texts)} original texts from 'original_text' column in {data_file}")
            else:
                print(f"WARNING: 'original_text' column not found in {data_file}")

            if "adversarial_text" in df.columns:
                adversarial_texts = df["adversarial_text"].astype(str).tolist()
                print(f"Loaded {len(adversarial_texts)} adversarial texts from 'adversarial_text' column in {data_file}")
            else:
                print(f"WARNING: 'adversarial_text' column not found in {data_file}. Adversarial texts will be empty for this load.")
        else:
            print(f"WARNING: Data file not found: {data_file}")
    except Exception as e:
        print(f"ERROR: Error loading texts from {data_file}: {e}")
        
    return original_texts, adversarial_texts


def get_awi_compute_function(
    method_name: str, 
    args: argparse.Namespace
) -> Optional[Callable[..., List[torch.Tensor]]]:
    """Returns the AWI computation function based on its name and configures its specific arguments."""
    if method_name == "vanilla_gradient":
        return lambda texts, model, tokenizer, max_length, batch_encoding=None, word_maps=None: compute_awi_vanilla_gradient(
            texts, model, tokenizer, 
            max_length=max_length, 
            batch_encoding=batch_encoding, 
            processed_text_data=word_maps
        )
    elif method_name == "integrated_gradients":
        return lambda texts, model, tokenizer, max_length, batch_encoding=None, word_maps=None: compute_awi_integrated_gradients(
            texts, model, tokenizer, 
            max_length=max_length, 
            steps=args.ig_steps, 
            baseline_type=args.ig_baseline_type,
            batch_encoding=batch_encoding, 
            processed_text_data=word_maps
        )
    elif method_name == "guided_backpropagation":
        return lambda texts, model, tokenizer, max_length, batch_encoding=None, word_maps=None: compute_awi_guided_backpropagation(
            texts, model, tokenizer, 
            max_length=max_length, 
            batch_encoding=batch_encoding, 
            processed_text_data=word_maps
        )
    elif method_name == "lrp":
        return lambda texts, model, tokenizer, max_length, batch_encoding=None, word_maps=None: compute_awi_lrp(
            texts, model, tokenizer, 
            max_length=max_length, 
            batch_encoding=batch_encoding, 
            processed_text_data=word_maps
        )
    else:
        print(f"ERROR: Unknown AWI method: {method_name}")
        return None

def convert_awi_tensors_to_lists(awi_tensors: List[torch.Tensor]) -> List[List[float]]:
    """Converts a list of AWI torch tensors to a list of Python lists."""
    result = []
    for tensor in awi_tensors:
        if tensor.numel() > 0:
            # Convert to a plain list of floats
            result.append(tensor.tolist())
        else:
            result.append([])
    return result


def save_json_file(file_path: str, data: Dict[str, Any]):
    """Saves a dictionary to a JSON file."""
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"CRITICAL ERROR: Error saving data to {file_path}: {e}")

def save_combined_dataset_json(
    file_path: str,
    all_methods_features_orig: Dict[str, List[List[float]]],
    all_methods_features_adv: Dict[str, List[List[float]]],
    processed_awi_methods: List[str],
    num_orig_samples: int, 
    num_adv_samples: int,  
    args: argparse.Namespace
):
    """Saves combined original and adversarial features (from potentially multiple AWI methods) and labels to a JSON file."""
    # Create labels
    labels_orig = np.zeros(num_orig_samples, dtype=int).tolist()
    labels_adv = np.ones(num_adv_samples, dtype=int).tolist()
    combined_labels = labels_orig + labels_adv
    
    # Combine original and adversarial features into multi-channel format
    print(f"Saving multi-channel format with {len(processed_awi_methods)} channels")
    
    # Combine original and adversarial features
    all_orig_features = {method: all_methods_features_orig.get(method, []) for method in processed_awi_methods}
    all_adv_features = {method: all_methods_features_adv.get(method, []) for method in processed_awi_methods}
    
    # Convert to multi-channel format
    multi_channel_orig_features = convert_to_multi_channel_format(all_orig_features, processed_awi_methods)
    multi_channel_adv_features = convert_to_multi_channel_format(all_adv_features, processed_awi_methods)
    
    # Combine original and adversarial multi-channel features
    multi_channel_features = multi_channel_orig_features + multi_channel_adv_features
    
    # Get multi-channel sample shapes
    multi_channel_shapes = []
    if multi_channel_features:
        for i in range(min(3, len(multi_channel_features))):
            try:
                if multi_channel_features[i]:
                    seq_len = len(multi_channel_features[i])
                    num_channels = len(multi_channel_features[i][0]) if seq_len > 0 else 0
                    multi_channel_shapes.append([seq_len, num_channels])
                else:
                    multi_channel_shapes.append([0, 0])
            except:
                multi_channel_shapes.append("unknown_shape")
    
    data_dict = {
        "features": multi_channel_features,
        "labels": combined_labels,
        "channel_methods": processed_awi_methods,  # Order of methods in channels
        "metadata": {
            "awi_methods_included": processed_awi_methods,
            "feature_count_total_samples": len(combined_labels), # Total samples
            "original_count": num_orig_samples,
            "adversarial_count": num_adv_samples,
            "max_token_length_setting": args.max_token_length,
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "feature_description": {
                "features": "List of multi-channel features. Each feature has shape [sequence_length, num_channels] "
                            "where each channel represents a different AWI method's importance score for the predicted class."
            },
            "sample_shapes_example": {
                "features": multi_channel_shapes
            }
        }
    }
    
    # Add method-specific config to metadata if only one method, or handle separately in feature_info
    if len(processed_awi_methods) == 1 and processed_awi_methods[0] == "integrated_gradients":
        data_dict["metadata"]["ig_steps"] = args.ig_steps
        data_dict["metadata"]["ig_baseline_type"] = args.ig_baseline_type
        
    save_json_file(file_path, data_dict)


def save_feature_info_file(
    file_path: str,
    processed_awi_methods: List[str],
    model_name_str: str,
    dataset_name_str: str,
    attack_name_str: str,
    num_classes: int,
    num_cal_orig_texts: int, num_cal_adv_texts: int,
    num_test_orig_texts: int, num_test_adv_texts: int,
    args: argparse.Namespace
):
    """Saves a feature_info.json file for potentially multiple AWI methods."""
    
    method_specific_configs = {}
    if "integrated_gradients" in processed_awi_methods:
        method_specific_configs["integrated_gradients"] = {
            "ig_steps": args.ig_steps,
            "ig_baseline_type": args.ig_baseline_type
        }
    if "vanilla_gradient" in processed_awi_methods:
        method_specific_configs.setdefault("vanilla_gradient", {})
    if "guided_backpropagation" in processed_awi_methods:
        method_specific_configs.setdefault("guided_backpropagation", {})
    if "lrp" in processed_awi_methods:
        method_specific_configs.setdefault("lrp", {})
    
    feature_info = {
        "awi_methods_processed": processed_awi_methods,
        "model_used": model_name_str,
        "dataset_used": dataset_name_str,
        "attack_type_context": attack_name_str,
        "num_model_classes": num_classes,
        "max_token_length_setting": args.max_token_length,
        "storage_format": "json (see feature_description)",
        "method_specific_configs": method_specific_configs,
        "text_sample_counts": { # Clarify these are text counts
            "calibration_original": num_cal_orig_texts,
            "calibration_adversarial": num_cal_adv_texts,
            "test_original": num_test_orig_texts,
            "test_adversarial": num_test_adv_texts,
        },
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        
        # Multi-channel format information
        "feature_formats": {
            "features": "Contains multi-channel AWI features where each channel represents a different AWI method's importance score. "
            "Access using 'features' key in data files."
        },
        "expected_matrix_shapes": {
            "features": f"(variable_sequence_length, {len(processed_awi_methods)})"
        },
        "channel_method_order": processed_awi_methods,
        "compatible_models": {
            "features": ["bilstm", "cnn", "transformer"]
        },
        "awi_feature_description": "Each feature is a 2D array with shape [sequence_length, num_channels], where each channel represents importance scores from a different AWI method."
    }
    
    save_json_file(file_path, feature_info)


def save_text_references_file(
    file_path: str,
    cal_orig_texts: List[str], cal_adv_texts: List[str],
    test_orig_texts: List[str], test_adv_texts: List[str],
    max_samples_to_store: int = 5
):
    """Saves a text_references.json file with sample texts."""
    text_refs = {
        "sample_texts": {
            "calibration_original": cal_orig_texts[:min(max_samples_to_store, len(cal_orig_texts))],
            "calibration_adversarial": cal_adv_texts[:min(max_samples_to_store, len(cal_adv_texts))],
            "test_original": test_orig_texts[:min(max_samples_to_store, len(test_orig_texts))],
            "test_adversarial": test_adv_texts[:min(max_samples_to_store, len(test_adv_texts))],
        },
        "text_counts": {
            "calibration_original": len(cal_orig_texts),
            "calibration_adversarial": len(cal_adv_texts),
            "test_original": len(test_orig_texts),
            "test_adversarial": len(test_adv_texts),
        },
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }
    save_json_file(file_path, text_refs)


def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    """Split a list of texts into batches of specified size."""
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

def compute_awi_in_batches(
    texts: List[str], 
    model, 
    tokenizer, 
    compute_func: Callable,
    max_token_length: int,
    batch_size: int,
    text_type: str = "Unknown",
) -> List[torch.Tensor]:
    """Compute AWI features in batches to manage memory usage."""
    if not texts:
        return []
    
    batches = batch_texts(texts, batch_size)
    total_batches = len(batches)
    
    all_awi_tensors = []
    
    # Configure nested tqdm 
    batch_processing_desc = f"{text_type} ({len(texts)} texts)"
    for batch_idx, batch in tqdm(enumerate(batches), total=total_batches, desc=batch_processing_desc, 
                               leave=False, position=1, dynamic_ncols=True, unit="batch"):
        
        # Get cached tokenization and word maps
        batch_encoding, word_maps = get_cached_tokenization(
            batch, tokenizer, max_token_length
        )
        
        # Pass cached tokenization and word maps to compute function
        batch_awi_tensors = compute_func(
            batch, model, tokenizer, max_token_length,
            batch_encoding=batch_encoding, word_maps=word_maps
        )
        
        all_awi_tensors.extend(batch_awi_tensors)
        
        # Clear CUDA cache if using GPU to prevent memory accumulation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return all_awi_tensors

# Function to log timing metrics for AWI methods
def log_timing_metrics(
    dataset_name: str,
    model_name: str, 
    attack_name: str, 
    awi_method: str, 
    num_samples: int,
    processing_time: float
):
    """Log timing metrics for an AWI method run"""
    time_per_sample = processing_time / max(num_samples, 1)  # Avoid division by zero
    
    # Add to metrics dictionary
    timing_metrics["dataset"].append(dataset_name)
    timing_metrics["model"].append(model_name)
    timing_metrics["attack"].append(attack_name)
    timing_metrics["method"].append(awi_method)
    timing_metrics["num_samples"].append(num_samples)
    timing_metrics["total_time"].append(processing_time)
    timing_metrics["time_per_sample"].append(time_per_sample)
    
    # Print to console
    print(f"  Timing for {awi_method}: {processing_time:.2f}s total, {time_per_sample:.4f}s per sample")
    
def save_timing_metrics(output_dir: str):
    """Save timing metrics to CSV file"""
    if not timing_metrics["method"]:
        print("WARNING: No timing metrics to save")
        return
        
    # Create DataFrame from metrics
    df = pd.DataFrame(timing_metrics)
    
    # Calculate combined metrics per dataset/model/attack combination
    combined_metrics = df.groupby(['dataset', 'model', 'attack']).agg({
        'num_samples': 'first', 
        'total_time': 'sum',
        'time_per_sample': lambda x: sum(x)
    }).reset_index()
    
    # Add method column with "combined" value
    combined_metrics['method'] = 'combined'
    
    # Recompute time_per_sample for combined metrics
    combined_metrics['time_per_sample'] = combined_metrics['total_time'] / combined_metrics['num_samples']
    
    df = pd.concat([df, combined_metrics], ignore_index=True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"awi_timing_metrics_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    
    df.to_csv(output_file, index=False)
    print(f"Saved timing metrics to {output_file}")
    
    # Create a summary with averages per method
    method_summary = df[df['method'] != 'combined'].groupby('method').agg({
        'time_per_sample': ['mean', 'min', 'max', 'std'],
        'num_samples': 'sum'
    }).reset_index()
    
    summary_file = os.path.join(output_dir, f"awi_timing_summary_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    method_summary.to_csv(summary_file, index=False)
    print(f"Saved timing summary to {summary_file}")

def convert_to_multi_channel_format(
    all_method_features: Dict[str, List[torch.Tensor]],
    method_order: List[str]
) -> List[List[List[float]]]:
    """
    Convert AWI features from separate method tensors to multi-channel sequential format.
    This function produces the primary feature representation for AWI features.
    
    Args:
        all_method_features: Dictionary mapping method names to feature tensors (1D per sample)
        method_order: List specifying the order of methods for channels
        
    Returns:
        List of multi-channel features, where each feature has shape [sequence_length, num_channels].
        Each channel corresponds to importance scores from a different AWI method.
    """
    if not all_method_features or not method_order:
        return []
    
    # Get sample count from first method's features
    first_method = method_order[0]
    if first_method not in all_method_features or not all_method_features[first_method]:
        return []
    
    num_samples = len(all_method_features[first_method])
    multi_channel_features = []
    
    # Process each sample
    for sample_idx in range(num_samples):
        # Find the maximum sequence length across all methods for this sample
        max_seq_len = 0
        for method in method_order:
            if (method in all_method_features and 
                sample_idx < len(all_method_features[method]) and
                len(all_method_features[method][sample_idx]) > 0):
                max_seq_len = max(max_seq_len, len(all_method_features[method][sample_idx]))
        
        if max_seq_len == 0:
            # No valid sequence length found, add empty feature
            multi_channel_features.append([])
            continue
        
        # Create multi-channel feature array for this sample
        # Shape: [max_seq_len, num_channels]
        sample_feature = [[0.0] * len(method_order) for _ in range(max_seq_len)]
        
        # Fill in features from each method
        for channel_idx, method in enumerate(method_order):
            if (method in all_method_features and 
                sample_idx < len(all_method_features[method]) and
                len(all_method_features[method][sample_idx]) > 0):
                
                method_feature = all_method_features[method][sample_idx]
                
                # Convert tensor to list of floats if needed
                if isinstance(method_feature, torch.Tensor):
                    method_feature = method_feature.tolist()
                    
                # Fill in the values up to the length of this method's feature
                for word_idx, importance in enumerate(method_feature):
                    if word_idx < max_seq_len:
                        sample_feature[word_idx][channel_idx] = float(importance)
        
        multi_channel_features.append(sample_feature)
    
    return multi_channel_features

def main(args: argparse.Namespace):
    """Main execution function."""
    overall_start_time = time.time()
    print(f"Starting AWI feature extraction with args: {args}")
    print(f"Using device: {device}")

    # Set timing metrics output directory
    timing_metrics_dir = args.timing_metrics_dir if args.timing_metrics_dir else os.path.join(args.base_data_dir, "timing_metrics")
    print(f"Timing metrics will be saved to: {timing_metrics_dir}")

    dataset_count = len(DATASETS)
    model_count = len(MODEL_BASES)
    attack_count = len(ATTACKS)
    method_count = len(ALL_AWI_METHODS) if args.awi_method == "all" else 1
    
    total_combinations = dataset_count * model_count * attack_count * method_count
    print(f"Total configurations to process: {total_combinations}")
    
    combination_counter = 0
    # Configure main progress bar with a more informative description and format
    progress_bar = tqdm(
        total=total_combinations, 
        desc="AWI Extraction Progress", 
        unit="config", 
        position=0, 
        leave=True, 
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for dataset_name in DATASETS:
        for model_base_name in MODEL_BASES:
            model_name_suffix = f"_{dataset_name.replace('-', '_')}"
            full_model_name = f"{model_base_name}{model_name_suffix}"
            model_load_path = os.path.join(args.base_model_dir, full_model_name)

            current_config_desc_base = f"D:{dataset_name[:7]} M:{model_base_name[:7]}"
            progress_bar.set_description(f"{current_config_desc_base} (Loading Model)")
            progress_bar.set_postfix({}) # Clear previous postfix

            model, tokenizer = load_model_and_tokenizer(model_load_path)
            if not model or not tokenizer:
                print(f"WARNING: Skipping model {full_model_name} due to loading error.")
                progress_bar.set_description(f"{current_config_desc_base} (Skip:ModelErr)")
                skipped_updates = 0
                if args.awi_method == "all":
                    skipped_updates = len(ATTACKS) * len(ALL_AWI_METHODS)
                else:
                    skipped_updates = len(ATTACKS) 
                progress_bar.update(skipped_updates)
                continue
            
            num_classes = model.config.num_labels

            for attack_name in ATTACKS:
                # Clear caches before each attack to manage memory
                clear_caches()
                
                current_data_path_prefix = os.path.join(args.base_data_dir, dataset_name, model_base_name, attack_name)
                
                current_config_desc_attack = f"{current_config_desc_base} A:{attack_name[:10]}"
                progress_bar.set_description(f"{current_config_desc_attack} (Loading Data)")
                progress_bar.set_postfix({}) 

                print(f"Processing: Dataset='{dataset_name}', Model='{full_model_name}', Attack='{attack_name}'")
                print(f"Data path prefix: {current_data_path_prefix}")

                cal_orig_texts, cal_adv_texts = load_text_data(current_data_path_prefix, attack_name, "calibration")
                test_orig_texts, test_adv_texts = load_text_data(current_data_path_prefix, attack_name, "test")

                if not (cal_orig_texts or cal_adv_texts or test_orig_texts or test_adv_texts):
                    print("WARNING: No text data found for this combination. Skipping.")
                    progress_bar.set_description(f"{current_config_desc_attack} (Skip:NoData)")
                    skipped_updates = 0
                    if args.awi_method == "all":
                        skipped_updates = len(ALL_AWI_METHODS)
                    else:
                        skipped_updates = 1 
                    progress_bar.update(skipped_updates)
                    continue

                selected_awi_methods = [args.awi_method] if args.awi_method != "all" else ALL_AWI_METHODS
                
                # Unified output directory for AWI features for this dataset/model/attack combo
                awi_output_dir = os.path.join(current_data_path_prefix, "awi")
                os.makedirs(awi_output_dir, exist_ok=True)

                # Dictionaries to store features from all selected methods
                all_cal_orig_awi_lists: Dict[str, List[List[float]]] = {}
                all_cal_adv_awi_lists: Dict[str, List[List[float]]] = {}
                all_test_orig_awi_lists: Dict[str, List[List[float]]] = {}
                all_test_adv_awi_lists: Dict[str, List[List[float]]] = {}

                for awi_method_name in selected_awi_methods:
                    loop_start_time = time.time()
                    print(f"  Using AWI method: {awi_method_name}")

                    # Update progress bar description and postfix with more information
                    progress_bar.set_description(f"{current_config_desc_attack} | {awi_method_name}")
                    progress_bar.set_postfix({"stage": "computing"}, refresh=True)
                    
                    awi_compute_func = get_awi_compute_function(awi_method_name, args)
                    if not awi_compute_func:
                        print(f"  WARNING: Skipping AWI method {awi_method_name} due to invalid function.")
                        progress_bar.update(1)
                        continue

                    # --- Compute AWI Features for the current method ---
                    processing_stages = [
                        ("calibration original", cal_orig_texts),
                        ("calibration adversarial", cal_adv_texts),
                        ("test original", test_orig_texts),
                        ("test adversarial", test_adv_texts)
                    ]
                    
                    # Count non-empty stages for progress bar
                    active_stages = [(name, texts) for name, texts in processing_stages if texts]
                    
                    stage_progress = tqdm(
                        active_stages, 
                        desc=f"AWI {awi_method_name}", 
                        leave=False, 
                        position=2, 
                        unit="stage"
                    )
                    
                    for stage_name, texts in stage_progress:
                        stage_progress.set_description(f"AWI {awi_method_name} - {stage_name}")
                        
                        if stage_name == "calibration original":
                            cal_orig_awi_tensors = compute_awi_in_batches(
                                texts, model, tokenizer, awi_compute_func, args.max_token_length, 
                                args.batch_size, stage_name
                            )
                            all_cal_orig_awi_lists[awi_method_name] = convert_awi_tensors_to_lists(cal_orig_awi_tensors)
                        elif stage_name == "calibration adversarial":
                            cal_adv_awi_tensors = compute_awi_in_batches(
                                texts, model, tokenizer, awi_compute_func, args.max_token_length, 
                                args.batch_size, stage_name
                            )
                            all_cal_adv_awi_lists[awi_method_name] = convert_awi_tensors_to_lists(cal_adv_awi_tensors)
                        elif stage_name == "test original":
                            test_orig_awi_tensors = compute_awi_in_batches(
                                texts, model, tokenizer, awi_compute_func, args.max_token_length, 
                                args.batch_size, stage_name
                            )
                            all_test_orig_awi_lists[awi_method_name] = convert_awi_tensors_to_lists(test_orig_awi_tensors)
                        elif stage_name == "test adversarial":
                            test_adv_awi_tensors = compute_awi_in_batches(
                                texts, model, tokenizer, awi_compute_func, args.max_token_length, 
                                args.batch_size, stage_name
                            )
                            all_test_adv_awi_lists[awi_method_name] = convert_awi_tensors_to_lists(test_adv_awi_tensors)
                    
                    stage_progress.close()
                    
                    # Handle empty text lists by ensuring the dictionaries have the method keys
                    if not cal_orig_texts:
                        all_cal_orig_awi_lists[awi_method_name] = []
                    if not cal_adv_texts:
                        all_cal_adv_awi_lists[awi_method_name] = []
                    if not test_orig_texts:
                        all_test_orig_awi_lists[awi_method_name] = []
                    if not test_adv_texts:
                        all_test_adv_awi_lists[awi_method_name] = []
                    
                    # Count how many matrices were successfully generated
                    total_processed = (
                        len(all_cal_orig_awi_lists[awi_method_name]) +
                        len(all_cal_adv_awi_lists[awi_method_name]) +
                        len(all_test_orig_awi_lists[awi_method_name]) +
                        len(all_test_adv_awi_lists[awi_method_name])
                    )
                    
                    total_expected = (
                        len(cal_orig_texts) + 
                        len(cal_adv_texts) + 
                        len(test_orig_texts) + 
                        len(test_adv_texts)
                    )
                                        
                    loop_duration = time.time() - loop_start_time
                    
                    # Log timing metrics
                    log_timing_metrics(
                        dataset_name, 
                        model_base_name, 
                        attack_name, 
                        awi_method_name, 
                        total_processed, 
                        loop_duration
                    )
                    
                    print(f"  Finished {awi_method_name}: {total_processed}/{total_expected} texts processed in {loop_duration:.2f}s")
                    
                    progress_bar.update(1)
                    combination_counter += 1
                
                # --- Save Aggregated Data and Metadata (after processing all selected AWI methods) ---
                if not selected_awi_methods: 
                    print("WARNING: No AWI methods were selected or processed. Skipping save.")
                    progress_bar.set_description(current_config_desc_attack)
                    progress_bar.set_postfix(status="NoMethodsToSave", refresh=True)
                    continue

                progress_bar.set_description(f"{current_config_desc_attack} | Saving")
                progress_bar.set_postfix({"stage": "saving data"}, refresh=True)

                cal_data_path = os.path.join(awi_output_dir, "cal_data.json")
                save_combined_dataset_json(
                    cal_data_path, 
                    all_cal_orig_awi_lists, 
                    all_cal_adv_awi_lists, 
                    selected_awi_methods,
                    len(cal_orig_texts), 
                    len(cal_adv_texts), 
                    args
                )

                test_data_path = os.path.join(awi_output_dir, "test_data.json")
                save_combined_dataset_json(
                    test_data_path, 
                    all_test_orig_awi_lists, 
                    all_test_adv_awi_lists, 
                    selected_awi_methods,
                    len(test_orig_texts),
                    len(test_adv_texts),  
                    args
                )
                
                feature_info_path = os.path.join(awi_output_dir, "feature_info.json")
                save_feature_info_file(
                    feature_info_path, 
                    selected_awi_methods, 
                    full_model_name, 
                    dataset_name, 
                    attack_name, 
                    num_classes,
                    len(cal_orig_texts), len(cal_adv_texts),
                    len(test_orig_texts), len(test_adv_texts), 
                    args
                )

                text_refs_path = os.path.join(awi_output_dir, "text_references.json")
                save_text_references_file(
                    text_refs_path, cal_orig_texts, cal_adv_texts, test_orig_texts, test_adv_texts
                )
                
                # Update progress bar to show completion of this dataset/model/attack combination
                progress_bar.set_postfix({"stage": "completed", "methods": len(selected_awi_methods)}, refresh=True)
            
            # Unload model from GPU after processing all attacks for a model
            if model is not None and device.type == 'cuda':
                model.cpu()
                del model
                torch.cuda.empty_cache()
                print(f"Unloaded model {full_model_name} from GPU")
    
    progress_bar.set_description("Completed")
    progress_bar.set_postfix({})
    progress_bar.close()
    overall_duration = time.time() - overall_start_time
    print(f"Total AWI feature extraction completed in {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes).")
    print(f"Processed {combination_counter} out of {total_combinations} planned configurations.")
    
    # Save timing metrics to a global output directory
    save_timing_metrics(timing_metrics_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Adaptive Word Importance (AWI) features.")
    parser.add_argument(
        "--awi_method", type=str, default="vanilla_gradient",
        choices=ALL_AWI_METHODS + ["all"],
        help="AWI method to use or 'all' to run all methods."
    )
    parser.add_argument(
        "--base_model_dir", type=str, default="models/",
        help="Base directory where pre-trained models are stored."
    )
    parser.add_argument(
        "--base_data_dir", type=str, default="data/",
        help="Base directory for datasets."
    )
    parser.add_argument(
        "--timing_metrics_dir", type=str, default=None,
        help="Directory to save timing metrics CSV files. If not specified, will use base_data_dir/timing_metrics."
    )
    parser.add_argument(
        "--max_token_length", type=int, default=128,
        help="Maximum token length for tokenizer and AWI computation."
    )
    # IG specific args
    parser.add_argument(
        "--ig_steps", type=int, default=5,
        help="Number of steps for Integrated Gradients."
    )
    parser.add_argument(
        "--ig_baseline_type", type=str, default="zero", choices=["zero", "pad", "random"],
        help="Baseline type for Integrated Gradients."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for processing texts (to manage memory usage)."
    )
    
    parsed_args = parser.parse_args()
    main(parsed_args) 