#!/usr/bin/env python3
"""
RS Feature Extraction Script

A script to extract RS sensitivity features and save them as JSON files
for use with external detection models.

This script runs the full RS pipeline but only saves the extracted features
without training any detection models.

"""

import json
import os
import time
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .main import parse_arguments, create_experiment_directories
from .core.pipeline import (
    _prepare_experiment_data, _compute_embeddings, _compute_sensitivity_maps,
    _extract_and_process_features
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_data(args):
    """Load model, tokenizer, and datasets."""
    
    # Load model and tokenizer
    model_path = args.model_path if args.model_path else f"redasers/{args.model_name}"
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, attn_implementation="eager").to(device)
    model.eval()
    
    # Load datasets
    cal_path = f"{args.data_dir}/calibration_data.csv"
    test_path = f"{args.data_dir}/test_data.csv"
    
    print(f"Loading data from {args.data_dir}")
    calibration_df = pd.read_csv(cal_path)
    test_df = pd.read_csv(test_path)
    
    calibration_data = Dataset.from_pandas(calibration_df)
    test_data = Dataset.from_pandas(test_df)
    
    # Check for attack log files
    attack_name = os.path.basename(args.data_dir)
    cal_attack_logs_path = f"{args.data_dir}/{attack_name}_calibration_attack_logs.csv"
    test_attack_logs_path = f"{args.data_dir}/{attack_name}_test_attack_logs.csv"
    
    if not os.path.exists(cal_attack_logs_path):
        cal_attack_logs_path = None
    if not os.path.exists(test_attack_logs_path):
        test_attack_logs_path = None
    
    return model, tokenizer, calibration_data, test_data, cal_attack_logs_path, test_attack_logs_path


def save_features_as_json(features, labels, output_path, metadata=None):
    """Save features and labels as JSON format."""
    
    # Convert features to JSON-serializable format
    json_features = []
    for feature in features:
        if hasattr(feature, 'tolist'):
            json_features.append(feature.tolist())
        else:
            json_features.append(feature)
    
    # Convert labels to JSON-serializable format
    json_labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
    
    # Create output data
    output_data = {
        "features": json_features,
        "labels": json_labels,
        "metadata": metadata or {}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Features saved to: {output_path}")


def extract_features_only(args):
    """Extract RS features without training models."""
    
    print("="*60)
    print("RS FEATURE EXTRACTION MODE")
    print("="*60)
    
    # Create experiment directories
    dirs = create_experiment_directories(args)
    print(f"Output directory: {dirs['root']}")
    
    # Load model and data
    model, tokenizer, calibration_data, test_data, cal_attack_logs_path, test_attack_logs_path = load_model_and_data(args)
    
    # Determine word selection strategy
    word_selection_strategy = "all"
    if args.use_gradient_attention:
        word_selection_strategy = "gradient_attention"
    elif args.use_attention:
        word_selection_strategy = "attention"
    elif args.use_saliency:
        word_selection_strategy = "saliency"
    elif args.use_random:
        word_selection_strategy = "random"
    
    print(f"Word selection strategy: {word_selection_strategy}")
    
    # Step 1: Prepare experiment data
    print("Step 1/4: Preparing experiment data...")
    cal_orig_texts, cal_adv_texts, test_orig_texts, test_adv_texts, cal_perturbed_positions, test_perturbed_positions = _prepare_experiment_data(
        calibration_data, test_data, model, tokenizer, args.max_length,
        cal_attack_logs_path, test_attack_logs_path
    )
    
    # Step 2: Compute embeddings
    print("Step 2/4: Computing text embeddings...")
    cal_orig_emb, cal_adv_emb, test_orig_emb, test_adv_emb = _compute_embeddings(
        cal_orig_texts, cal_adv_texts, test_orig_texts, test_adv_texts,
        model, tokenizer, args.max_length, args.batch_size
    )
    
    # Step 3: Compute sensitivity maps
    print("Step 3/4: Computing sensitivity maps...")
    cal_orig_sensitivity, cal_adv_sensitivity, \
    test_orig_sensitivity, test_adv_sensitivity, \
    computed_word_selection_strategy = _compute_sensitivity_maps(
        cal_orig_texts,
        cal_adv_texts,
        test_orig_texts,
        test_adv_texts,
        cal_orig_emb,
        cal_adv_emb,
        test_orig_emb,
        test_adv_emb,
        model,
        tokenizer,
        args.distance_metric,  # This should be the distance metric, not strategy
        # Word selection strategy flags
        use_attention=args.use_attention,
        use_saliency=args.use_saliency,
        use_random=args.use_random,
        use_gradient_attention=args.use_gradient_attention,
        use_integrated_gradients=False,
        # Strategy-specific parameters
        attention_top_n=args.top_n if args.use_attention else 5,
        gradient_attention_top_n=args.top_n if args.use_gradient_attention else 5,
        saliency_top_n=args.top_n if args.use_saliency else 5,
        random_top_n=args.top_n if args.use_random else 5,
        random_seed=args.random_seed,
        # Processing parameters
        max_length=args.max_length,
        main_batch_size=args.batch_size,
        perturb_batch_size=args.batch_size,
        use_importance_channel=args.use_importance_channel,
    )
    
    # Use the computed strategy for consistency
    word_selection_strategy = computed_word_selection_strategy
    print(f"Confirmed word selection strategy: {word_selection_strategy}")
    
    # Step 4: Extract features
    print("Step 4/4: Extracting features...")
    cal_orig_features, cal_adv_features, test_orig_features, test_adv_features = _extract_and_process_features(
        cal_orig_sensitivity=cal_orig_sensitivity,  
        cal_adv_sensitivity=cal_adv_sensitivity,   
        test_orig_sensitivity=test_orig_sensitivity, 
        test_adv_sensitivity=test_adv_sensitivity,  
        cal_orig_emb=cal_orig_emb,
        cal_adv_emb=cal_adv_emb,
        test_orig_emb=test_orig_emb,
        test_adv_emb=test_adv_emb,
        # Add raw importance maps
        cal_orig_imp_maps=cal_orig_sensitivity.raw_importance_maps,
        cal_adv_imp_maps=cal_adv_sensitivity.raw_importance_maps,
        test_orig_imp_maps=test_orig_sensitivity.raw_importance_maps,
        test_adv_imp_maps=test_adv_sensitivity.raw_importance_maps,
        use_importance_channel=args.use_importance_channel,
        filter_importance_scores=args.filter_importance_scores,
        # Strategy flags and parameters for determining top_n
        use_gradient_attention=args.use_gradient_attention,
        use_integrated_gradients=False,  
        use_attention=args.use_attention,
        use_saliency=args.use_saliency,
        use_random=args.use_random,
        gradient_attention_top_n=args.top_n if args.use_gradient_attention else 10,
        integrated_gradients_top_n=10,  # Not used
        attention_top_n=args.top_n if args.use_attention else 10,
        saliency_top_n=args.top_n if args.use_saliency else 10,
        random_top_n=args.top_n if args.use_random else 10,
    )
    
    # Create metadata
    metadata = {
        "strategy": word_selection_strategy,
        "top_n": args.top_n,
        "distance_metric": args.distance_metric,
        "model_name": args.model_name,
        "data_dir": args.data_dir,
        "use_importance_channel": args.use_importance_channel,
        "filter_importance_scores": args.filter_importance_scores,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_samples": {
            "original": len(cal_orig_features),
            "adversarial": len(cal_adv_features)
        },
        "test_samples": {
            "original": len(test_orig_features),
            "adversarial": len(test_adv_features)
        }
    }
    
    # Save features
    print("Saving extracted features...")
    
    # Create labels (0 = original, 1 = adversarial)
    import numpy as np
    cal_labels = np.concatenate([
        np.zeros(len(cal_orig_features)),
        np.ones(len(cal_adv_features))
    ])
    test_labels = np.concatenate([
        np.zeros(len(test_orig_features)),
        np.ones(len(test_adv_features))
    ])
    
    # Combine features
    cal_features = cal_orig_features + cal_adv_features
    test_features = test_orig_features + test_adv_features
    
    # Save calibration data
    cal_output_path = dirs['root'] / 'calibration_features.json'
    save_features_as_json(cal_features, cal_labels, cal_output_path, metadata)
    
    # Save test data
    test_output_path = dirs['root'] / 'test_features.json'
    save_features_as_json(test_features, test_labels, test_output_path, metadata)
    
    print(f"Feature extraction completed!")
    print(f"Results saved to: {dirs['root']}")
    
    return dirs['root']


def main():
    """Main entry point for feature extraction."""
    # Parse arguments using the same parser as main RS
    args = parse_arguments()
    
    # Run feature extraction
    output_dir = extract_features_only(args)
    
    print(f"\nRS feature extraction completed successfully!")
    print(f"Features saved to: {output_dir}")


if __name__ == "__main__":
    main()