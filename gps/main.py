"""
Guided Perturbation Sensitivity (GPS) System - Main Entry Point

This module implements the main entry point for the GPS adversarial detection system.
The GPS system detects adversarial text examples by analyzing word-level sensitivity
patterns through guided perturbations based on different importance metrics.

Key Components:
- Word selection strategies: attention, saliency, gradient×attention, random
- Distance metrics: cosine, euclidean, manhattan  
- Detection methods: BiLSTM
- Progressive word removal for sensitivity analysis

Usage:
    python adversarial_detector_progressive_word.py --model_name roberta_ag_news \
        --data_dir data/ag_news/roberta/textfooler --use_attention --top_n 20
"""

# Standard library imports
import os
import random
import time
import argparse
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

# Scientific computing
import numpy as np
import pandas as pd

# Deep learning
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .core.pipeline import (
    _prepare_experiment_data,
    _compute_embeddings,
    _compute_sensitivity_maps,
    _extract_and_process_features,
    _train_evaluate_detectors,
)

from .analysis.perturbation_eval import _calculate_and_save_perturbation_overlap

# Global configuration
SEED = 42  # Fixed seed for reproducibility across all random number generators

# Initialize random number generators for reproducible results
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device for GPU/CPU computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments for GPS experiments.
    
    Configures argument parser with all required options for running GPS
    adversarial detection experiments, including word selection strategies,
    processing parameters, and detection methods.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments with validation
        
    Raises:
        SystemExit: If required arguments are missing or invalid
    """
    parser = argparse.ArgumentParser(
        description='Run Guided Perturbation Sensitivity (GPS) adversarial detection experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core required arguments
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name (e.g., roberta_ag_news, deberta_imdb)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory path (e.g., data/ag_news/roberta/textfooler)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Base output directory (default: output)')
    
    # Word selection strategy (mutually exclusive group)
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument('--use_attention', action='store_true',
                               help='Use attention-based word selection')
    strategy_group.add_argument('--use_saliency', action='store_true',
                               help='Use saliency-based word selection')
    strategy_group.add_argument('--use_gradient_attention', action='store_true',
                               help='Use gradient×attention word selection')
    strategy_group.add_argument('--use_random', action='store_true',
                               help='Use random word selection')
    
    # Strategy-specific parameters  
    parser.add_argument('--top_n', type=int, default=25,
                       help='Number of top words to select (default: 25)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for random strategy (default: 42)')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length (default: 128)')
    parser.add_argument('--distance_metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'manhattan'],
                       help='Distance metric for sensitivity computation (default: cosine)')
    
    # Detection methods
    parser.add_argument('--detection_methods', nargs='+', default=['bilstm'],
                       choices=['bilstm', 'logistic', 'svm', 'random_forest'],
                       help='Detection methods to use (default: bilstm)')
    
    # Fixed parameters (previously in config)
    parser.add_argument('--use_importance_channel', action='store_true', default=True,
                       help='Use importance channel in features (default: True)')
    parser.add_argument('--filter_importance_scores', action='store_true', default=True,
                       help='Filter importance scores (default: True)')
    
    return parser.parse_args()


def create_experiment_directories(args) -> Dict[str, Path]:
    """Create hierarchical experiment directory structure for organized result storage.
    
    Creates nested directory structure based on dataset, attack method, model architecture,
    and word selection strategy to avoid naming conflicts and enable systematic analysis.
    
    Directory Structure:
        output/{dataset}_{attack}/{model_arch}/{strategy}/
        
    Examples:
        - output/ag_news_textfooler/roberta/attention_top20/
        - output/imdb_bertattack/deberta/saliency_top25/
    
    Args:
        args: Parsed command-line arguments containing data_dir, model_name,
              output_dir, and strategy flags
    
    Returns:
        Dict[str, Path]: Dictionary with 'root' key pointing to experiment directory
        
    Note:
        Auto-detects dataset name from supported options: ag_news, yelp, imdb
        Falls back to 'unknown_dataset' if detection fails
    """
    # Extract attack method from the final directory component
    data_path = Path(args.data_dir)
    attack_method = data_path.name
    
    # Auto-detect dataset name from data directory path
    dataset_name = "unknown_dataset"
    supported_datasets = ["ag_news", "yelp", "imdb"]
    for part in data_path.parts:
        if part in supported_datasets:
            dataset_name = part
            break
    
    # Extract model architecture from model name (e.g., "roberta_ag_news" -> "roberta")
    model_architecture = args.model_name.split('_')[0]
    
    # Generate strategy identifier based on selected word selection method
    if args.use_attention:
        strategy = f"attention_top{args.top_n}"
    elif args.use_saliency:
        strategy = f"saliency_top{args.top_n}"
    elif args.use_gradient_attention:
        strategy = f"gradattn_top{args.top_n}"
    elif args.use_random:
        strategy = f"random_top{args.top_n}_seed{args.random_seed}"
    else:
        strategy = "all"  # Fallback for all-word analysis
    
    # Build hierarchical directory path: output/{dataset}_{attack}/{model}/{strategy}/
    experiment_dir = Path(args.output_dir) / f"{dataset_name}_{attack_method}" / model_architecture / strategy
    
    # Create directory structure
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return {"root": experiment_dir}

def run_word_level_experiment(
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    calibration_data: Dataset,
    test_data: Dataset,
    cal_attack_logs_path: Optional[str] = None,
    test_attack_logs_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Execute complete GPS word-level sensitivity experiment pipeline.
    
    Orchestrates the full GPS adversarial detection workflow by computing word-level
    sensitivity maps through guided perturbation, extracting discriminative features,
    and training/evaluating multiple detection models.
    
    Experiment Pipeline:
        1. Prepare and validate input data
        2. Compute embeddings for original and adversarial texts  
        3. Generate word-level sensitivity maps via guided perturbation
        4. Extract statistical features from sensitivity patterns
        5. Train and evaluate detector models (BiLSTM, SVM, etc.)
        6. Calculate perturbation overlap metrics
        7. Save comprehensive results and visualizations
    
    Args:
        args: Command-line arguments with experiment configuration
        dirs: Directory paths for saving results and outputs
        model: Pre-trained transformer model for embedding computation
        tokenizer: Corresponding tokenizer for text preprocessing
        calibration_data: Dataset for detector training (original + adversarial pairs)
        test_data: Dataset for final evaluation (original + adversarial pairs)
        cal_attack_logs_path: Optional path to calibration attack logs for overlap analysis
        test_attack_logs_path: Optional path to test attack logs for overlap analysis
        
    Returns:
        Tuple containing:
            - pd.DataFrame: Performance metrics for all detection methods
            - Dict: Trained detector models and additional results
            
    Raises:
        ValueError: If datasets lack required columns or are incompatible
        RuntimeError: If GPU memory insufficient for batch processing
    """
    print(f"\n{'='*80}\nRunning GPS Word-Level Sensitivity Experiment\n{'='*80}")
    print(f"Distance metric: {args.distance_metric}")
    print(f"Max sequence length: {args.max_length}")
    
    # Set output directory for all experiment results
    exp_results_dir = dirs["root"]
    
    # Map command-line flags to strategy identifiers
    word_selection_strategy = "all"  # Default fallback
    if args.use_gradient_attention:
        word_selection_strategy = "gradient_attention"
    elif args.use_attention:
        word_selection_strategy = "attention"
    elif args.use_saliency:
        word_selection_strategy = "saliency"
    elif args.use_random:
        word_selection_strategy = "random"
    print(f"Word selection strategy: {word_selection_strategy}")
    
    # Step 1: Prepare and validate experiment datasets
    print("Step 1/6: Preparing experiment data...")
    cal_orig_texts, cal_adv_texts, test_orig_texts, test_adv_texts, cal_perturbed_positions, test_perturbed_positions = _prepare_experiment_data(
        calibration_data,
        test_data,
        model,  
        tokenizer,
        args.max_length,
        cal_attack_logs_path,
        test_attack_logs_path
    )
    
    # Step 2: Compute text embeddings using pre-trained model  
    print("Step 2/6: Computing text embeddings...")
    cal_orig_emb, cal_adv_emb, test_orig_emb, test_adv_emb = _compute_embeddings(
        cal_orig_texts,
        cal_adv_texts,
        test_orig_texts,
        test_adv_texts,
        model,
        tokenizer,
        args.max_length,
        args.batch_size,
    )
    
    # Step 3: Generate word-level sensitivity maps via guided perturbation
    print("Step 3/6: Computing word-level sensitivity maps...")
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
        args.distance_metric,
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
    
    # Ensure consistency between requested and computed strategy
    word_selection_strategy = computed_word_selection_strategy
    print(f"Confirmed word selection strategy: {word_selection_strategy}")
    
    # Step 4: Extract statistical features from sensitivity patterns  
    print("Step 4/6: Extracting discriminative features...")
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
    
    # Step 5: Analyze perturbation overlap between attacks and GPS predictions
    print("Step 5/6: Calculating perturbation overlap metrics...")
    all_metrics, all_plot_data, top_n = _calculate_and_save_perturbation_overlap(
        cal_perturbed_positions=cal_perturbed_positions,
        test_perturbed_positions=test_perturbed_positions,
        cal_adv_sensitivity=cal_adv_sensitivity,   
        test_adv_sensitivity=test_adv_sensitivity, 
        output_dir=str(exp_results_dir),
        word_selection_strategy=word_selection_strategy,  # Use computed strategy
        top_n=args.top_n if word_selection_strategy != "all" else 0
    )

    # Step 6: Train and evaluate adversarial detection models
    print("Step 6/6: Training and evaluating detection models...")
    results_df, detector_results_metrics = _train_evaluate_detectors(
        cal_orig_features,
        cal_adv_features,
        test_orig_features,
        test_adv_features,
        cal_orig_sensitivity,  
        cal_adv_sensitivity,   
        test_orig_sensitivity, 
        test_adv_sensitivity,  
        output_dir=str(exp_results_dir),
        distance_metric=args.distance_metric,
        detection_methods=args.detection_methods,
        random_seed=args.random_seed,
    )

    print(f"\nGPS experiment pipeline completed successfully!")
    print(f"All results saved to: {exp_results_dir}")

    return results_df, detector_results_metrics


def main() -> None:
    """Execute GPS adversarial detection experiment from command-line interface.
    
    Main entry point that coordinates the complete GPS experimental workflow:
    parsing arguments, loading models/data, running experiments, and saving results.
    Includes comprehensive error handling and execution time tracking.
    
    Workflow:
        1. Parse and validate command-line arguments
        2. Create organized output directory structure  
        3. Load pre-trained model, tokenizer, and datasets
        4. Execute GPS word-level sensitivity experiment
        5. Save results and report execution statistics
        
    Raises:
        FileNotFoundError: If model or data files cannot be located
        ValueError: If arguments are invalid or incompatible  
        RuntimeError: If CUDA operations fail or memory insufficient
        
    Note:
        Execution time and memory usage are tracked and reported.
        All errors are logged with full stack traces for debugging.
    """
    # Initialize experiment configuration
    args = parse_arguments()
    dirs = create_experiment_directories(args)
    
    print(f"GPS Experiment Configuration: {vars(args)}")
    print(f"Output directory: {dirs['root']}")
    
    # Load pre-trained model and tokenizer
    model_path = os.path.join("models", args.model_name)
    print(f"Loading model components from {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            attn_implementation="eager"  # Ensure attention weights are accessible
        ).to(device)
        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"ERROR: Failed to load model from {model_path}: {e}")
        return

    # Setup data file paths
    calibration_path = os.path.join(args.data_dir, "calibration_data.csv")
    test_path = os.path.join(args.data_dir, "test_data.csv")
    
    # Auto-detect attack method from directory structure
    data_dir_parts = args.data_dir.replace('\\', '/').split('/')
    attack_name = data_dir_parts[-1] if data_dir_parts else "textfooler"
    print(f"Detected attack method: {attack_name}")
    
    # Locate optional attack log files for perturbation overlap analysis
    cal_attack_logs_path = os.path.join(args.data_dir, f"{attack_name}_calibration_attack_logs.csv")
    test_attack_logs_path = os.path.join(args.data_dir, f"{attack_name}_test_attack_logs.csv")
    
    # Check availability of attack logs (used for overlap metrics)
    if not os.path.exists(cal_attack_logs_path):
        print(f"WARNING: Calibration attack logs not found - overlap analysis will be limited")
        cal_attack_logs_path = None
    
    if not os.path.exists(test_attack_logs_path):
        print(f"WARNING: Test attack logs not found - overlap analysis will be limited")
        test_attack_logs_path = None
    
    # Load experiment datasets
    print(f"Loading datasets from {args.data_dir}")
    try:
        calibration_df = pd.read_csv(calibration_path)
        test_df = pd.read_csv(test_path)
        print(f"Loaded {len(calibration_df)} calibration and {len(test_df)} test examples")
    except FileNotFoundError as e:
        print(f"ERROR: Dataset files not found: {e}")
        print(f"Expected files: {calibration_path}, {test_path}")
        return

    # Convert pandas DataFrames to HuggingFace Dataset format
    calibration_data = Dataset.from_pandas(calibration_df)
    test_data = Dataset.from_pandas(test_df)
    
    # Generate experiment configuration summary
    strategy_name = ""
    if args.use_attention:
        strategy_name = f"attention-top{args.top_n}"
    elif args.use_saliency:
        strategy_name = f"saliency-top{args.top_n}"
    elif args.use_gradient_attention:
        strategy_name = f"gradient-attention-top{args.top_n}"
    elif args.use_random:
        strategy_name = f"random-top{args.top_n}-seed{args.random_seed}"
    
    print(f"Experiment configuration: {strategy_name} | {args.distance_metric} distance | {len(args.detection_methods)} detectors")
    
    # Begin execution timing
    start_time = time.time()
    print(f"Starting GPS experiment at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Execute main GPS experiment pipeline with error handling
    try:
        results_df, detector_results = run_word_level_experiment(
            args,
            dirs,
            model, 
            tokenizer,
            calibration_data, 
            test_data,
            cal_attack_logs_path=cal_attack_logs_path,
            test_attack_logs_path=test_attack_logs_path
        )
        
        # Save comprehensive results to CSV  
        if results_df is not None and not results_df.empty:
            results_path = dirs["root"] / "experiment_results.csv"
            results_df.to_csv(results_path, index=False)
            print(f"Final results saved to: {results_path}")
        else:
            print("WARNING: No results generated to save")
        
    except Exception as e:
        print(f"ERROR: Experiment failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Report execution statistics
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"GPS Experiment Completed Successfully!")
    print(f"Total execution time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
    print(f"Results directory: {dirs['root']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()