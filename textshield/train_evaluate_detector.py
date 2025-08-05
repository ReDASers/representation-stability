import os
import random
import time
import argparse
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gps.training.detector import train_sensitivity_based_detector, evaluate_sensitivity_based_detector

# Constants
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_features_from_json(json_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load features and labels from a JSON file, with validation.
    
    Reads feature data from a JSON file and performs validation to ensure
    that features are properly formatted as 2D NumPy arrays with numeric values.
    Skips invalid samples with appropriate warning messages.
    
    Args:
        json_path: Path to the JSON file containing features and labels
        
    Returns:
        Tuple of (features_list_of_np_arrays, labels_np_array)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed_features = []
    if 'features' in data and isinstance(data['features'], list):
        for i, feature_list_sample in enumerate(data['features']):
            try:
                if not feature_list_sample: # Skip empty feature lists for a sample
                    print(f"Sample {i} in {json_path} has empty features. Skipping.")
                    continue
                
                # Attempt to convert to NumPy array, expecting 2D numeric data
                np_feature_sample = np.array(feature_list_sample, dtype=float)
                
                # Basic validation: ensure it's 2D (sequence_length, num_channels)
                if np_feature_sample.ndim != 2:
                    print(f"Sample {i} in {json_path} is not 2D (shape: {np_feature_sample.shape}). Skipping.")
                    continue
                if np_feature_sample.shape[0] == 0: # sequence length is 0
                    print(f"Sample {i} in {json_path} has 0 sequence length. Skipping.")
                    continue
                
                processed_features.append(np_feature_sample)
            except ValueError as e:
                print(f"Could not convert sample {i} features in {json_path} to numeric np.ndarray: {e}. Skipping sample.")
            except Exception as e:
                print(f"Unexpected error processing sample {i} features in {json_path}: {e}. Skipping sample.")
    else:
        print(f"'features' key missing or not a list in {json_path}. Returning empty features.")

    labels = np.array(data.get('labels', []), dtype=int)
    
    # Adjust labels if some features were skipped
    if len(processed_features) != len(data.get('features', [])) and len(labels) == len(data.get('features', [])):
        print(f"Number of processed features ({len(processed_features)}) doesn't match original number of feature samples in {json_path}. Label correspondence might be affected if any samples were skipped.")    

    return processed_features, labels

def run_experiment(config: Dict[str, Any]):
    """
    Run a detector training and evaluation experiment with single seed.
    
    Performs the complete detector training and evaluation workflow:
    1. Loads and processes training data
    2. Trains an adversarial detector model with fixed seed
    3. Evaluates on test set
    4. Computes and reports performance metrics
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        Tuple of (results_dataframe, trained_detector)
    """
    features_dir = getattr(config, "features_dir", "data")
    output_dir = getattr(config, "output_dir", "output/detector_results")
    strategy = getattr(config, "strategy", "gradattn")
    detection_method = getattr(config, "detection_method", "bilstm")
    dataset = getattr(config, "dataset", None)
    model = getattr(config, "model", None)
    attack = getattr(config, "attack", None)
    
    # Use single seed for evaluation
    print("Using single seed evaluation")
    
    # Update the experiment directory structure to include dataset/model/attack/strategy hierarchy
    if dataset and model and attack:
        experiment_output_dir = Path(output_dir) / dataset / model / attack / strategy
    else:
        experiment_output_dir = Path(output_dir) / strategy
    
    os.makedirs(experiment_output_dir, exist_ok=True)
    
    # Create subdirectories for logs and results
    logs_dir = experiment_output_dir / "logs"
    results_dir = experiment_output_dir / "results"
    models_dir = experiment_output_dir / "models"
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    
    experiment_config = vars(config).copy()
    
    experiment_config["experiment_name"] = f"{strategy}_{detection_method}"
    experiment_config["max_len"] = 128  # Ensure max_len is set
    experiment_config["timestamp"] = time.strftime("%Y%m%d-%H%M%S")

    config_path = experiment_output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=4)
    
    print(f"Experiment directory: {experiment_output_dir}")
    print(f"Strategy: {strategy}, Detection method: {detection_method}")
    
    # Construct paths for calibration and test data
    cal_data_path = None
    test_data_path = None
    
    if dataset and model and attack:
        # If dataset, model, and attack are specified, use them to construct the path
        if strategy == "awi":
            # Special handling for awi features
            data_path = Path(features_dir) / dataset / model / attack / "awi"
        else:
            data_path = Path(features_dir) / dataset / model / attack / strategy
            
        cal_data_path = data_path / "cal_data.json"
        test_data_path = data_path / "test_data.json"
        
        if not cal_data_path.exists() or not test_data_path.exists():
            print(f"Calibration or test data not found at {data_path}")
            return None, None
    else:
        # If dataset, model, or attack are not specified, look for strategy directory directly
        if strategy == "awi":
            strategy_dir = Path(features_dir) / "awi"
        else:
            strategy_dir = Path(features_dir) / strategy
            
        if strategy_dir.exists():
            cal_data_path = strategy_dir / "cal_data.json"
            test_data_path = strategy_dir / "test_data.json"
            
            if not cal_data_path.exists() or not test_data_path.exists():
                print(f"Calibration or test data not found at {strategy_dir}")
                return None, None
        else:
            print(f"Feature directory not found: {strategy_dir}")
            return None, None
    
    # Load calibration (training) data
    print(f"Loading calibration data from {cal_data_path}")
    train_features, train_labels = load_features_from_json(cal_data_path)
    
    if not train_features:
        print("No valid training features found. Aborting experiment.")
        return None, None
    
    orig_indices = np.where(train_labels == 0)[0]
    adv_indices = np.where(train_labels == 1)[0]
    orig_features = [train_features[i] for i in orig_indices]
    adv_features = [train_features[i] for i in adv_indices]
    
    print(f"Loaded {len(orig_features)} original and {len(adv_features)} adversarial training samples")
    
    # Load test data
    print(f"Loading test data from {test_data_path}")
    test_features, test_labels = load_features_from_json(test_data_path)
    
    if not test_features:
        print("No valid test features found. Skipping evaluation.")
        return None, None
    
    test_orig_indices = np.where(test_labels == 0)[0]
    test_adv_indices = np.where(test_labels == 1)[0]
    test_orig_features = [test_features[i] for i in test_orig_indices]
    test_adv_features = [test_features[i] for i in test_adv_indices]
    
    print(f"Loaded {len(test_orig_features)} original and {len(test_adv_features)} adversarial test samples")
    
    # Train and evaluate with single seed
    print(f"Training detector with {detection_method}...")
    
    start_time = time.time()
    detector = train_sensitivity_based_detector(
        orig_features=orig_features,
        adv_features=adv_features,
        method=detection_method,
        random_state=SEED
    )
    training_time = time.time() - start_time
    
    print(f"Detector trained successfully in {training_time:.2f} seconds")
    
    # Evaluate detector
    print("Evaluating detector on test data...")
    
    start_time = time.time()
    metrics_dict = evaluate_sensitivity_based_detector(test_orig_features, test_adv_features, detector)
    evaluation_time = time.time() - start_time
    
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Add additional information to metrics
    metrics_dict.update({
        "dataset": dataset or "unknown",
        "model": model or "unknown",
        "attack": attack or "unknown",
        "strategy": strategy,
        "detection_method": detection_method,
        "orig_train_samples": len(orig_features),
        "adv_train_samples": len(adv_features),
        "orig_test_samples": len(test_orig_features),
        "adv_test_samples": len(test_adv_features),
        "training_time": training_time,
        "evaluation_time": evaluation_time
    })
    
    # Save results
    results_df = pd.DataFrame([metrics_dict])
    results_path = results_dir / f"{strategy}_{detection_method}_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Log key metrics
    print("\nPerformance Metrics:")
    print(f"Accuracy: {metrics_dict.get('accuracy', 0.0):.4f}")
    print(f"F1 Score: {metrics_dict.get('f1', 0.0):.4f}")
    print(f"Precision: {metrics_dict.get('precision', 0.0):.4f}")
    print(f"Recall: {metrics_dict.get('recall', 0.0):.4f}")
    print(f"AUC: {metrics_dict.get('auc', 0.0):.4f}")
    print(f"Average Precision: {metrics_dict.get('avg_precision', 0.0):.4f}")
    
    print(f"Results saved to {results_path}")
    
    return results_df, detector

def main():
    """
    Main function to parse arguments and run the experiment.
    """
    parser = argparse.ArgumentParser(description='Train and evaluate adversarial detection models')
    parser.add_argument('--features_dir', type=str, default="data", help='Directory containing feature files')
    parser.add_argument('--output_dir', type=str, default="output/detector_results", help='Base directory to save results')
    parser.add_argument('--strategy', type=str, default="awi", help='Feature extraction strategy to use (default: awi)')
    parser.add_argument('--detection_method', type=str, default="bilstm", help='Detection method to use')
    parser.add_argument('--dataset', type=str, help='Dataset name (optional)')
    parser.add_argument('--model', type=str, help='Model name (optional)')
    parser.add_argument('--attack', type=str, help='Attack name (optional)')
    
    args = parser.parse_args()
    
    print("Starting detector training and evaluation experiment")
    print(f"Configuration: {vars(args)}")
    
    # Run the experiment
    results_df, detector = run_experiment(args)
    
    if results_df is not None:
        print("Experiment completed successfully")
    else:
        print("Experiment failed")

if __name__ == "__main__":
    main() 