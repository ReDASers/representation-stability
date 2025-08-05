"""Training and evaluation pipeline for adversarial text detection models.

This module provides a unified interface for training and evaluating BiLSTM-based
adversarial detection models using sensitivity features. It handles data preprocessing,
model training with validation splits, and performance evaluation.

The module supports:
- Sequential sensitivity feature processing with 128-token limit
- Automatic train/validation splitting with stratification
- External validation data handling
- Metric evaluation (accuracy, precision, recall, F1, AUC)
"""

from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from ..models.bilstm import create_bilstm_classifier

# Default random seed for reproducible experiments
SEED = 42

# Configuration for available detection models
MODEL_CONFIGS = {
    "bilstm": {
        "fallback_fn": lambda: create_bilstm_classifier(),
        "has_feature_importances": False,  # BiLSTM doesn't provide feature importance
    },
}

def train_sensitivity_based_detector(
    orig_features: Union[List[np.ndarray], np.ndarray],
    adv_features: Union[List[np.ndarray], np.ndarray], 
    val_orig_features: Optional[Union[List[np.ndarray], np.ndarray]] = None,
    val_adv_features: Optional[Union[List[np.ndarray], np.ndarray]] = None,
    method: str = "bilstm",
    val_split: float = 0.2,
    random_state: int = SEED
) -> Dict[str, Any]:
    """Train an adversarial text detection model using sensitivity features.
    
    Creates and trains a BiLSTM-based classifier to distinguish between clean and
    adversarial text samples based on their gradient attribution patterns. Handles
    data preprocessing, validation splitting, and model training with early stopping.
    
    Args:
        orig_features: Sensitivity feature sequences from clean/original text samples.
            Can be list of arrays (variable length) or fixed arrays.
        adv_features: Sensitivity feature sequences from adversarial text samples.
            Must match format of orig_features.
        val_orig_features: Optional validation data for clean samples.
            If provided, used instead of internal train/val split.
        val_adv_features: Optional validation data for adversarial samples.
            Must be provided together with val_orig_features.
        method: Detection model type. Currently supports 'bilstm'.
        val_split: Fraction of training data to use for validation (0.0-1.0).
            Only used if external validation data not provided.
        random_state: Random seed for reproducible data splits and model training.
            
    Returns:
        Dictionary containing training results:
            - 'model': Trained detector model ready for inference
            - 'validation_data': Tuple of (X_val, y_val) for evaluation
            - 'is_sequence_features': Boolean indicating sequential data format
            
    Note:
        All sequences are automatically truncated to 128 tokens for memory efficiency
        and consistent processing. The function handles both fixed-size arrays and
        variable-length sequence lists.
        
    Raises:
        ValueError: If validation features are partially provided (only one of the pair).
    """
    print(f"Training sensitivity-based detector with {method} classifier (seed={random_state})")

    # Validate input data
    if len(orig_features) == 0 or len(adv_features) == 0:
        print("Warning: Empty feature arrays provided to detector training.")
        return {"dummy": "No model trained due to empty features"}

    # Apply sequence length limit for memory efficiency and consistency
    def truncate_sequences(features, max_length=128):
        """Truncate sequences to maximum length."""
        if isinstance(features, list):
            return [feat[:max_length] if feat.shape[0] > max_length else feat for feat in features]
        return features
    
    orig_features = truncate_sequences(orig_features)
    adv_features = truncate_sequences(adv_features)
    
    if val_orig_features is not None:
        val_orig_features = truncate_sequences(val_orig_features)
    if val_adv_features is not None:
        val_adv_features = truncate_sequences(val_adv_features)
    
    print("Applied 128-token sequence length limit for memory efficiency")

    # Detect and standardize feature format
    is_sequence_features = isinstance(orig_features, list) and isinstance(adv_features, list)
    
    if not is_sequence_features:
        print("Converting fixed-size features to sequence format for BiLSTM compatibility")
        # Convert arrays to list of sequences with single channel
        if not isinstance(orig_features, list):
            orig_features = [np.expand_dims(feat, axis=1) for feat in orig_features]
        if not isinstance(adv_features, list):
            adv_features = [np.expand_dims(feat, axis=1) for feat in adv_features]
        is_sequence_features = True
    
    # Combine training data: 0=clean, 1=adversarial
    X_train = orig_features + adv_features
    y_train = np.concatenate([
        np.zeros(len(orig_features)),   # Clean samples labeled as 0
        np.ones(len(adv_features))      # Adversarial samples labeled as 1
    ])

    # Setup validation data
    if val_orig_features is not None and val_adv_features is not None:
        # Use provided external validation data
        if not isinstance(val_orig_features, list):
            val_orig_features = [np.expand_dims(feat, axis=1) for feat in val_orig_features]
        if not isinstance(val_adv_features, list):
            val_adv_features = [np.expand_dims(feat, axis=1) for feat in val_adv_features]
        
        X_val = val_orig_features + val_adv_features
        y_val = np.concatenate([
            np.zeros(len(val_orig_features)),
            np.ones(len(val_adv_features))
        ])
        print(f"Using external validation data: {len(X_val)} samples")
    else:
        # Create stratified train/validation split
        train_indices, val_indices = train_test_split(
            np.arange(len(X_train)), 
            test_size=val_split, 
            random_state=random_state, 
            stratify=y_train  # Maintain class balance in splits
        )
        X_val = [X_train[i] for i in val_indices]
        y_val = y_train[val_indices]
        X_train = [X_train[i] for i in train_indices]
        y_train = y_train[train_indices]
        print(f"Created stratified validation split ({val_split*100:.0f}%): {len(X_val)} samples")

    # Shuffle training and validation data for better training dynamics
    rng = np.random.RandomState(random_state)
    
    # Shuffle training data
    train_shuffle_idx = rng.permutation(len(y_train))
    X_train = [X_train[i] for i in train_shuffle_idx]
    y_train = y_train[train_shuffle_idx]
    
    # Shuffle validation data
    val_shuffle_idx = rng.permutation(len(y_val))
    X_val = [X_val[i] for i in val_shuffle_idx]
    y_val = y_val[val_shuffle_idx]

    # Initialize and configure the detection model
    if method not in MODEL_CONFIGS:
        print(f"Unknown method '{method}', defaulting to 'bilstm'")
        method = "bilstm"

    model = create_bilstm_classifier(random_state=random_state)
    
    # Configure external validation for early stopping
    if hasattr(model, "_no_internal_val_split"):
        model._no_internal_val_split = True  # Use our validation split
        model.X_val = X_val
        model.y_val = y_val
    
    print(f"Training {method} model with early stopping...")
    model.fit(X_train, y_train)
    
    # Evaluate model performance on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average='binary'
    )
    
    # Compute combined metric (average of F1 and accuracy)
    val_combined = (val_f1 + val_accuracy) / 2.0
    
    print(
        f"Validation performance - Accuracy: {val_accuracy:.4f}, "
        f"F1: {val_f1:.4f}, Combined: {val_combined:.4f}"
    )
    
    # Calculate AUC if model supports probability prediction
    val_auc = 0.5  # Default to random classifier
    if hasattr(model, "predict_proba") and callable(model.predict_proba):
        try:
            y_val_proba = model.predict_proba(X_val)[:, 1]  # Probability of adversarial class
            val_auc = roc_auc_score(y_val, y_val_proba)
            print(f"Validation AUC: {val_auc:.4f}")
        except Exception as e:
            print(f"Could not calculate AUC: {e}")

    return {
        "model": model,                                    # Trained detection model
        "validation_data": (X_val, y_val),              # Validation set for evaluation
        "is_sequence_features": is_sequence_features     # Data format indicator
    }

def evaluate_sensitivity_based_detector(
    test_orig_features: Union[List[np.ndarray], np.ndarray],
    test_adv_features: Union[List[np.ndarray], np.ndarray],
    detector_models: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate trained adversarial detection model on test data.
    
    Applies the trained detector to unseen test samples and computes comprehensive
    performance metrics. Handles data preprocessing to match training format and
    provides detailed classification statistics.
    
    Args:
        test_orig_features: Sensitivity feature sequences from clean test samples.
            Should match the format used during training.
        test_adv_features: Sensitivity feature sequences from adversarial test samples.
            Should match the format used during training.
        detector_models: Dictionary containing trained model artifacts from
            train_sensitivity_based_detector(), including 'model' key.
            
    Returns:
        Dictionary containing comprehensive test metrics:
            - 'accuracy': Overall classification accuracy (0.0-1.0)
            - 'precision': Precision for adversarial detection (0.0-1.0)
            - 'recall': Recall for adversarial detection (0.0-1.0)
            - 'f1': F1-score for adversarial detection (0.0-1.0)
            - 'auc': Area under ROC curve if model supports probabilities (0.0-1.0)
            - 'original_support': Number of clean samples in test set
            - 'adversarial_support': Number of adversarial samples in test set
            
    Note:
        Test sequences are automatically truncated to 128 tokens to match training.
        Returns zeros for all metrics if evaluation fails or no valid model provided.
    """
    # Validate test data
    if len(test_orig_features) == 0 or len(test_adv_features) == 0:
        print("Warning: Empty test feature arrays provided to detector evaluation.")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.5,
            "original_support": 0,
            "adversarial_support": 0
        }
    
    # Preprocess test features to match training format
    def preprocess_test_features(features, max_length=128):
        """Convert and truncate features to match training format."""
        if isinstance(features, list):
            # Truncate sequences that exceed max length
            return [feat[:max_length] if feat.shape[0] > max_length else feat for feat in features]
        elif isinstance(features, np.ndarray):
            # Convert array format to list of sequences with single channel
            return [np.expand_dims(feat, axis=1) for feat in features]
        return features
    
    test_orig_features = preprocess_test_features(test_orig_features)
    test_adv_features = preprocess_test_features(test_adv_features)
    
    # Combine test data with labels: 0=clean, 1=adversarial
    test_X = test_orig_features + test_adv_features
    test_y = np.concatenate([
        np.zeros(len(test_orig_features)),   # Clean samples
        np.ones(len(test_adv_features))      # Adversarial samples
    ])

    # Shuffle test data for consistent evaluation
    shuffle_idx = np.random.RandomState(SEED).permutation(len(test_y))
    test_X = [test_X[i] for i in shuffle_idx]
    test_y = test_y[shuffle_idx]
    
    # Extract model from training results
    model = detector_models.get("model")
    
    if model is None or isinstance(model, str):
        print("Warning: No valid model found in detector_models dictionary")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.5,
            "original_support": len(test_orig_features),
            "adversarial_support": len(test_adv_features)
        }
    
    # Perform model evaluation
    if not (hasattr(model, "predict") and callable(model.predict)):
        print("Warning: Model is not properly fitted and cannot make predictions")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.5,
            "original_support": len(test_orig_features),
            "adversarial_support": len(test_adv_features)
        }
    
    try:
        # Generate predictions
        y_pred = model.predict(test_X)
        
        # Calculate primary classification metrics
        accuracy = accuracy_score(test_y, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            test_y, y_pred, average=None  # Return per-class metrics
        )
        
        # Handle per-class metrics safely (precision, recall, f1, support are arrays)
        if support is not None and len(support) >= 2:
            original_support = int(support[0])
            adversarial_support = int(support[1])
            # Use adversarial class (class 1) metrics for binary classification
            binary_precision = float(precision[1]) if len(precision) > 1 else float(precision[0])
            binary_recall = float(recall[1]) if len(recall) > 1 else float(recall[0])
            binary_f1 = float(f1[1]) if len(f1) > 1 else float(f1[0])
        else:
            # Fall back to calculating from test data
            original_support = int(np.sum(test_y == 0))
            adversarial_support = int(np.sum(test_y == 1))
            binary_precision = float(precision[1]) if hasattr(precision, '__len__') and len(precision) > 1 else 0.0
            binary_recall = float(recall[1]) if hasattr(recall, '__len__') and len(recall) > 1 else 0.0
            binary_f1 = float(f1[1]) if hasattr(f1, '__len__') and len(f1) > 1 else 0.0
        
        # Prepare results dictionary
        results = {
            "accuracy": float(accuracy),
            "precision": binary_precision,
            "recall": binary_recall,
            "f1": binary_f1,
            "original_support": original_support,
            "adversarial_support": adversarial_support
        }
        
        # Calculate AUC if model supports probability prediction
        if hasattr(model, "predict_proba") and callable(model.predict_proba):
            try:
                y_scores = model.predict_proba(test_X)
                # Extract probability for positive class (adversarial)
                if y_scores.ndim > 1 and y_scores.shape[1] > 1:
                    y_scores = y_scores[:, 1]
                
                auc = roc_auc_score(test_y, y_scores)
                results["auc"] = float(auc)
                
                print(
                    f"Test Results - Accuracy: {accuracy:.4f}, F1: {binary_f1:.4f}, "
                    f"Precision: {binary_precision:.4f}, Recall: {binary_recall:.4f}, AUC: {auc:.4f}"
                )
            except (AttributeError, ValueError, IndexError) as e:
                print(f"Could not calculate AUC: {e}")
                results["auc"] = 0.5  # Random classifier baseline
                print(
                    f"Test Results - Accuracy: {accuracy:.4f}, F1: {binary_f1:.4f}, "
                    f"Precision: {binary_precision:.4f}, Recall: {binary_recall:.4f}"
                )
        else:
            results["auc"] = 0.5
            print(
                f"Test Results - Accuracy: {accuracy:.4f}, F1: {binary_f1:.4f}, "
                f"Precision: {binary_precision:.4f}, Recall: {binary_recall:.4f}"
            )
            
    except Exception as e:
        print(f"Warning: Error during model evaluation: {e}")
        results = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.5,
            "original_support": len(test_orig_features),
            "adversarial_support": len(test_adv_features)
        }
    
    return results