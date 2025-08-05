"""
Sharpness-based adversarial detection using First-Order Stationary Condition (FOSC).

This module implements adversarial text detection based on the sharpness of the loss landscape
around input samples. The approach uses First-Order Stationary Condition (FOSC) to measure
how quickly the loss changes in the neighborhood of an input, with adversarial examples
typically exhibiting higher sharpness values.

The implementation is based on:
"Detecting Adversarial Samples through Sharpness of Loss Landscape" (ACL 2023)

Key Features:
- FOSC-based sharpness computation with adaptive step sizes
- Support for both L2 and L∞ perturbation norms
- Optional Frank-Wolfe gap computation as alternative convergence metric
- Batch processing for efficient evaluation
- Cross-domain generalization experiments

Classes:
    AdaptiveAdvSize: Manages adaptive adjustment of adversarial step sizes
    SharpnessDetector: Main detector class implementing FOSC-based detection

Functions:
    load_data_from_csv: Loads adversarial/original text pairs from CSV files
    load_pretrained_model: Resolves model paths for different datasets
    evaluate_detector: Computes detection performance metrics
    run_experiment: Executes single detection experiment
    run_generalization_experiment: Runs cross-domain generalization studies
"""

import os
import json
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup logger
logger = logging.getLogger("sharpness_detector")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def set_random_seed(seed: int):
    """
    Set random seed across all relevant libraries for reproducible results.
    
    Args:
        seed: Integer seed value for random number generation
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug(f"Set random seed to {seed}")


class AdaptiveAdvSize:
    """
    Manages adaptive adjustment of adversarial step sizes based on convergence metrics.
    
    This class implements adaptive step size control for adversarial perturbation generation,
    where the step size is adjusted based on convergence metrics like FOSC values or 
    Frank-Wolfe gaps. The adjustment follows a linear interpolation between initial and
    minimum step sizes based on the current convergence metric value.
    
    Attributes:
        _c0 (float): Initial convergence metric value for scaling
        _cmin (float): Minimum convergence metric threshold
        _alpha (float): Base adversarial step size
        warmup_steps (int): Number of initial steps to use full step size
    """

    def __init__(self, c_0, c_min, adv_size, warmup_steps):
        """
        Initialize adaptive step size controller.
        
        Args:
            c_0: Initial convergence metric value for normalization
            c_min: Minimum convergence threshold (target value)
            adv_size: Base adversarial step size (learning rate)
            warmup_steps: Number of initial steps to maintain full step size
        """
        self._c0 = c_0
        self._cmin = c_min
        self._alpha = adv_size
        self.warmup_steps = warmup_steps

    def update(self, ct, current_step):
        """
        Compute adaptive step size based on current convergence metric.
        
        The step size is linearly interpolated based on how close the current
        convergence metric is to the target minimum. During warmup, the step
        size is clamped to be at least the base step size.
        
        Args:
            ct: Current convergence metric value (FOSC or Frank-Wolfe gap)
            current_step: Current optimization step number
            
        Returns:
            torch.Tensor: Adaptive step size for current iteration
        """
        alpha_t = (self._cmin - ct) / (self._cmin - self._c0 + 1e-12) * self._alpha
        if current_step < self.warmup_steps:
            return torch.clamp(alpha_t, min=self._alpha)
        else:
            return torch.clamp(alpha_t, min=0)


class SharpnessDetector:
    """
    Adversarial text detection based on loss landscape sharpness analysis.
    
    This detector identifies adversarial examples by measuring the sharpness of the loss
    landscape around input samples using the First-Order Stationary Condition (FOSC).
    The key insight is that adversarial examples typically lie in sharper regions of
    the loss landscape compared to natural examples.
    
    The detection process involves:
    1. Computing adversarial perturbations using gradient-based optimization
    2. Measuring convergence using FOSC or Frank-Wolfe gap
    3. Using adaptive step sizes based on convergence metrics
    4. Classifying samples based on final loss or loss difference
    
    Key Features:
    - Support for both L2 and L∞ perturbation norms
    - Adaptive step size adjustment during perturbation generation
    - Optional Frank-Wolfe gap as alternative to FOSC
    - Configurable detection thresholds
    - Batch processing for efficiency
    
    Attributes:
        device (torch.device): Computing device (CPU/GPU)
        threshold (float): Detection threshold for binary classification
        model: Pre-trained transformer model for text classification
        tokenizer: Tokenizer corresponding to the model
        max_seq_length (int): Maximum sequence length for tokenization
        
    Hyperparameters:
        fosc_c (float): FOSC convergence threshold
        warmup_step (int): Number of warmup steps for adaptive sizing
        adv_steps (int): Number of adversarial optimization steps
        adv_init_mag (float): Initial perturbation magnitude
        adv_lr (float): Base learning rate for adversarial updates
        adv_max_norm (float): Maximum perturbation norm constraint
        adv_norm_type (str): Norm type for perturbations ('l2' or 'linf')
    """
    
    def __init__(self, model_name: str, device: str = 'cuda', threshold: float = 0.25, use_adaptive_lr: bool = True,
                 use_delta_loss: bool = False, adv_norm_type: str = 'l2',
                 use_fw_gap: bool = False):
        """
        Initialize the sharpness-based adversarial detector.
        
        Args:
            model_name: Path or identifier for pre-trained transformer model
            device: Computing device ('cuda' or 'cpu')
            threshold: Detection threshold for binary classification
            use_adaptive_lr: Whether to use adaptive step sizes during perturbation
            use_delta_loss: If True, use loss difference; if False, use perturbed loss only
            adv_norm_type: Perturbation norm type ('l2' or 'linf')
            use_fw_gap: Whether to use Frank-Wolfe gap instead of FOSC for convergence
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, attn_implementation="eager")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model.to(self.device)
        
        # Default hyperparameters from the paper
        self.fosc_c = 0.1  
        self.warmup_step = 4
        self.adv_steps = 4
        self.adv_init_mag = 0.05
        self.adv_lr = 0.03
        self.adv_max_norm = 0.2
        self.adv_norm_type = adv_norm_type 
        self.max_seq_length = None  
        
        # Enable adaptive step size based on FOSC
        self.do_adap_size = use_adaptive_lr
        self.use_delta_loss = use_delta_loss  
        self.use_fw_gap = use_fw_gap  
    
    def _determine_max_seq_length(self, texts: List[str]) -> int:
        """
        Determine optimal maximum sequence length for tokenization.
        
        Analyzes the provided texts to find the actual maximum token length,
        then caps it at 128 tokens.
        
        Args:
            texts: List of input texts to analyze for length determination
            
        Returns:
            int: Maximum sequence length, capped at 128 tokens
        """
        # Tokenize all texts to find max length
        max_length = 0
        for text in texts:
            # Tokenize without truncation to get actual length
            tokens = self.tokenizer(text, truncation=False, return_tensors='pt')
            length = tokens['input_ids'].size(1)
            max_length = max(max_length, length)
        
        # Cap at 128 as specified
        determined_length = min(max_length, 128)
        logger.info(f"Determined max_seq_length: {determined_length} (actual max: {max_length}, capped at 128, analyzed {len(texts)} texts)")
        
        return determined_length
    
    def compute_sharpness_features(self, texts: List[str], labels: List[int]) -> np.ndarray:
        """
        Compute sharpness-based features for adversarial detection.
        
        This method implements the core sharpness computation by:
        1. Computing clean loss on original inputs
        2. Generating adversarial perturbations using FOSC-guided optimization
        3. Computing perturbed loss after perturbation
        4. Returning either perturbed loss or loss difference as sharpness score
        
        The sharpness score captures how sensitive the model's predictions are
        to small perturbations in the embedding space.
        
        Args:
            texts: List of input texts to analyze
            labels: List of true class labels (used for clean loss computation)
            
        Returns:
            np.ndarray: Sharpness scores for each input text
        """
        if self.max_seq_length is None:
            self.max_seq_length = self._determine_max_seq_length(texts)
        
        self.model.zero_grad()
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        labels_tensor = torch.tensor(labels).to(self.device)
        
        # Get clean loss and predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = outputs.logits
            clean_loss = F.cross_entropy(clean_logits, labels_tensor, reduction='none')
            # Use model's predictions as targets for adversarial perturbation
            _, predicted_labels = clean_logits.max(dim=-1)
        
        # Generate perturbations targeting predicted labels (not ground truth)
        perturbed_loss = self._compute_fosc_loss(input_ids, attention_mask, predicted_labels)

        if self.use_delta_loss:
            # Use loss difference as sharpness measure
            sharpness = (perturbed_loss - clean_loss).detach().cpu().numpy()
        else:
            # Use absolute perturbed loss as sharpness measure
            sharpness = perturbed_loss.detach().cpu().numpy()

        return sharpness
    
    def _compute_fw_gap(self, delta: torch.Tensor, delta_grad: torch.Tensor, 
                       epsilon: float) -> torch.Tensor:
        """
        Compute Frank-Wolfe gap as convergence metric for adversarial optimization.
        
        The Frank-Wolfe gap measures the optimality gap in constrained optimization
        and serves as an alternative to FOSC for measuring convergence. The formula
        used is: g(x^k) = √ε||∇f(x^k)||_F - ⟨x^k - x^0, ∇f(x^k)⟩
        
        
        Args:
            delta: Current perturbation vector (x^k - x^0)
            delta_grad: Gradient of loss w.r.t. perturbation at current point
            epsilon: Perturbation radius constraint
            
        Returns:
            torch.Tensor: Frank-Wolfe gap values for each sample in the batch
        """
        # Reshape for batch computation
        delta_flat = delta.view(delta.size(0), -1)
        grad_flat = delta_grad.view(delta_grad.size(0), -1)
        
        # Compute gradient norm
        grad_norm = torch.norm(grad_flat, p=2, dim=1)
        
        # Compute inner product <x^k - x^0, ∇xf(x^k)>
        inner_product = torch.sum(delta_flat * grad_flat, dim=1)
        
        # Frank-Wolfe gap
        fw_gap = epsilon * grad_norm - inner_product
        
        return fw_gap
    
    def _compute_fosc_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial perturbations and compute perturbed loss using FOSC.
        
        This method implements the core adversarial perturbation generation process:
        1. Initialize perturbations in embedding space
        2. Iteratively optimize perturbations using gradient ascent
        3. Apply norm constraints and adaptive step sizing
        4. Use FOSC or Frank-Wolfe gap to guide convergence
        5. Return final loss on perturbed inputs
        
        The perturbations are generated to maximize the loss (find sharp regions)
        while staying within the specified norm constraints.
        
        Args:
            input_ids: Token IDs for input texts [batch_size, seq_len]
            attention_mask: Attention mask for padding [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size]
            
        Returns:
            torch.Tensor: Loss values after adversarial perturbation [batch_size]
        """
        self.model.eval()
        
        # Get embeddings
        word_embedding_layer = self.model.get_input_embeddings()
        embedding_init = word_embedding_layer(input_ids)
        
        # Initialize random perturbation within specified magnitude
        if self.adv_init_mag > 0:
            input_mask = attention_mask.to(embedding_init)
            input_lengths = torch.sum(input_mask, 1)
            
            if self.adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embedding_init.size(-1)
                magnitude = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * magnitude.view(-1, 1, 1))
            elif self.adv_norm_type == 'linf':
                delta = torch.zeros_like(embedding_init).uniform_(-self.adv_init_mag, self.adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embedding_init)
        
        # Iterative adversarial perturbation generation
        adap_adv_size = None
        epsilon = np.sqrt(self.adv_max_norm)  # Radius for convergence metric computation

        for step in range(self.adv_steps):
            delta.requires_grad_()
            
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            logits = self.model(**batch).logits
            
            losses = F.cross_entropy(logits, labels.squeeze(-1))
            loss = losses 
            loss.backward()
            
            delta_grad = delta.grad.clone().detach()

            if self.use_fw_gap:
                # Use Frank-Wolfe gap to measure convergence
                convergence_metric = self._compute_fw_gap(delta, delta_grad, epsilon)
                logger.debug(f"Step {step} - Avg FW gap: {convergence_metric.mean().item():.4f}")
            else:
                # Use First-Order Stationary Condition (FOSC) for convergence
                # FOSC formula: |⟨∇f(x), x-x₀⟩ + ε||∇f(x)||²|
                grad_norm_squared = torch.pow(torch.norm(delta_grad.view(delta_grad.size(0), -1).float(), p=2, dim=1), 2).detach()
                fosc = torch.ones_like(grad_norm_squared)
                
                for i in range(delta_grad.shape[0]):
                    fosc[i] = torch.abs(-torch.dot(delta_grad[i].view(-1), delta[i].view(-1)) + epsilon * grad_norm_squared[i])
                
                convergence_metric = fosc
                logger.debug(f"Step {step} - Avg FOSC: {fosc.mean().item():.4f}")

            if self.do_adap_size:
                if step == 0:
                    adap_adv_size = AdaptiveAdvSize(convergence_metric, self.fosc_c, self.adv_lr, self.warmup_step)

                adap_lr = adap_adv_size.update(convergence_metric, step).to(delta).view(-1, 1, 1)
            else:
                adap_lr = self.adv_lr

            # Prepare final perturbed embeddings for loss computation
            if step == self.adv_steps - 1:
                embedding_init = word_embedding_layer(input_ids)
                batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
                break

            # Update perturbation using gradient ascent
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adap_lr * delta_grad / denorm).detach()
                
                # Apply L2 norm constraint
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embedding_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adap_lr * delta_grad / denorm).detach()
                
                # Apply L∞ norm constraint
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            
            embedding_init = word_embedding_layer(input_ids)  # Refresh embeddings for next iteration
        
        # Compute final loss on perturbed inputs
        self.model.eval()
        outputs = self.model(**batch)
        logits = outputs.logits
        perturbed_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Perform backward pass for gradient computation
        loss = F.cross_entropy(logits, labels.squeeze(-1))
        loss.backward()

        return perturbed_loss
    
    def fit(self, texts: List[str], labels: List[int], adversarial_labels: List[int] = None):
        """
        Calibrate the detector on training data.
        
        This method computes sharpness scores on the calibration set to understand
        the distribution of scores for benign vs adversarial examples. However,
        it maintains a fixed threshold rather than learning one from the data.
        The original paper recommends doing a grid search to find the best threshold.
        
        The method primarily serves to:
        1. Set the maximum sequence length based on training data
        2. Log statistics about sharpness score distributions
        3. Validate the detector setup
        
        Args:
            texts: List of input texts for calibration
            labels: List of true class labels for each text
            adversarial_labels: Optional binary labels (0=benign, 1=adversarial)
                              for logging score distribution statistics
        """
        # Set max_seq_length if not already set
        if self.max_seq_length is None:
            self.max_seq_length = self._determine_max_seq_length(texts)
        
        # Compute sharpness scores on calibration data for analysis
        sharpness_scores = []
        batch_size = 32
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing sharpness scores"):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            scores = self.compute_sharpness_features(batch_texts, batch_labels)
            sharpness_scores.extend(scores)
        
        sharpness_scores = np.array(sharpness_scores)
        
        # Log statistics for reference but don't change threshold
        if adversarial_labels is not None:
            benign_scores = sharpness_scores[np.array(adversarial_labels) == 0]
            adv_scores = sharpness_scores[np.array(adversarial_labels) == 1]
            
            logger.info(f"Benign scores - Mean: {np.mean(benign_scores):.4f}, Std: {np.std(benign_scores):.4f}")
            logger.info(f"Adversarial scores - Mean: {np.mean(adv_scores):.4f}, Std: {np.std(adv_scores):.4f}")
        else:
            logger.info(f"Overall scores - Mean: {np.mean(sharpness_scores):.4f}, Std: {np.std(sharpness_scores):.4f}")
        
        # Maintain fixed threshold as per original paper methodology
        logger.info(f"Using hardcoded threshold: {self.threshold:.4f}")
    
    def predict(self, texts: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect adversarial examples using sharpness-based analysis.
        
        This method computes sharpness scores for the input texts and applies
        the detection threshold to classify them as benign (0) or adversarial (1).
        The sharpness scores can also be used for ranking or further analysis.
        
        Args:
            texts: List of input texts to classify
            labels: List of true class labels for each text
            
        Returns:
            Tuple containing:
                - predictions (np.ndarray): Binary classifications (0=benign, 1=adversarial)
                - sharpness_scores (np.ndarray): Raw sharpness scores for each text
        """

        if self.max_seq_length is None:
            self.max_seq_length = self._determine_max_seq_length(texts)
        
        # Compute sharpness scores
        sharpness_scores = []
        batch_size = 32
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            scores = self.compute_sharpness_features(batch_texts, batch_labels)
            sharpness_scores.extend(scores)
        
        sharpness_scores = np.array(sharpness_scores)
        
        # Predict based on threshold
        predictions = (sharpness_scores > self.threshold).astype(int)
        
        return predictions, sharpness_scores


def load_data_from_csv(csv_path: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Load paired original and adversarial texts from CSV file.
    
    Expects a CSV file with columns 'original_text' and 'adversarial_text'
    containing paired examples. Creates binary labels where 0 indicates
    original (benign) examples and 1 indicates adversarial examples.
    
    Args:
        csv_path: Path to CSV file containing text pairs
        
    Returns:
        Tuple containing:
            - original_texts: List of original (benign) text samples
            - adversarial_texts: List of corresponding adversarial examples
            - labels: Combined binary labels (0 for original, 1 for adversarial)
    """
    df = pd.read_csv(csv_path)
    
    # Get original and adversarial texts
    original_texts = df['original_text'].tolist()
    adversarial_texts = df['adversarial_text'].tolist()
    
    # Create combined lists
    all_texts = original_texts + adversarial_texts
    all_labels = [0] * len(original_texts) + [1] * len(adversarial_texts)
    
    return original_texts, adversarial_texts, all_labels


def load_pretrained_model(model_type: str, dataset: str):
    """
    Resolve model path/identifier for specified model type and dataset.
    
    Attempts to locate fine-tuned models in local directories first, then
    falls back to pre-trained models from HuggingFace Hub. This supports
    both RoBERTa and DeBERTa models across IMDB, AG News, and Yelp datasets.
    
    Args:
        model_type: Model architecture ('roberta' or 'deberta')
        dataset: Target dataset ('imdb', 'ag_news', or 'yelp')
        
    Returns:
        str: Model path (local directory) or HuggingFace model identifier
    """
    # Model name mappings
    model_map = {
        'roberta': {
            'imdb': 'textattack/roberta-base-imdb',
            'ag_news': 'textattack/roberta-base-ag-news', 
            'yelp': 'textattack/roberta-base-yelp-polarity'
        },
        'deberta': {
            'imdb': 'microsoft/deberta-base', 
            'ag_news': 'microsoft/deberta-base',
            'yelp': 'microsoft/deberta-base'
        }
    }
    
    # Check for local fine-tuned models first
    local_model_paths = {
        'deberta': {
            'imdb': 'models/deberta_imdb',
            'ag_news': 'models/deberta_ag_news',
            'yelp': 'models/deberta_yelp'
        },
        'roberta': {
            'imdb': 'models/roberta_imdb',
            'ag_news': 'models/roberta_ag_news',
            'yelp': 'models/roberta_yelp'
        }
    }
    
    # Try local path first
    local_path = local_model_paths.get(model_type, {}).get(dataset)
    if local_path and Path(local_path).exists():
        return local_path
    
    # Fall back to HuggingFace models
    return model_map.get(model_type, {}).get(dataset, f'{model_type}-base')


def evaluate_detector(detector: SharpnessDetector, texts: List[str], true_labels: List[int], 
                     adversarial_labels: List[int]) -> Dict[str, float]:
    """
    Evaluate detector performance using standard classification metrics.
    
    Computes evaluation metrics including accuracy, precision,
    recall, F1-score, AUC-ROC, and average precision. Also includes support
    counts for both benign and adversarial classes.
    
    Args:
        detector: Trained SharpnessDetector instance
        texts: List of input texts to evaluate
        true_labels: List of true class labels for classification task
        adversarial_labels: List of binary labels (0=benign, 1=adversarial)
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics:
            - accuracy, precision, recall, f1: Standard classification metrics
            - auc: Area under ROC curve
            - avg_precision: Average precision score
            - original_support, adversarial_support: Class counts
    """
    # Get predictions
    predictions, sharpness_scores = detector.predict(texts, true_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(adversarial_labels, predictions),
        'precision': precision_score(adversarial_labels, predictions, zero_division=0),
        'recall': recall_score(adversarial_labels, predictions, zero_division=0),
        'f1': f1_score(adversarial_labels, predictions, zero_division=0),
        'original_support': sum(1 for l in adversarial_labels if l == 0),
        'adversarial_support': sum(1 for l in adversarial_labels if l == 1)
    }
    
    # Calculate AUC and Average Precision
    try:
        # Use the sharpness_scores already computed by predict()
        metrics['auc'] = roc_auc_score(adversarial_labels, sharpness_scores)
        metrics['avg_precision'] = average_precision_score(adversarial_labels, sharpness_scores)
    except Exception as e:
        logger.warning(f"Could not compute AUC/AP: {e}")
        metrics['auc'] = 0.0
        metrics['avg_precision'] = 0.0
    
    return metrics


def run_experiment(config: Dict[str, Any]) -> Tuple[pd.DataFrame, SharpnessDetector]:
    """
    Execute a complete single-domain sharpness detection experiment.
    
    This function orchestrates a full experimental pipeline including:
    1. Loading and preprocessing data
    2. Initializing and training the detector
    3. Evaluating performance on test data
    4. Saving results and configuration
    
    Args:
        config: Dictionary containing experiment configuration with keys:
            - dataset: Dataset name ('imdb', 'ag_news', 'yelp')
            - model: Model type ('roberta', 'deberta')
            - attack: Attack method ('textfooler', 'bert-attack', 'deepwordbug')
            - data_dir: Directory containing data files
            - output_dir: Directory for saving results
            - eval_seed_base: Random seed for reproducibility
            
    Returns:
        Tuple containing:
            - results_df: DataFrame with evaluation metrics and metadata
            - detector: Trained SharpnessDetector instance
    """
    # Extract configuration
    dataset = config['dataset']
    model = config['model']
    attack = config['attack']
    data_dir = config.get('data_dir', 'data')
    output_dir = config.get('output_dir', 'output/sharpness_results')
    seed = config.get('eval_seed_base', 42)
    
    # Set random seed
    set_random_seed(seed)
    
    # Create output directory
    experiment_output_dir = Path(output_dir) / dataset / model / attack / 'sharpness'
    os.makedirs(experiment_output_dir, exist_ok=True)
    
    # Create subdirectories
    results_dir = experiment_output_dir / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    config_path = experiment_output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Running experiment: {dataset}/{model}/{attack}")
    
    # Load model path
    model_name = load_pretrained_model(model, dataset)
    logger.info(f"Using model: {model_name}")
    
    # Load data paths
    cal_data_path = Path(data_dir) / dataset / model / attack / 'calibration_data.csv'
    test_data_path = Path(data_dir) / dataset / model / attack / 'test_data.csv'
    
    if not cal_data_path.exists() or not test_data_path.exists():
        logger.error(f"Data files not found for {dataset}/{model}/{attack}")
        logger.error(f"Looking for: {cal_data_path} and {test_data_path}")
        return None, None
    
    # Load calibration data
    logger.info("Loading calibration data...")
    cal_orig_texts, cal_adv_texts, _ = load_data_from_csv(cal_data_path)
    
    # Combine original and adversarial texts for training
    cal_all_texts = cal_orig_texts + cal_adv_texts
    cal_adv_labels = [0] * len(cal_orig_texts) + [1] * len(cal_adv_texts)
    
    # Load test data
    logger.info("Loading test data...")
    test_orig_texts, test_adv_texts, _ = load_data_from_csv(test_data_path)
    
    # Combine for testing
    test_all_texts = test_orig_texts + test_adv_texts
    test_adv_labels = [0] * len(test_orig_texts) + [1] * len(test_adv_texts)
    
    logger.info(f"Calibration data: {len(cal_all_texts)} samples ({len(cal_orig_texts)} orig, {len(cal_adv_texts)} adv)")
    logger.info(f"Test data: {len(test_all_texts)} samples ({len(test_orig_texts)} orig, {len(test_adv_texts)} adv)")
    
    logger.info(f"Running with seed {seed}")
    
    # Initialize detector
    detector = SharpnessDetector(model_name)
    
    # Train (fit threshold)
    start_time = time.time()
    # Get true labels for calibration set
    cal_true_labels = []
    batch_size = 32
    detector.model.eval()
    
    # Set max_seq_length based on calibration data
    if detector.max_seq_length is None:
        detector.max_seq_length = detector._determine_max_seq_length(cal_all_texts)
    
    with torch.no_grad():
        for i in range(0, len(cal_all_texts), batch_size):
            batch_texts = cal_all_texts[i:i+batch_size]
            encodings = detector.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=detector.max_seq_length,
                return_tensors='pt'
            )
            outputs = detector.model(
                input_ids=encodings['input_ids'].to(detector.device),
                attention_mask=encodings['attention_mask'].to(detector.device)
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            cal_true_labels.extend(preds)
    
    detector.fit(cal_all_texts, cal_true_labels, cal_adv_labels)
    training_time = time.time() - start_time
    
    # Evaluate
    start_time = time.time()
    # Get true labels for test set
    test_true_labels = []
    # Ensure max_seq_length is set (should already be set from training)
    if detector.max_seq_length is None:
        all_texts_for_length = cal_all_texts + test_all_texts
        detector.max_seq_length = detector._determine_max_seq_length(all_texts_for_length)
    
    with torch.no_grad():
        for i in range(0, len(test_all_texts), batch_size):
            batch_texts = test_all_texts[i:i+batch_size]
            encodings = detector.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=detector.max_seq_length,
                return_tensors='pt'
            )
            outputs = detector.model(
                input_ids=encodings['input_ids'].to(detector.device),
                attention_mask=encodings['attention_mask'].to(detector.device)
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            test_true_labels.extend(preds)
    
    metrics = evaluate_detector(detector, test_all_texts, test_true_labels, test_adv_labels)
    evaluation_time = time.time() - start_time
    
    # Add metadata with consistent timing fields
    metrics.update({
        "detection_method": "fosc",
        "analysis_type": "sharpness",
        "random_seed": seed,
        "seed_run": 1,
        "cal_time_total_s": training_time,
        "test_time_total_s": evaluation_time,
        "cal_time_per_sample_s": training_time / len(cal_all_texts),
        "test_time_per_sample_s": evaluation_time / len(test_all_texts),
        "dataset": dataset,
        "model": model,
        "attack": attack,
        "strategy": "sharpness",
        "orig_train_samples": len(cal_orig_texts),
        "adv_train_samples": len(cal_adv_texts),
        "orig_test_samples": len(test_orig_texts),
        "adv_test_samples": len(test_adv_texts),
        "is_summary": False
    })
    
    # Save results
    results_df = pd.DataFrame([metrics])
    results_path = results_dir / 'detection_results_single_seed.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Log results
    logger.info(f"Seed {seed} - Acc: {metrics['accuracy']:.4f}, "
               f"F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    
    return results_df, detector


def run_generalization_experiment(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Execute cross-domain generalization experiments for sharpness detection.
    
    This function implements comprehensive generalization testing by training
    a detector on one domain and evaluating on potentially different domains.
    Supports three types of generalization experiments:
    
    1. Cross-dataset: Train on one dataset, test on different datasets
    2. Cross-attack: Train on one attack method, test on different attacks
    3. Cross-encoder: Train on one model type, test on different model types
    
    The function aggregates training data from multiple specified combinations
    and evaluates on both in-domain and out-of-domain test sets.
    
    Args:
        config: Dictionary containing experiment configuration with keys:
            - experiment_type: Type of generalization ('cross_dataset', 'cross_attack', 'cross_encoder')
            - train_datasets/models/attacks: Lists of training domain components
            - test_datasets/models/attacks: Lists for out-of-domain testing
            - data_dir: Directory containing data files
            - output_dir: Directory for saving results
            - eval_seed_base: Random seed for reproducibility
            
    Returns:
        pd.DataFrame: Results containing metrics for all evaluated combinations
                     with domain type indicators (in_domain vs out_of_domain)
    """
    experiment_type = config.get('experiment_type', 'standard')
    train_datasets = config.get('train_datasets')
    train_models = config.get('train_models') 
    train_attacks = config.get('train_attacks')
    test_datasets = config.get('test_datasets')
    test_models = config.get('test_models')
    test_attacks = config.get('test_attacks')
    output_dir = config.get('output_dir', 'output/sharpness_results')
    data_dir = config.get('data_dir', 'data')
    seed = config.get('eval_seed_base', 42)  # Use configurable seed
    
    # Create output directory structure
    experiment_output_dir = Path(output_dir)
    os.makedirs(experiment_output_dir, exist_ok=True)
    results_dir = experiment_output_dir / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    config_path = experiment_output_dir / f'config_{experiment_type}_sharpness.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Running {experiment_type} experiment")
    logger.info(f"Train - DS: {train_datasets}, Models: {train_models}, Attacks: {train_attacks}")
    if experiment_type != 'standard':
        logger.info(f"Test - DS: {test_datasets}, Models: {test_models}, Attacks: {test_attacks}")
    
    all_results = []
    
    # Set seed
    logger.info(f"Using seed {seed}")
    set_random_seed(seed)
    
    # Collect training data from all specified combinations
    train_orig_texts_all = []
    train_adv_texts_all = []
    train_labels_all = []
    
    for dataset in train_datasets:
        for model in train_models:
            for attack in train_attacks:
                cal_path = Path(data_dir) / dataset / model / attack / 'calibration_data.csv'
                if cal_path.exists():
                    orig_texts, adv_texts, _ = load_data_from_csv(cal_path)
                    train_orig_texts_all.extend(orig_texts)
                    train_adv_texts_all.extend(adv_texts)
                    logger.info(f"Loaded {len(orig_texts)} samples from {dataset}/{model}/{attack}")
                else:
                    logger.warning(f"Calibration file not found: {cal_path}")
    
    if not train_orig_texts_all:
        logger.error("No training data found!")
        return pd.DataFrame()
        
    # Combine training data
    train_texts = train_orig_texts_all + train_adv_texts_all
    train_adv_labels = [0] * len(train_orig_texts_all) + [1] * len(train_adv_texts_all)
    
    # Train detector on first model from train_models
    model_name = load_pretrained_model(train_models[0], train_datasets[0])
    detector = SharpnessDetector(model_name)
    
    # Set max_seq_length based on training data
    if detector.max_seq_length is None:
        detector.max_seq_length = detector._determine_max_seq_length(train_texts)
    
    # Get predicted labels for training
    train_true_labels = []
    batch_size = 32
    detector.model.eval()
    
    with torch.no_grad():
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i+batch_size]
            encodings = detector.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=detector.max_seq_length,
                return_tensors='pt'
            )
            outputs = detector.model(
                input_ids=encodings['input_ids'].to(detector.device),
                attention_mask=encodings['attention_mask'].to(detector.device)
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            train_true_labels.extend(preds)
    
    # Fit detector
    start_time = time.time()
    detector.fit(train_texts, train_true_labels, train_adv_labels)
    train_time = time.time() - start_time
    
    # Evaluate on test sets
    evaluation_configs = []
    
    # In-domain evaluation
    for dataset in train_datasets:
        for model in train_models:
            for attack in train_attacks:
                test_path = Path(data_dir) / dataset / model / attack / 'test_data.csv'
                if test_path.exists():
                    evaluation_configs.append({
                        'path': test_path,
                        'dataset': dataset,
                        'model': model,
                        'attack': attack,
                        'domain_type': 'in_domain'
                    })
    
    # Out-of-domain evaluation (if applicable)
    if experiment_type != 'standard':
        # Determine OOD test sets based on experiment type
        ood_datasets = test_datasets if test_datasets else train_datasets
        ood_models = test_models if test_models else train_models
        ood_attacks = test_attacks if test_attacks else train_attacks
        
        if experiment_type == 'cross_dataset':
            ood_datasets = test_datasets
        elif experiment_type == 'cross_attack':
            ood_attacks = test_attacks
        elif experiment_type == 'cross_encoder':
            ood_models = test_models
                
        for dataset in ood_datasets:
            for model in ood_models:
                for attack in ood_attacks:
                    test_path = Path(data_dir) / dataset / model / attack / 'test_data.csv'
                    if test_path.exists():
                        # Check if this is truly OOD
                        is_ood = False
                        if experiment_type == 'cross_dataset' and dataset not in train_datasets:
                            is_ood = True
                        elif experiment_type == 'cross_attack' and attack not in train_attacks:
                            is_ood = True
                        elif experiment_type == 'cross_encoder' and model not in train_models:
                            is_ood = True
                                
                        if is_ood:
                            evaluation_configs.append({
                                'path': test_path,
                                'dataset': dataset,
                                'model': model,
                                'attack': attack,
                                'domain_type': 'out_of_domain'
                            })
    
    # Run evaluations
    for eval_config in evaluation_configs:
        test_orig_texts, test_adv_texts, _ = load_data_from_csv(eval_config['path'])
        test_texts = test_orig_texts + test_adv_texts
        test_adv_labels = [0] * len(test_orig_texts) + [1] * len(test_adv_texts)
        
        # Get true labels (max_seq_length should already be set from training)
        test_true_labels = []
        with torch.no_grad():
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i+batch_size]
                encodings = detector.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=detector.max_seq_length,
                    return_tensors='pt'
                )
                outputs = detector.model(
                    input_ids=encodings['input_ids'].to(detector.device),
                    attention_mask=encodings['attention_mask'].to(detector.device)
                )
                preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                test_true_labels.extend(preds)
        
        # Evaluate
        start_time = time.time()
        metrics = evaluate_detector(detector, test_texts, test_true_labels, test_adv_labels)
        eval_time = time.time() - start_time
        
        # Add metadata
        metrics.update({
            'experiment_type': experiment_type,
            'seed': seed,
            'domain_type': eval_config['domain_type'],
            'test_dataset': eval_config['dataset'],
            'test_model': eval_config['model'],
            'test_attack': eval_config['attack'],
            'train_datasets': ','.join(train_datasets),
            'train_models': ','.join(train_models),
            'train_attacks': ','.join(train_attacks),
            'detection_method': 'sharpness',
            'training_time_sec': train_time,
            'evaluation_time_sec': eval_time,
            'orig_train_samples': len(train_orig_texts_all),
            'adv_train_samples': len(train_adv_texts_all),
            'orig_test_samples': len(test_orig_texts),
            'adv_test_samples': len(test_adv_texts)
        })
        
        all_results.append(metrics)
        
        logger.info(f"{eval_config['domain_type']} | {eval_config['dataset']}/{eval_config['model']}/{eval_config['attack']} | "
                   f"F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    
    # Save all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = results_dir / f'all_results_{experiment_type}_sharpness.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved all results to {results_path}")
        
        return results_df
    else:
        logger.error("No results collected!")
        return pd.DataFrame()


def main():
    """
    Command-line interface for running sharpness-based adversarial detection experiments.
    """
    parser = argparse.ArgumentParser(description='Run sharpness-based adversarial detection')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='output/sharpness_results', help='Output directory')
    
    # Standard single experiment args
    parser.add_argument('--dataset', type=str, help='Dataset name (imdb, yelp, ag_news)')
    parser.add_argument('--model', type=str, help='Model name (roberta, deberta)')
    parser.add_argument('--attack', type=str, help='Attack name (textfooler, bert-attack, deepwordbug)')
    
    # Generalization experiment args
    parser.add_argument('--experiment_type', type=str, choices=['standard', 'cross_dataset', 'cross_attack', 'cross_encoder'],
                       default='standard', help='Type of experiment to run')
    parser.add_argument('--train_datasets', type=str, help='Comma-separated list of datasets for training')
    parser.add_argument('--train_models', type=str, help='Comma-separated list of models for training')
    parser.add_argument('--train_attacks', type=str, help='Comma-separated list of attacks for training')
    parser.add_argument('--test_datasets', type=str, help='Comma-separated list of datasets for OOD testing')
    parser.add_argument('--test_models', type=str, help='Comma-separated list of models for OOD testing')
    parser.add_argument('--test_attacks', type=str, help='Comma-separated list of attacks for OOD testing')
    
    # Evaluation parameters
    parser.add_argument('--eval_seed_base', type=int, default=42, help='Base seed for evaluation')
    
    args = parser.parse_args()
    
    # Parse list arguments
    def parse_list_arg(arg_val):
        if arg_val:
            return [item.strip() for item in arg_val.split(',')]
        return None
    
    # Handle experiment type
    if args.experiment_type == 'standard':
        # Use single experiment args
        config = {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'dataset': args.dataset,
            'model': args.model,
            'attack': args.attack,
            'eval_seed_base': args.eval_seed_base
        }
        # Run original single experiment
        run_experiment(config)
    else:
        # Run generalization experiment
        config = {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'experiment_type': args.experiment_type,
            'train_datasets': parse_list_arg(args.train_datasets) or ([args.dataset] if args.dataset else None),
            'train_models': parse_list_arg(args.train_models) or ([args.model] if args.model else None),
            'train_attacks': parse_list_arg(args.train_attacks) or ([args.attack] if args.attack else None),
            'test_datasets': parse_list_arg(args.test_datasets),
            'test_models': parse_list_arg(args.test_models),
            'test_attacks': parse_list_arg(args.test_attacks),
            'eval_seed_base': args.eval_seed_base
        }
        
        # Validate config based on experiment type
        missing_params = []
        
        # Always require training parameters
        if not (config['train_datasets'] and len(config['train_datasets']) > 0):
            missing_params.append('train_datasets')
        if not (config['train_models'] and len(config['train_models']) > 0):
            missing_params.append('train_models')
        if not (config['train_attacks'] and len(config['train_attacks']) > 0):
            missing_params.append('train_attacks')
        
        # Check experiment-specific requirements
        if config['experiment_type'] == 'cross_dataset':
            if not (config['test_datasets'] and len(config['test_datasets']) > 0):
                missing_params.append('test_datasets (required for cross_dataset)')
        elif config['experiment_type'] == 'cross_encoder':
            if not (config['test_models'] and len(config['test_models']) > 0):
                missing_params.append('test_models (required for cross_encoder)')
        elif config['experiment_type'] == 'cross_attack':
            if not (config['test_attacks'] and len(config['test_attacks']) > 0):
                missing_params.append('test_attacks (required for cross_attack)')
        
        if missing_params:
            parser.error(f"Missing required parameters: {', '.join(missing_params)}")
        
        run_generalization_experiment(config)


if __name__ == '__main__':
    main() 