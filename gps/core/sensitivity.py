"""Adversarial detection utilities for sensitivity analysis and feature extraction.

This module provides utilities for processing sensitivity maps, extracting statistical
features, and analyzing adversarial attack characteristics. It includes tools for:

- Computing embedding distances with various metrics
- Extracting distributional and morphological features from sensitivity sequences
- Processing sequential sensitivity and importance data
- Analyzing attack logs and perturbed text patterns

The module is designed to work with output from attribution methods
and supports both statistical feature extraction and raw sequential data processing
for classification models.
"""

from __future__ import annotations

import logging
import re
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

from ..utils.text_processing import split_into_words

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Numerical stability constant for avoiding division by zero
EPS = 1e-10

# Feature extraction configuration
EXPECTED_FEATURES = 15  # Total number of statistical features extracted
NONZERO_FEATURES = 8    # Number of features expected to be non-zero
MAX_SAFE_GINI_N = 10_000  # Maximum array size for standard Gini computation

# Standard feature names for sensitivity analysis (15 total)
BASE_FEATURE_NAMES = [
    # Basic distribution statistics (4 features)
    "Mean", "Std Dev", "Max", "Range",
    # Robust statistics and concentration measures (2 features)
    "IQR", "Gini",
    # Information theory (1 feature)
    "Shannon Entropy",
    # Shape and trend characteristics (3 features)
    "Slope", "Curvature", "Normalized AUC",
    # Change rate measures (2 features)
    "Mean Abs Change", "Max Abs Change",
    # Peak detection features (3 features)
    "Peak Density", "Avg Peak Height", "Max Peak Height",
]


def compute_embedding_distance(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    *,
    distance_metric: Literal[
        "l2",
        "euclidean",
        "l1",
        "manhattan",
        "cosine",
    ] = "cosine",
) -> torch.Tensor:
    """Compute pairwise distances between embedding tensors.
    
    Calculates element-wise distances between two embedding tensors using the
    specified distance metric. Supports common distance measures used in
    adversarial detection and embedding analysis.
    
    Args:
        emb1: First embedding tensor with shape [..., embedding_dim].
        emb2: Second embedding tensor with identical shape to emb1.
        distance_metric: Distance measure to use. Options:
            - 'l2'/'euclidean': Euclidean (L2) distance
            - 'l1'/'manhattan': Manhattan (L1) distance  
            - 'cosine': Cosine distance (1 - cosine_similarity)
            
    Returns:
        Tensor of distances with shape [...] (last dimension removed).
        Values are non-negative floats.
        
    Raises:
        ValueError: If embeddings have different shapes or distance_metric is invalid.
        
    Examples:
        >>> emb1 = torch.randn(10, 128)  # 10 embeddings of dimension 128
        >>> emb2 = torch.randn(10, 128)
        >>> distances = compute_embedding_distance(emb1, emb2, distance_metric='cosine')
        >>> distances.shape
        torch.Size([10])
    """
    if emb1.shape != emb2.shape:
        raise ValueError("emb1 and emb2 must have identical shape, got " f"{emb1.shape} vs {emb2.shape}.")

    if distance_metric in {"l2", "euclidean"}:
        return torch.norm(emb1 - emb2, p=2, dim=-1)

    if distance_metric in {"l1", "manhattan"}:
        return torch.norm(emb1 - emb2, p=1, dim=-1)

    if distance_metric == "cosine":
        return 1.0 - F.cosine_similarity(emb1, emb2, dim=-1)

    raise ValueError(f"Unsupported distance_metric: {distance_metric}")


class _FeatureExtractor:
    """Internal class for extracting statistical features from 1D sequences.
    
    This class encapsulates the computation of various distributional, morphological,
    and information-theoretic features from sensitivity sequences. It's designed to
    be efficient for repeated feature extraction operations.
    
    The extracted features are commonly used in adversarial detection to characterize
    the distribution patterns of gradient-based attribution scores.
    
    Attributes:
        x: Input sequence as float array.
        n: Length of the input sequence.
    """

    def __init__(self, x: np.ndarray):
        """Initialize the feature extractor.
        
        Args:
            x: 1D numpy array containing the sequence to analyze.
        """
        self.x = x.astype(float, copy=False)  # Ensure float type without copying if possible
        self.n = len(x)

    def basic(self) -> list[float]:
        """Extract basic statistical features.
        
        Returns:
            List of 4 features: [mean, standard_deviation, maximum, range].
            Returns zeros for empty sequences.
        """
        if self.n == 0:
            return [0.0] * 4
        return [
            float(np.mean(self.x)),    # Mean (central tendency)
            float(np.std(self.x)),     # Standard deviation (spread)
            float(np.max(self.x)),     # Maximum value
            float(np.ptp(self.x))      # Range (peak-to-peak)
        ]

    def robust(self) -> list[float]:
        """Extract robust statistical measures.
        
        Computes statistics that are less sensitive to outliers than basic measures.
        
        Returns:
            List of 2 features: [IQR, Gini_coefficient].
            Returns zeros for empty or all-zero sequences.
        """
        if self.n == 0 or np.allclose(self.x, 0):
            return [0.0, 0.0]
        
        # Interquartile range: robust measure of spread
        q25, q75 = np.percentile(self.x, [25, 75])
        iqr = float(q75 - q25)
        
        # Gini coefficient: measure of inequality/concentration
        gini = self._gini()
        
        return [iqr, gini]

    def _gini(self) -> float:
        """Calculate Gini coefficient of inequality.
        
        The Gini coefficient measures how unequally values are distributed,
        ranging from 0 (perfect equality) to 1 (maximum inequality).
        
        Returns:
            Gini coefficient as float between 0 and 1.
        """
        if self.n <= 1:
            return 0.0
            
        # Use approximation for very large arrays to avoid memory issues
        if self.n > MAX_SAFE_GINI_N:
            # Alternative computation: mean absolute difference method
            abs_diffs = np.abs(self.x[:, None] - self.x).sum()
            return abs_diffs / (2 * self.n**2 * np.mean(self.x) + EPS)
        
        # Standard Gini computation using sorted values
        sorted_x = np.sort(self.x)
        cum = np.cumsum(sorted_x)
        # Gini = 1 - 2 * (sum of cumulative areas) / (n * total_area)
        return 1.0 - 2.0 * np.sum(cum) / (self.n * cum[-1])

    def entropy(self) -> float:
        """Calculate Shannon entropy of the sequence.
        
        Shannon entropy measures the information content or randomness in the
        distribution. Higher entropy indicates more uniform distributions.
        
        Returns:
            Shannon entropy in bits (base-2 logarithm).
            Returns 0 for sequences that sum to zero.
        """
        s = self.x.sum()
        if s < EPS:
            return 0.0
            
        # Convert to probability distribution
        p = self.x / s
        # Shannon entropy: -sum(p * log2(p))
        return float(-(p * np.log2(p + EPS)).sum())

    def shape(self) -> list[float]:
        """Extract shape and trend characteristics.
        
        Computes features that describe the overall shape and trend of the sequence.
        
        Returns:
            List of 3 features: [slope, curvature, normalized_AUC].
            Returns zeros for empty sequences.
        """
        if self.n == 0:
            return [0.0, 0.0, 0.0]
            
        # Overall slope: linear trend from first to last value
        slope = float((self.x[-1] - self.x[0]) / max(self.n - 1, 1))
        
        # Curvature: mean absolute second derivative (measures "wiggliness")
        curvature = float(np.mean(np.abs(np.diff(self.x, n=2)))) if self.n > 2 else 0.0
        
        # Normalized area under curve: average height
        auc = float(np.trapz(self.x) / self.n) if self.n > 1 else 0.0
        
        return [slope, curvature, auc]

    def change(self) -> list[float]:
        """Extract change rate characteristics.
        
        Measures how rapidly the sequence changes between adjacent points.
        
        Returns:
            List of 2 features: [mean_absolute_change, max_absolute_change].
            Returns zeros for sequences with fewer than 2 points.
        """
        if self.n < 2:
            return [0.0, 0.0]
            
        # First differences: change between adjacent points
        diffs = np.abs(np.diff(self.x))
        
        return [
            float(diffs.mean()),  # Average rate of change
            float(diffs.max())    # Maximum single-step change
        ]

    def peaks(self) -> list[float]:
        """Extract peak detection features.
        
        Identifies significant peaks in the sequence and computes statistics about them.
        Peaks are defined as local maxima above mean + std_dev.
        
        Returns:
            List of 3 features: [peak_density, avg_peak_height, max_peak_height].
            Returns zeros for peak heights when no peaks are found.
        """
        if self.n == 0:
            return [0.0, 0.0, 0.0]
            
        # Peak detection threshold: mean + 1 standard deviation
        threshold = self.x.mean() + self.x.std()
        peak_indices, _ = find_peaks(self.x, height=threshold)
        
        # Peak density: fraction of points that are peaks
        density = len(peak_indices) / self.n
        
        if peak_indices.size > 0:
            peak_heights = self.x[peak_indices]
            return [
                density,
                float(peak_heights.mean()),  # Average height of detected peaks
                float(peak_heights.max())    # Maximum peak height
            ]
        
        return [density, 0.0, 0.0]  # No peaks found


def extract_sensitivity_features(sensitivity_maps: list[list[float]]) -> np.ndarray:
    """Extract comprehensive statistical features from sensitivity sequences.
    
    Transforms variable-length sensitivity sequences into fixed-size feature vectors
    suitable for traditional machine learning classifiers. Extracts 15 features
    covering distribution, shape, change patterns, and peak characteristics.
    
    Args:
        sensitivity_maps: List of sensitivity sequences, where each sequence
            contains gradient attribution scores for one text sample.
        
    Returns:
        Feature matrix with shape [num_samples, 15] containing extracted features.
        Returns empty array with correct shape if input is empty.
        
    Note:
        Feature order matches BASE_FEATURE_NAMES constant:
        [Mean, Std Dev, Max, Range, IQR, Gini, Shannon Entropy, 
         Slope, Curvature, Normalized AUC, Mean Abs Change, Max Abs Change,
         Peak Density, Avg Peak Height, Max Peak Height]
    """
    if not sensitivity_maps:
        return np.empty((0, EXPECTED_FEATURES))

    features = []
    for smap in sensitivity_maps:
        # Extract all feature categories for this sequence
        extractor = _FeatureExtractor(np.asarray(smap))
        feature_vector = (
            extractor.basic() +      # 4 features: mean, std, max, range
            extractor.robust() +     # 2 features: IQR, Gini
            [extractor.entropy()] +  # 1 feature: Shannon entropy
            extractor.shape() +      # 3 features: slope, curvature, AUC
            extractor.change() +     # 2 features: mean/max absolute change
            extractor.peaks()        # 3 features: peak density, avg/max height
        )
        features.append(feature_vector)
        
    return np.asarray(features, dtype=float)

def extract_raw_sensitivity_values(
    sensitivity_maps: List[List[float]],
    top_n: int = None,
    *,
    zero_tol: float = 0.0,
    max_seq_length: int = None
) -> List[List[float]]:
    """Extract and filter sensitivity values while preserving sequence structure.
    
    Processes sensitivity maps by keeping only the most important values (by magnitude)
    while maintaining their original positions in the sequence. This is useful for
    focusing on the most salient features for adversarial detection.
    
    Args:
        sensitivity_maps: List of sensitivity sequences, one per text sample.
        top_n: Number of highest-magnitude values to retain per sequence.
            If None, keeps all values above zero_tol.
        zero_tol: Absolute value threshold below which values are considered zero.
        max_seq_length: Deprecated parameter, no longer used. Models now handle
            variable-length sequences directly.
        
    Returns:
        List of processed sensitivity sequences with same structure as input.
        Non-selected values are set to 0.0 but sequence lengths are preserved.
        
    Example:
        >>> sensitivity_maps = [[0.1, 0.8, 0.3, 0.05], [0.9, 0.2]]
        >>> result = extract_raw_sensitivity_values(sensitivity_maps, top_n=2)
        >>> # Keeps top 2 values per sequence: [[0.0, 0.8, 0.3, 0.0], [0.9, 0.2]]
    """
    if max_seq_length is not None:
        # Legacy parameter ignored - models now handle variable-length sequences
        pass
    
    processed_maps = []
    
    for smap in sensitivity_maps:
        processed_map = list(smap)  # Start with copy of original
        
        if top_n is not None and top_n > 0:
            # Find values above tolerance with their indices
            indexed_abs_vals = [
                (i, abs(val)) for i, val in enumerate(smap) 
                if abs(val) > zero_tol
            ]
            
            # If we have more significant values than top_n, filter to top values
            if len(indexed_abs_vals) > top_n:
                # Get indices of top_n highest magnitude values
                top_indices = [
                    idx for idx, _ in sorted(
                        indexed_abs_vals, 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:top_n]
                ]
                
                # Create mask for positions to keep
                top_indices_set = set(top_indices)
                
                # Zero out non-selected positions
                processed_map = [
                    val if i in top_indices_set else 0.0 
                    for i, val in enumerate(smap)
                ]
        
        processed_maps.append(processed_map)
    
    return processed_maps

def analyze_curve_characteristics(
    orig_curves: List[List[float]],
    adv_curves: List[List[float]],
) -> dict:
    """Analyze and compare characteristics between original and adversarial sensitivity curves.
    
    Performs comprehensive statistical analysis to identify differences between
    sensitivity patterns of clean and adversarial text samples. This analysis
    helps understand how adversarial attacks affect gradient attribution patterns.
    
    Args:
        orig_curves: List of sensitivity sequences from clean/original text samples.
        adv_curves: List of sensitivity sequences from adversarial text samples.
        
    Returns:
        Dictionary containing comparative analysis results:
            - feature_names: List of feature names (matches BASE_FEATURE_NAMES)
            - orig_means/adv_means: Mean feature values for each group
            - orig_stds/adv_stds: Standard deviations for each group  
            - feature_ratios: Ratio of adversarial to original means
            - feature_importance: Normalized difference scores
            - most_important_features: Top 3 most discriminative features
            - importance_values: Importance scores for top features
            
    Note:
        This analysis is useful for understanding which sensitivity characteristics
        are most affected by adversarial perturbations and can guide feature
        selection for detection models.
    """
    # Extract statistical features for both groups
    orig_features = extract_sensitivity_features(orig_curves)
    adv_features = extract_sensitivity_features(adv_curves)
    feature_names = BASE_FEATURE_NAMES.copy()

    # Compute descriptive statistics for each group
    orig_means = orig_features.mean(axis=0)
    adv_means = adv_features.mean(axis=0)
    orig_stds = orig_features.std(axis=0)
    adv_stds = adv_features.std(axis=0)
    
    # Calculate feature ratios (adversarial/original)
    ratios = adv_means / np.where(orig_means == 0, EPS, orig_means)
    
    # Compute feature importance using pooled standard error
    # Higher values indicate features that differ most between groups
    importance_scores = np.abs(adv_means - orig_means) / np.sqrt(orig_stds**2 + adv_stds**2 + EPS)
    
    # Identify most discriminative features
    top_indices = importance_scores.argsort()[::-1][:3]

    return {
        "feature_names": feature_names,
        "orig_means": orig_means.tolist(),
        "adv_means": adv_means.tolist(), 
        "orig_stds": orig_stds.tolist(),
        "adv_stds": adv_stds.tolist(),
        "feature_ratios": ratios.tolist(),
        "feature_importance": importance_scores.tolist(),
        "most_important_features": [feature_names[i] for i in top_indices],
        "importance_values": importance_scores[top_indices].tolist(),
    }


def parse_attack_logs(path: str, *, keep_only_successful: bool = True) -> pd.DataFrame:
    """Load and parse adversarial attack log files.
    
    Reads attack log CSV files and optionally filters to only successful attacks.
    These logs typically contain information about attack parameters, success rates,
    and generated adversarial examples.
    
    Args:
        path: Path to the CSV file containing attack logs.
        keep_only_successful: Whether to filter results to only successful attacks.
            If True, filters rows where 'result_type' column equals 'Successful'.
            
    Returns:
        DataFrame containing attack log data. Columns depend on the specific
        attack framework used but typically include attack parameters, 
        original/adversarial text, and success indicators.
        
    Note:
        The function logs the number of records before and after filtering
        for monitoring purposes.
    """
    logger.info(f"Loading attack logs from {path}")
    df = pd.read_csv(path)
    
    # Filter to successful attacks if requested and column exists
    if keep_only_successful and "result_type" in df.columns:
        original_count = len(df)
        df = df[df["result_type"] == "Successful"]
        logger.info(
            f"Filtered from {original_count} to {len(df)} successful attacks "
            f"({len(df)/original_count*100:.1f}% success rate)"
        )
    
    return df


def extract_perturbed_word_positions(
    original_text: str,
) -> List[int]:
    """Extract positions of adversarially perturbed words from marked text.
    
    Parses text where adversarially modified words are marked with double brackets
    [[word]] and returns their positions in the clean word sequence. This is useful
    for analyzing which word positions were targeted by adversarial attacks.
    
    Args:
        original_text: Text string where perturbed words are marked as [[word]].
            The brackets are used to indicate which words were modified during
            the adversarial attack process.
            
    Returns:
        List of zero-based word indices corresponding to perturbed positions.
        Empty list if no perturbed words are found.
        
    Example:
        >>> text = "The [[quick]] brown [[fox]] jumps"
        >>> extract_perturbed_word_positions(text)
        [1, 3]  # "quick" at position 1, "fox" at position 3
    """
    # Remove bracket markers to get clean text
    clean_text = original_text.replace("[[", "").replace("]]", "")
    clean_words = split_into_words(clean_text)
    
    # Extract words that were marked as perturbed
    perturbed_words = re.findall(r"\[\[(.*?)\]\]", original_text)

    # Find positions of perturbed words in the clean word sequence
    positions = []
    target_idx = 0
    
    for word_idx, word in enumerate(clean_words):
        # Check if this word matches the next expected perturbed word
        if target_idx < len(perturbed_words) and word == perturbed_words[target_idx]:
            positions.append(word_idx)
            target_idx += 1
            
    return positions

def extract_sequential_features(
    sensitivity_maps: List[List[float]],
    importance_maps: List[List[float]],
    *,
    top_n: int | None = None,
    zero_tol: float = 1e-10,
    filter_importance_scores: bool = True,
) -> List[np.ndarray]:
    """Extract paired sensitivity and importance sequences for neural network training.
    
    Combines gradient-based sensitivity scores with importance values into 2-channel
    sequential features suitable for LSTM-based adversarial detection models. 
    Optionally filters to keep only the most salient positions.
    
    Args:
        sensitivity_maps: List of gradient attribution sequences, one per text sample.
        importance_maps: List of importance score sequences, one per text sample.
            Must have same length and sequence structure as sensitivity_maps.
        top_n: Maximum number of highest-magnitude sensitivity values to retain
            per sequence. If None, keeps all values above zero_tol.
        zero_tol: Absolute threshold below which sensitivity values are treated as zero.
        filter_importance_scores: Whether to zero out importance scores at positions
            where sensitivity scores are zeroed by top_n filtering.
            
    Returns:
        List of 2D arrays, one per input sample. Each array has shape 
        [sequence_length, 2] where:
            - Channel 0: Filtered sensitivity values
            - Channel 1: Corresponding importance scores
            
    Raises:
        ValueError: If input lists have mismatched lengths or if corresponding
            sequences within the lists have different lengths.
            
    Note:
        This function is designed to prepare data for BiLSTM classifiers that
        process sequential sensitivity patterns for adversarial detection.
    """
    # Validate input consistency
    if len(sensitivity_maps) != len(importance_maps):
        raise ValueError(
            f"Input lists must have same length: got {len(sensitivity_maps)} "
            f"sensitivity maps and {len(importance_maps)} importance maps"
        )

    processed_features = []

    for sample_idx, (s_map, i_map) in enumerate(zip(sensitivity_maps, importance_maps)):
        if len(s_map) != len(i_map):
            raise ValueError(
                f"Sample {sample_idx} has mismatched sequence lengths: "
                f"sensitivity={len(s_map)}, importance={len(i_map)}"
            )

        seq_len = len(s_map)
        # Initialize 2-channel output: [seq_len, 2]
        output_channels = np.zeros((seq_len, 2), dtype=np.float32)

        if seq_len == 0:
            processed_features.append(output_channels)
            continue

        # Convert to numpy arrays for efficient processing
        sensitivity_array = np.array(s_map, dtype=np.float32)
        importance_array = np.array(i_map, dtype=np.float32)

        # Initialize with original values
        output_channels[:, 0] = sensitivity_array  # Channel 0: sensitivity
        output_channels[:, 1] = importance_array   # Channel 1: importance

        # Apply top_n filtering if specified
        if top_n is not None and top_n > 0:
            abs_sensitivity = np.abs(sensitivity_array)
            significant_indices = np.where(abs_sensitivity > zero_tol)[0]

            if len(significant_indices) > top_n:
                # Use argpartition for efficient top-k selection
                partition_indices = np.argpartition(
                    abs_sensitivity[significant_indices], -top_n
                )[-top_n:]
                top_positions = set(significant_indices[partition_indices])

                # Create mask for positions to keep
                keep_mask = np.array([idx in top_positions for idx in range(seq_len)])

                # Zero out non-selected sensitivity values
                output_channels[~keep_mask, 0] = 0.0
                
                # Optionally zero out corresponding importance scores
                if filter_importance_scores:
                    output_channels[~keep_mask, 1] = 0.0

            else:
                # Fewer significant values than top_n, just filter by tolerance
                zero_mask = abs_sensitivity <= zero_tol
                output_channels[zero_mask, 0] = 0.0
                
                if filter_importance_scores:
                    output_channels[zero_mask, 1] = 0.0

        elif top_n is None:
            # No top_n filtering, only apply zero tolerance
            zero_mask = np.abs(sensitivity_array) <= zero_tol
            output_channels[zero_mask, 0] = 0.0
            
            if filter_importance_scores:
                output_channels[zero_mask, 1] = 0.0

        processed_features.append(output_channels)
    return processed_features
