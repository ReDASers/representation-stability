"""Perturbation Overlap Analysis and Ranking Metrics.

This module analyzes the overlap between ground-truth perturbed word positions
(from adversarial attacks) and words selected by importance-based strategies.
It computes ranking metrics (MRR, MAP, NDCG) and overlap statistics across
different perturbation count bins.

Key Components:
    - Overlap computation for calibration and test phases
    - Ranking metrics: Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), 
      Normalized Discounted Cumulative Gain (NDCG)
    - Binned analysis by perturbation count
    - NDCG optimization for strategy parameter tuning
    - CSV reporting and visualization

Typical Usage:
    Used to evaluate how well word selection strategies (attention, saliency, etc.)
    identify the same words that adversarial attacks actually modify.
"""

import os
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..core.pipeline import SensitivityResults

def _compute_overlap_metrics_for_phase(
    perturbed_positions: List[List[int]],
    selected_indices: List[List[int]],
    word_ranks: List[List[Tuple[int, float]]],
    top_n: int,
    bins: Dict[str, Tuple[int, float]],
    phase_prefix: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute overlap and ranking metrics for a single experimental phase.
    
    Analyzes how well selected word indices overlap with ground-truth perturbed
    positions, computing both overlap statistics and ranking quality metrics.
    Results are binned by perturbation count for detailed analysis.
    
    Args:
        perturbed_positions: Ground-truth word indices that were actually perturbed
            during adversarial attacks, one list per text sample
        selected_indices: Word indices selected by the importance strategy,
            one list per text sample (should match perturbed_positions length)
        word_ranks: Full importance rankings as (word_index, score) tuples,
            sorted by decreasing importance score, one list per text sample
        top_n: Number of top words selected by strategy (0 means "all" words)
        bins: Perturbation count bins for analysis, mapping bin names to
            (min_count, max_count) tuples
        phase_prefix: Prefix for metric keys ("cal" for calibration, "test" for test)

    Returns:
        Tuple containing:
            - phase_metrics: Dictionary of computed metrics (MRR, MAP, NDCG, recalls, etc.)
            - phase_plot_data: Raw data arrays for visualization (ranks, recalls by bin)
            
    Note:
        If word_ranks is None or has mismatched length, ranking metrics will be
        set to NaN but overlap metrics will still be computed.
    """
    num_samples = len(perturbed_positions)
    phase_metrics = {}
    phase_plot_data = {}
    rank_data_available = False

    # Early validation of input data
    if num_samples == 0:
        print(f"No samples found for {phase_prefix} phase.")
        return phase_metrics, phase_plot_data
        
    if num_samples != len(selected_indices):
        print(f"{phase_prefix.capitalize()} data length mismatch: "
              f"perturbed_positions ({num_samples}) vs selected_indices ({len(selected_indices)}). "
              f"Cannot compute overlap.")
        return phase_metrics, phase_plot_data

    # Check availability of ranking data for advanced metrics
    if word_ranks is None or num_samples != len(word_ranks):
        print(f"{phase_prefix.capitalize()} word rank data unavailable or mismatched length "
              f"({len(word_ranks) if word_ranks else 'None'} vs {num_samples}). "
              f"Skipping ranking metrics.")
        rank_data_available = False
    else:
        rank_data_available = True
        print(f"Rank data available for {phase_prefix.capitalize()} phase.")

    # Initialize metric accumulators
    total_overlap = 0
    total_perturbed = 0
    num_perturbations_per_sample = []
    first_hit_ranks = []  # Rank of first relevant item found
    reciprocal_ranks = []  # 1/rank for MRR calculation
    average_precisions = []  # Average precision per sample for MAP
    ndcg_values = []  # NDCG scores per sample
    
    # Define evaluation points for recall@k and NDCG@k metrics
    recall_k_values = [1, 3, 5, 10, 15, 20, 25, 30, 40, 50]
    ndcg_k_values = recall_k_values
    ndcg_at_k_values = {k: [] for k in ndcg_k_values}

    # Initialize bin-wise metric accumulators
    bin_metrics_accumulators = {
        key: {"sample_recalls": [], "sample_norm_recalls": [], "count": 0} 
        for key in bins
    }

    # Process each sample to compute overlap and ranking metrics
    for i in range(num_samples):
        ground_truth_set = set(perturbed_positions[i])
        selected_set = set(selected_indices[i])

        num_perturbed = len(ground_truth_set)
        num_selected = len(selected_set)
        overlap_count = len(ground_truth_set & selected_set)  # Intersection size

        # Accumulate global statistics
        total_overlap += overlap_count
        total_perturbed += num_perturbed

        # Calculate sample-level overlap rate (recall)
        sample_overlap_rate = overlap_count / num_perturbed if num_perturbed > 0 else 1.0
        
        # Calculate normalized recall (accounts for selection strategy constraints)
        # For top_n strategies, max possible overlap is limited by selection size
        if top_n > 0:
            min_relevant = min(num_selected, num_perturbed)
        else:
            # For "all" strategy, use selected count if available, otherwise perturbed count
            min_relevant = min(num_selected, num_perturbed) if num_selected > 0 else num_perturbed
        
        sample_normalized_recall = overlap_count / min_relevant if min_relevant > 0 else 1.0
        num_perturbations_per_sample.append(num_perturbed)

        # Initialize ranking metrics for this sample
        current_first_hit_rank = float('inf')
        if rank_data_available:
            # Create mapping from word index to rank position (1-indexed)
            rank_map = {idx: rank + 1 for rank, (idx, score) in enumerate(word_ranks[i])}
            # Find ranks of all perturbed words that appear in the ranking
            perturbed_ranks = [rank_map[pidx] for pidx in ground_truth_set if pidx in rank_map]
            if perturbed_ranks:
                # First hit rank is the highest-ranked (lowest rank number) perturbed word
                current_first_hit_rank = min(perturbed_ranks)
                
                # Calculate Average Precision (AP) for this sample
                # AP = (1/|relevant|) * Σ(precision@rank_i) for each relevant item i
                sorted_ranks = sorted(perturbed_ranks)
                precisions = []
                for j, rank in enumerate(sorted_ranks):
                    if rank > 0:
                        # Precision at this rank = (# relevant items up to rank) / rank
                        precisions.append((j + 1) / rank)
                    else:
                        print(f"WARNING: Invalid rank 0 encountered in sample {i}")

                # Compute average precision for this sample
                if ground_truth_set and precisions:
                    average_precision = sum(precisions) / len(ground_truth_set)
                    average_precisions.append(average_precision)
                elif not ground_truth_set:
                    # No relevant items: AP = 0
                    average_precisions.append(0.0)
                else:
                    # Relevant items but no valid precisions: AP = 0
                    average_precisions.append(0.0)

                # Calculate Normalized Discounted Cumulative Gain (NDCG)
                # NDCG measures ranking quality with position-based discounting
                all_ranks = [(idx, score, idx in ground_truth_set) for idx, score in word_ranks[i]]
                all_ranks.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
                
                # Calculate DCG (Discounted Cumulative Gain)
                dcg = 0
                for rank, (_, _, is_relevant) in enumerate(all_ranks, 1):
                    if is_relevant:
                        # Standard NDCG formula: relevance / log2(rank + 1)
                        dcg += 1.0 / np.log2(rank + 1)
                
                # Calculate IDCG (Ideal DCG) - maximum possible DCG
                idcg = 0
                num_relevant_items = len(ground_truth_set)
                if num_relevant_items > 0:
                    # Ideal case: all relevant items ranked at top positions
                    for rank in range(1, num_relevant_items + 1):
                        idcg += 1.0 / np.log2(rank + 1)
                
                # Normalize DCG by IDCG
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_values.append(ndcg)
                
                # Calculate NDCG@k for different cutoff values
                for k in ndcg_k_values:
                    # DCG considering only top-k ranked items
                    dcg_k = 0
                    for rank, (_, _, is_relevant) in enumerate(all_ranks[:k], 1):
                        if is_relevant:
                            dcg_k += 1.0 / np.log2(rank + 1)
                    
                    # IDCG for top-k: ideal case with relevant items in top positions
                    idcg_k = 0
                    num_relevant_in_top_k_ideal = min(num_relevant_items, k)
                    if num_relevant_in_top_k_ideal > 0:
                        for rank in range(1, num_relevant_in_top_k_ideal + 1):
                            idcg_k += 1.0 / np.log2(rank + 1)
                    
                    # Compute NDCG@k
                    ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0
                    ndcg_at_k_values[k].append(ndcg_k)

        # Store ranking metrics for this sample
        first_hit_ranks.append(current_first_hit_rank)
        reciprocal_ranks.append(1.0 / current_first_hit_rank if current_first_hit_rank != float('inf') else 0.0)

        # Assign sample to perturbation count bin for stratified analysis
        assigned_bin = None
        for bin_key, (lower, upper) in bins.items():
            if lower <= num_perturbed <= upper:
                assigned_bin = bin_key
                break
        
        # Accumulate bin-specific metrics
        if assigned_bin:
            bin_metrics_accumulators[assigned_bin]["sample_recalls"].append(sample_overlap_rate)
            bin_metrics_accumulators[assigned_bin]["sample_norm_recalls"].append(sample_normalized_recall)
            bin_metrics_accumulators[assigned_bin]["count"] += 1

    # Compute aggregated ranking metrics
    if rank_data_available and reciprocal_ranks:
        # Mean Reciprocal Rank (MRR): average of 1/rank for first relevant item
        phase_metrics[f'{phase_prefix}_mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else np.nan
        
        # Recall@k: fraction of samples with first hit within top k positions
        for k in recall_k_values:
            hits_at_k = sum(1 for r in first_hit_ranks if r <= k and r != float('inf'))
            phase_metrics[f'{phase_prefix}_recall_at_{k}'] = hits_at_k / num_samples if num_samples > 0 else np.nan
            
        # Mean Average Precision (MAP): average of sample-level APs
        phase_metrics[f'{phase_prefix}_map'] = np.mean(average_precisions) if average_precisions else np.nan
        
        # Mean NDCG: average of sample-level NDCG scores
        phase_metrics[f'{phase_prefix}_ndcg'] = np.mean(ndcg_values) if ndcg_values else np.nan
            
        # NDCG@k: average NDCG computed at different cutoff points
        for k in ndcg_k_values:
            phase_metrics[f'{phase_prefix}_ndcg_at_{k}'] = np.mean(ndcg_at_k_values[k]) if ndcg_at_k_values.get(k) else np.nan
            
        # Store valid ranks for visualization (exclude samples with no hits)
        phase_plot_data[f'{phase_prefix}_first_hit_ranks'] = [r for r in first_hit_ranks if r != float('inf')]
    else:
        # Set all ranking metrics to NaN when rank data is unavailable
        phase_metrics[f'{phase_prefix}_mrr'] = np.nan
        for k in recall_k_values:
            phase_metrics[f'{phase_prefix}_recall_at_{k}'] = np.nan
        phase_metrics[f'{phase_prefix}_map'] = np.nan
        phase_metrics[f'{phase_prefix}_ndcg'] = np.nan
        for k in ndcg_k_values:
            phase_metrics[f'{phase_prefix}_ndcg_at_{k}'] = np.nan
        phase_plot_data[f'{phase_prefix}_first_hit_ranks'] = []

    # Compute bin-stratified overlap metrics
    for bin_key, bin_data in bin_metrics_accumulators.items():
        # Store sample count for this bin
        phase_metrics[f'{phase_prefix}_bin_{bin_key}_sample_count'] = bin_data["count"]
        
        # Calculate mean recall metrics (convert to percentages)
        if bin_data["count"] > 0:
            mean_recall = np.mean(bin_data["sample_recalls"]) * 100  # Standard recall
            mean_norm_recall = np.mean(bin_data["sample_norm_recalls"]) * 100  # Normalized recall
        else:
            mean_recall = np.nan
            mean_norm_recall = np.nan
            
        phase_metrics[f'{phase_prefix}_bin_{bin_key}_mean_recall_perc'] = mean_recall
        phase_metrics[f'{phase_prefix}_bin_{bin_key}_mean_norm_recall_perc'] = mean_norm_recall
        
        # Store raw sample-level data for visualization
        phase_plot_data[f'{phase_prefix}_sample_recalls_bin_{bin_key}'] = bin_data["sample_recalls"]
        phase_plot_data[f'{phase_prefix}_sample_norm_recalls_bin_{bin_key}'] = bin_data["sample_norm_recalls"]

    print(f"  {phase_prefix.capitalize()} phase metrics computed: MRR, MAP, NDCG, Recall@k, and binned statistics.")
    return phase_metrics, phase_plot_data


def _save_overlap_report(
    all_metrics: Dict[str, Any], 
    output_dir: str, 
    word_selection_strategy: str, 
    top_n: int, 
    delimiter: str = ','
):
    """Save perturbation overlap analysis results to CSV file.
    
    Exports computed metrics to a structured CSV file with predefined column
    ordering for consistent reporting across different experiments.
    
    Args:
        all_metrics: Dictionary containing all computed metrics (MRR, MAP, NDCG, 
            recall@k, binned statistics, etc.)
        output_dir: Directory path where the CSV file will be saved
        word_selection_strategy: Strategy name used for word selection 
            (e.g., 'attention', 'saliency', 'gradient_attention', 'all')
        top_n: Number of top words selected by the strategy (0 indicates 'all' words)
        delimiter: CSV delimiter character (default: ',')
        
    Returns:
        bool: True if file was saved successfully, False otherwise
        
    Note:
        Creates the output directory if it doesn't exist. Missing metrics
        are filled with NaN values to maintain consistent CSV structure.
    """
    # Validate input data
    if not all_metrics:
        print("WARNING: No metrics to save. Skipping overlap report.")
        return False

    try:
        # Construct output file path
        filename = f"perturbation_overlap_{word_selection_strategy}.csv"
        output_path = os.path.join(output_dir, filename)

        # Define standardized column ordering for consistent reporting
        preferred_column_order = [
            # Experiment metadata
            'selection_strategy', 'top_n',
            
            # Primary ranking metrics - Calibration phase
            'cal_mrr', 'cal_map', 'cal_ndcg',
            
            # Recall@K - Calibration
            'cal_recall_at_1', 'cal_recall_at_3', 'cal_recall_at_5', 'cal_recall_at_10', 
            'cal_recall_at_15', 'cal_recall_at_20', 'cal_recall_at_25', 'cal_recall_at_30',
            'cal_recall_at_40', 'cal_recall_at_50',
            
            # NDCG@K - Calibration
            'cal_ndcg_at_1', 'cal_ndcg_at_3', 'cal_ndcg_at_5', 'cal_ndcg_at_10', 'cal_ndcg_at_15', 'cal_ndcg_at_20', 'cal_ndcg_at_25', 'cal_ndcg_at_30',
            'cal_ndcg_at_40', 'cal_ndcg_at_50',
            
            # Bin Counts - Calibration
            'cal_bin_1_sample_count', 'cal_bin_2-3_sample_count', 'cal_bin_4-6_sample_count',
            'cal_bin_7-10_sample_count', 'cal_bin_11-16_sample_count', 'cal_bin_16-25_sample_count',
            'cal_bin_>25_sample_count',
            
            # Mean Binned Metrics - Calibration
            'cal_bin_1_mean_recall_perc', 'cal_bin_1_mean_norm_recall_perc',
            'cal_bin_2-3_mean_recall_perc', 'cal_bin_2-3_mean_norm_recall_perc',
            'cal_bin_4-6_mean_recall_perc', 'cal_bin_4-6_mean_norm_recall_perc',
            'cal_bin_7-10_mean_recall_perc', 'cal_bin_7-10_mean_norm_recall_perc',
            'cal_bin_11-16_mean_recall_perc', 'cal_bin_11-16_mean_norm_recall_perc',
            'cal_bin_16-25_mean_recall_perc', 'cal_bin_16-25_mean_norm_recall_perc',
            'cal_bin_>25_mean_recall_perc', 'cal_bin_>25_mean_norm_recall_perc',
            
            # Overall Metrics - Test
            'test_mrr', 'test_map', 'test_ndcg',
            
            # Recall@K - Test
            'test_recall_at_1', 'test_recall_at_3', 'test_recall_at_5', 'test_recall_at_10', 
            'test_recall_at_15', 'test_recall_at_20', 'test_recall_at_25', 'test_recall_at_30',
            'test_recall_at_40', 'test_recall_at_50',
            
            # NDCG@K - Test
            'test_ndcg_at_1', 'test_ndcg_at_3', 'test_ndcg_at_5', 'test_ndcg_at_10', 'test_ndcg_at_15', 'test_ndcg_at_20', 'test_ndcg_at_25', 'test_ndcg_at_30',
            'test_ndcg_at_40', 'test_ndcg_at_50',
            
            # Bin Counts - Test
            'test_bin_1_sample_count', 'test_bin_2-3_sample_count', 'test_bin_4-6_sample_count',
            'test_bin_7-10_sample_count', 'test_bin_11-16_sample_count', 'test_bin_16-25_sample_count',
            'test_bin_>25_sample_count',
            
            # Mean Binned Metrics - Test
            'test_bin_1_mean_recall_perc', 'test_bin_1_mean_norm_recall_perc',
            'test_bin_2-3_mean_recall_perc', 'test_bin_2-3_mean_norm_recall_perc',
            'test_bin_4-6_mean_recall_perc', 'test_bin_4-6_mean_norm_recall_perc',
            'test_bin_7-10_mean_recall_perc', 'test_bin_7-10_mean_norm_recall_perc',
            'test_bin_11-16_mean_recall_perc', 'test_bin_11-16_mean_norm_recall_perc',
            'test_bin_16-25_mean_recall_perc', 'test_bin_16-25_mean_norm_recall_perc',
            'test_bin_>25_mean_recall_perc', 'test_bin_>25_mean_norm_recall_perc',
        ]

        # Add experiment metadata to metrics
        all_metrics['selection_strategy'] = word_selection_strategy
        all_metrics['top_n'] = top_n

        # Ensure all expected columns exist (fill missing with NaN)
        for col in preferred_column_order:
            if col not in all_metrics:
                all_metrics[col] = np.nan

        # Create single-row DataFrame
        df = pd.DataFrame([all_metrics])

        # Reorder columns: preferred order first, then any additional columns
        existing_cols = set(df.columns)
        ordered_cols = [col for col in preferred_column_order if col in existing_cols]
        remaining_cols = sorted(list(existing_cols - set(ordered_cols)))
        df = df[ordered_cols + remaining_cols]

        # Save to CSV file
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False, sep=delimiter)
        print(f"Overlap report saved to: {output_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save overlap report: {e}")
        return False

def calculate_ndcg_at_k_values(
    perturbed_positions: List[List[int]],
    word_ranks: List[List[Tuple[int, float]]],
    k_values: List[int]
) -> Dict[int, float]:
    """Calculate NDCG@k metrics efficiently across multiple cutoff values.
    
    Computes Normalized Discounted Cumulative Gain at various cutoff points
    (k values) to evaluate ranking quality. More efficient than computing
    NDCG@k separately for each k value.
    
    Args:
        perturbed_positions: Ground-truth perturbed word indices for each sample
        word_ranks: Importance rankings as (word_index, score) tuples for each sample,
            sorted by decreasing importance score
        k_values: List of cutoff values (k) to evaluate NDCG at
        
    Returns:
        Dictionary mapping each k value to its average NDCG@k score across all samples
        
    Note:
        Returns zero values for all k if input data is missing or invalid.
        NDCG@k considers only the top-k ranked items for evaluation.
    """
    # Validate input data
    if not perturbed_positions or not word_ranks or not k_values:
        print("WARNING: Missing data for NDCG calculation. Returning zero values.")
        return {k: 0.0 for k in k_values}
    
    max_k = max(k_values)
    ndcg_values = {k: [] for k in k_values}  # Accumulate NDCG scores per k
    
    # Process each sample to compute NDCG@k values
    for i, (ground_truth, ranks) in enumerate(zip(perturbed_positions, word_ranks)):
        if not ground_truth or i >= len(word_ranks):
            continue
            
        ground_truth_set = set(ground_truth)
        
        # Create ranked list with relevance labels, sorted by importance score
        all_ranks = [(idx, score, idx in ground_truth_set) for idx, score in ranks]
        all_ranks.sort(key=lambda x: x[1], reverse=True)  # Highest score first
        
        # Precompute Ideal DCG (IDCG) values for all k cutoffs
        idcg_values = {}
        for k in k_values:
            idcg_k = 0
            # Ideal case: all relevant items in top-k positions
            for rank in range(1, min(len(ground_truth_set), k) + 1):
                idcg_k += 1.0 / np.log2(rank + 1)
            idcg_values[k] = idcg_k
            
        # Calculate DCG@k for each cutoff value
        for k in k_values:
            if not ground_truth_set or idcg_values[k] == 0:
                continue
                
            # Compute DCG considering only top-k items
            dcg_k = 0
            for rank, (_, _, is_relevant) in enumerate(all_ranks[:k], 1):
                if is_relevant:
                    dcg_k += 1.0 / np.log2(rank + 1)
            
            # Normalize by ideal DCG to get NDCG@k
            ndcg_k = dcg_k / idcg_values[k] if idcg_values[k] > 0 else 0
            ndcg_values[k].append(ndcg_k)
    
    # Compute average NDCG@k across all samples
    avg_ndcg = {}
    for k in k_values:
        if ndcg_values[k]:
            avg_ndcg[k] = np.mean(ndcg_values[k])
        else:
            avg_ndcg[k] = 0.0  # No valid samples for this k
            
    return avg_ndcg

def save_ndcg_optimization_results(
    output_dir: str,
    word_selection_strategy: str,
    ndcg_values: Dict[int, float],
    optimal_top_n: int,
    dpi: int = 300
):
    """Save NDCG optimization results with visualization.
    
    Exports NDCG@k values to CSV and creates a visualization showing
    NDCG performance across different k values, highlighting the optimal
    cutoff point and percentage improvements.
    
    Args:
        output_dir: Directory path to save results files
        word_selection_strategy: Name of the word selection strategy being optimized
        ndcg_values: Dictionary mapping k values to their corresponding NDCG scores
        optimal_top_n: Optimal k value determined by optimization process
        dpi: Resolution for saved visualization (default: 300)
        
    Returns:
        Tuple containing paths to the saved CSV and visualization files
        
    Note:
        Creates the output directory if it doesn't exist. If visualization
        generation fails, only the CSV file will be saved.
    """
    
    # Convert NDCG results to DataFrame
    data = {
        'k': list(ndcg_values.keys()), 
        'ndcg': list(ndcg_values.values())
    }
    df = pd.DataFrame(data).sort_values('k')
    
    # Save NDCG data to CSV
    csv_path = os.path.join(output_dir, f"ndcg_optimization_{word_selection_strategy}.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    # Create NDCG optimization visualization
    fig_path = os.path.join(output_dir, f"ndcg_optimization_{word_selection_strategy}.png")
    
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot main NDCG curve
        plt.plot(df['k'], df['ndcg'], 'b-', marker='o', linewidth=2, label='NDCG@k')
        
        # Mark the optimal k value
        plt.axvline(x=optimal_top_n, color='r', linestyle='--', 
                   label=f'Optimal k={optimal_top_n}')
        
        # Add secondary axis for percentage improvement analysis
        if len(df) > 1:
            ax2 = plt.gca().twinx()
            # Calculate percentage change between consecutive k values
            improvement = df['ndcg'].pct_change() * 100
            improvement[0] = 0  # First point has no baseline for improvement
            ax2.plot(df['k'], improvement, 'g-', alpha=0.5, label='% Improvement')
            ax2.set_ylabel('% Improvement', color='g')
            
            # Mark 1% improvement threshold for optimization stopping criteria
            ax2.axhline(y=1.0, color='g', linestyle=':', alpha=0.5, label='1% Threshold')
            
            plt.annotate('1% Improvement Threshold', 
                        xy=(df['k'].iloc[-1], 1.0),
                        xytext=(5, 5),
                        textcoords='offset points',
                        color='g', alpha=0.7)
        
        # Configure plot appearance
        plt.xlabel('k value')
        plt.ylabel('NDCG@k')
        plt.title(f'NDCG Optimization: {word_selection_strategy.capitalize()} Strategy')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Ensure integer tick marks on x-axis
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"NDCG optimization results saved:")
        print(f"  Data: {csv_path}")
        print(f"  Visualization: {fig_path}")
    except Exception as e:
        print(f"ERROR: Failed to generate NDCG optimization visualization: {e}")
        plt.close()
    
    return csv_path, fig_path


def _calculate_and_save_perturbation_overlap(
    cal_perturbed_positions: List[List[int]],
    test_perturbed_positions: List[List[int]],
    cal_adv_sensitivity: SensitivityResults,
    test_adv_sensitivity: SensitivityResults,
    output_dir: str,
    word_selection_strategy: str,
    top_n: int = 0
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """Calculate and save comprehensive perturbation overlap analysis.
    
    Orchestrates the complete overlap analysis pipeline: computes overlap metrics
    for both calibration and test phases, performs binned analysis by perturbation
    count, and saves results to CSV.
    
    Args:
        cal_perturbed_positions: Ground-truth perturbed word indices for calibration texts
        test_perturbed_positions: Ground-truth perturbed word indices for test texts
        cal_adv_sensitivity: Sensitivity analysis results for calibration adversarial texts
            (contains selected indices and word rankings)
        test_adv_sensitivity: Sensitivity analysis results for test adversarial texts
            (contains selected indices and word rankings)
        output_dir: Directory path to save analysis results
        word_selection_strategy: Strategy name for word importance ranking
            (e.g., 'attention', 'saliency', 'gradient_attention', 'all')
        top_n: Number of top-ranked words selected (0 means all words)

    Returns:
        Tuple containing:
            - all_metrics: Dictionary with computed overlap and ranking metrics
            - all_plot_data: Dictionary with raw data arrays for visualization
            - top_n: The top_n value used (returned for consistency)
            
    Note:
        Uses fixed perturbation count bins: 1, 2-3, 4-6, 7-10, 11-16, 16-25, >25.
        Saves results to a CSV file named 'perturbation_overlap_{strategy}.csv'.
    """
    print(f"Perturbation overlap analysis:")
    print(f"  Strategy: {word_selection_strategy}")
    print(f"  Top-N: {top_n if top_n > 0 else 'all'}")

    # Define standardized perturbation count bins for stratified analysis
    bins = {
        "1": (1, 1),           # Single perturbation
        "2-3": (2, 3),         # Few perturbations
        "4-6": (4, 6),         # Moderate perturbations
        "7-10": (7, 10),       # Many perturbations
        "11-16": (11, 16),     # Very many perturbations
        "16-25": (16, 25),     # Extensive perturbations
        ">25": (26, float('inf'))  # Massive perturbations
    }

    all_metrics = {}
    all_plot_data = {}

    # Compute overlap metrics for calibration phase
    cal_metrics, cal_plot_data = _compute_overlap_metrics_for_phase(
        perturbed_positions=cal_perturbed_positions,
        selected_indices=cal_adv_sensitivity.selected_indices,
        word_ranks=cal_adv_sensitivity.word_ranks,
        top_n=top_n,
        bins=bins,
        phase_prefix="cal"
    )
    all_metrics.update(cal_metrics)
    all_plot_data.update(cal_plot_data)

    # Compute overlap metrics for test phase
    test_metrics, test_plot_data = _compute_overlap_metrics_for_phase(
        perturbed_positions=test_perturbed_positions,
        selected_indices=test_adv_sensitivity.selected_indices,
        word_ranks=test_adv_sensitivity.word_ranks,
        top_n=top_n,
        bins=bins,
        phase_prefix="test"
    )
    all_metrics.update(test_metrics)
    all_plot_data.update(test_plot_data)

    # Export comprehensive results to CSV
    _save_overlap_report(
        all_metrics=all_metrics,
        output_dir=output_dir,
        word_selection_strategy=word_selection_strategy,
        top_n=top_n
    )

    print(f"Overlap analysis complete for {word_selection_strategy} strategy.")
    return all_metrics, all_plot_data, top_n