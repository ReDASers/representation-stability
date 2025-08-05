# TextShield: Adaptive Word Importance (AWI) Based Adversarial Text Detection

This directory contains our implementation of TextShield, an adversarial text detection method based on Adaptive Word Importance (AWI), from the paper "TextShield: Beyond Successfully Detecting Adversarial Sentences in Text Classification" (ICLR 2023).

## Overview

TextShield detects adversarial examples by analyzing the importance of individual words in text classification decisions. The method computes word-level importance scores using various attribution techniques and uses these features to train a detector that can distinguish between original and adversarial texts.

### Key Features

- **Multiple AWI computation methods** (Vanilla Gradient, Integrated Gradients, Guided Backpropagation, LRP)
- **Multi-channel feature representation** combining multiple attribution methods
- **Batch processing** with memory optimization and caching
- **BiLSTM-based detection architecture** (adapted due to insufficient implementation details in original paper)
- **Timing and performance metrics**

## Method

The TextShield detection process involves:

1. **Text Preprocessing**: Tokenize input texts using pre-trained model tokenizers
2. **AWI Computation**: Calculate word importance scores using gradient-based attribution methods
3. **Feature Extraction**: Convert importance scores to multi-channel sequential features
4. **Detector Training**: Train BiLSTM classifier on AWI features to distinguish adversarial examples
5. **Detection**: Use trained detector to classify new samples as original or adversarial

### Adaptive Word Importance (AWI) Methods

The implementation supports four AWI computation methods:

1. **Vanilla Gradient**: Basic gradient-based importance scores
2. **Integrated Gradients**: Path-integrated gradients from baseline to input
3. **Guided Backpropagation**: Modified backpropagation with positive gradients
4. **Layer-wise Relevance Propagation (LRP)**: Relevance scores propagated through network layers

### Multi-Channel Feature Format

Features are represented as multi-channel sequences where:
- Each sample has shape `[sequence_length, num_channels]`
- Each channel corresponds to a different AWI method
- Values represent word-level importance scores for the predicted class

## Files

- **`extract_awi_features.py`**: Main feature extraction script containing:
  - AWI computation functions for all supported methods
  - Batch processing with memory optimization
  - Multi-channel feature generation
  - Caching mechanisms for efficiency
- **`awi_utils.py`**: Core utilities for AWI computation
- **`train_evaluate_detector.py`**: Detector training and evaluation script
- **`run_textshield.bat`**: Batch script for running all experiments

## Usage

> **Note**: All commands should be run from the main project directory.

### Feature Extraction

First, extract AWI features from your adversarial datasets:

```bash
python textshield/extract_awi_features.py \
    --awi_method all \
    --base_model_dir models/ \
    --base_data_dir data/ \
    --timing_metrics_dir output/timing_metrics \
    --batch_size 32 \
    --max_token_length 128 \
    --ig_steps 5 \
    --ig_baseline_type zero
```

This will process all datasets, models, and attacks, generating AWI features for all supported methods.

### Detector Training and Evaluation

After feature extraction, train and evaluate detectors:

```bash
# Single experiment
python textshield/train_evaluate_detector.py \
    --features_dir data \
    --dataset imdb \
    --model roberta \
    --attack textfooler \
    --strategy awi \
    --output_dir output/detector_results

# All experiments
textshield/run_textshield.bat
```

## Configuration

### Feature Extraction Parameters

```bash
--awi_method              # AWI method: vanilla_gradient, integrated_gradients, 
                         # guided_backpropagation, lrp, or "all"
--base_model_dir         # Directory containing fine-tuned models
--base_data_dir          # Directory containing adversarial datasets
--timing_metrics_dir     # Directory to save timing performance metrics
--batch_size            # Batch size for processing (default: 32)
--max_token_length      # Maximum sequence length (default: 128)
--ig_steps              # Integration steps for Integrated Gradients (default: 5)
--ig_baseline_type      # Baseline type for IG: zero, pad, or random
```

### Detection Parameters

```bash
--features_dir          # Directory containing extracted AWI features
--dataset              # Dataset name (imdb, yelp, ag_news)
--model                # Model name (roberta, deberta)
--attack               # Attack name (textfooler, bert-attack, deepwordbug)
--strategy             # Feature type (awi)
--output_dir           # Output directory for results
```

## Input Data Format

The system expects CSV files with the following structure:

```
data/
├── {dataset}/
│   ├── {model}/
│   │   ├── {attack}/
│   │   │   ├── calibration_data.csv
│   │   │   ├── test_data.csv
│   │   │   └── awi/              # Generated AWI features
│   │   │       ├── cal_data.json
│   │   │       ├── test_data.json
│   │   │       ├── feature_info.json
│   │   │       └── text_references.json
```

### Required CSV Columns
- `original_text`: Original (benign) text samples
- `adversarial_text`: Corresponding adversarial examples

## Output Format

### AWI Features
Features are saved as JSON files containing:
- `features`: Multi-channel AWI feature arrays
- `labels`: Binary labels (0=original, 1=adversarial)
- `channel_methods`: Order of AWI methods in channels
- `metadata`: Configuration and sample information

### Detection Results
Results are saved as CSV files with:

#### Performance Metrics
- `accuracy`, `precision`, `recall`, `f1`: Classification performance
- `auc`: Area under ROC curve
- `avg_precision`: Average precision score

#### Sample Counts
- `orig_train_samples`, `adv_train_samples`: Training set sizes
- `orig_test_samples`, `adv_test_samples`: Test set sizes

#### Timing Information
- `training_time`, `evaluation_time`: Training and evaluation duration

#### Configuration
- `dataset`, `model`, `attack`, `strategy`: Experiment setup
- `detection_method`: Always "bilstm"

## Performance Considerations

### Computational Complexity
- **Integrated Gradients**: Computationally expensive, scales with `--ig_steps`
- **Increasing IG steps** becomes borderline prohibitive for large datasets
- **Batch processing** helps manage memory usage
- **Caching** reduces redundant tokenization overhead

### Memory Optimization
- Automatic CUDA cache clearing between batches
- Tokenization and word mapping caching
- Model unloading after processing each attack type

### Timing Metrics
The system automatically tracks and saves detailed timing information for performance analysis.

## Implementation Notes

### Differences from Original Paper
- **No official implementation** was available, so this is implemented based on the paper description
- **BiLSTM architecture** used for detection due to insufficient LSTM implementation details in the original paper
- **Multi-channel representation** added to leverage multiple AWI methods simultaneously

### Model Support
The detector works with fine-tuned models in the expected directory structure:
- `models/roberta_{dataset}/`
- `models/deberta_{dataset}/`

## Citation

If you use this implementation, please cite the original TextShield paper:

```bibtex
@inproceedings{
shen2023textshield,
title={TextShield: Beyond Successfully Detecting Adversarial Sentences in text classification},
author={Lingfeng Shen and Ze Zhang and Haiyun Jiang and Ying Chen},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=xIWfWvKM7aQ}
}
```

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce `--batch_size` or `--max_token_length`
2. **Slow Integrated Gradients**: Reduce `--ig_steps` (default: 5)
3. **Missing features**: Ensure AWI extraction completed successfully before training detectors
4. **Import errors**: Ensure you're running from the main project directory