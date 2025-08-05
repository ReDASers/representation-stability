# Sharpness-Based Adversarial Text Detection

This directory contains the implementation of sharpness-based adversarial text detection using the First-Order Stationary Condition (FOSC), based on the paper "Detecting Adversarial Samples through Sharpness of Loss Landscape" (ACL 2023).

## Overview

The sharpness-based detection method identifies adversarial examples by measuring how sharply the loss landscape changes around input samples. Adversarial examples typically lie in sharper regions of the loss landscape compared to natural examples.

### Key Features

- **FOSC-based sharpness computation** with adaptive step sizes
- **Support for multiple perturbation norms** (L2 and L∞)
- **Frank-Wolfe gap** as alternative convergence metric
- **Cross-domain generalization** experiments
- **Batch processing** for efficient evaluation
- **Comprehensive evaluation metrics** including AUC-ROC and Average Precision

## Method

The detection process involves:

1. **Clean Loss Computation**: Calculate loss on original inputs
2. **Adversarial Perturbation Generation**: Use gradient-based optimization to find perturbations that maximize loss
3. **Convergence Measurement**: Apply FOSC or Frank-Wolfe gap to guide optimization
4. **Adaptive Step Sizing**: Adjust perturbation step sizes based on convergence metrics
5. **Sharpness Score**: Use final perturbed loss or loss difference as detection signal

### First-Order Stationary Condition (FOSC)

The FOSC measures convergence using:
```
FOSC = |⟨∇f(x), x-x₀⟩ + ε||∇f(x)||²|
```

Where:
- `∇f(x)` is the gradient of loss at current point
- `x-x₀` is the perturbation vector
- `ε` is the perturbation radius

### Frank-Wolfe Gap (Alternative)

As an alternative convergence metric:
```
FW_gap = √ε||∇f(x)||_F - ⟨x-x₀, ∇f(x)⟩
```

## Files

- **`sharpness_detector.py`**: Main implementation containing:
  - `SharpnessDetector`: Core detection class
  - `AdaptiveAdvSize`: Adaptive step size controller
  - Evaluation and experiment utilities
- **`run_sharpness.bat`**: Script for running standard experiments across all dataset/model/attack combinations
- **`run_sharpness_generalization.bat`**: Script for cross-domain generalization experiments

## Usage

> **Note**: The batch scripts should be run from the main project directory. Individual Python commands can be run from either the main directory (using `sharpness/sharpness_detector.py`) or from the `sharpness/` directory (using `sharpness_detector.py`).

### Basic Usage

```python
from sharpness_detector import SharpnessDetector

# Initialize detector
detector = SharpnessDetector(
    model_name="textattack/roberta-base-imdb",
    device="cuda",
    threshold=0.25
)

# Train/calibrate detector
detector.fit(train_texts, train_labels, adversarial_labels)

# Detect adversarial examples
predictions, sharpness_scores = detector.predict(test_texts, test_labels)
```

### Command Line Interface

#### Standard Experiments

Run detection on a specific dataset/model/attack combination:

```bash
# From the main project directory
python sharpness/sharpness_detector.py \
    --data_dir data \
    --dataset imdb \
    --model roberta \
    --attack textfooler \
    --output_dir output/sharpness_results

# OR from the sharpness/ directory
cd sharpness
python sharpness_detector.py \
    --data_dir ../data \
    --dataset imdb \
    --model roberta \
    --attack textfooler \
    --output_dir output/sharpness_results
```

#### Generalization Experiments

**Cross-dataset generalization:**
```bash
# From the main project directory
python sharpness/sharpness_detector.py \
    --experiment_type cross_dataset \
    --train_datasets yelp \
    --train_models roberta \
    --train_attacks textfooler,deepwordbug,bert-attack \
    --test_datasets imdb \
    --test_models roberta \
    --test_attacks textfooler,deepwordbug,bert-attack \
    --data_dir data \
    --output_dir output/generalization_results
```

**Cross-attack generalization:**
```bash
# From the main project directory
python sharpness/sharpness_detector.py \
    --experiment_type cross_attack \
    --train_datasets yelp,imdb \
    --train_models roberta \
    --train_attacks textfooler \
    --test_attacks deepwordbug,bert-attack \
    --data_dir data \
    --output_dir output/generalization_results
```

**Cross-encoder generalization:**
```bash
# From the main project directory
python sharpness/sharpness_detector.py \
    --experiment_type cross_encoder \
    --train_datasets yelp,imdb \
    --train_models roberta \
    --train_attacks textfooler,deepwordbug,bert-attack \
    --test_models deberta \
    --data_dir data \
    --output_dir output/generalization_results
```

### Batch Scripts

**Run all standard experiments:**
```bash
# Windows - from the main project directory
sharpness/run_sharpness.bat

# Results saved to output/sharpness_results/{dataset}/{model}/{attack}/sharpness/results/
```

**Run all generalization experiments:**
```bash
# Windows - from the main project directory
sharpness/run_sharpness_generalization.bat

# Results saved to output/sharpness_generalization_results/
```

## Configuration

### Hyperparameters

The detector uses the following default hyperparameters (from the original paper):

```python
fosc_c = 0.1          # FOSC convergence threshold
warmup_step = 4       # Number of warmup steps for adaptive sizing
adv_steps = 4         # Number of adversarial optimization steps
adv_init_mag = 0.05   # Initial perturbation magnitude
adv_lr = 0.03         # Base learning rate for adversarial updates
adv_max_norm = 0.2    # Maximum perturbation norm constraint
```

### Detection Options

- **`use_adaptive_lr`**: Enable adaptive step size adjustment (default: True)
- **`use_delta_loss`**: Use loss difference vs. absolute perturbed loss (default: False)
- **`adv_norm_type`**: Perturbation norm type ('l2' or 'linf', default: 'l2')
- **`use_fw_gap`**: Use Frank-Wolfe gap instead of FOSC (default: False)
- **`threshold`**: Detection threshold for binary classification (default: 0.25)

## Input Data Format

The detector expects CSV files with the following columns:
- `original_text`: Original (benign) text samples
- `adversarial_text`: Corresponding adversarial examples

Data should be organized as (relative to the project root):
```
data/
├── {dataset}/
│   ├── {model}/
│   │   ├── {attack}/
│   │   │   ├── calibration_data.csv
│   │   │   └── test_data.csv
```

When running from the main project directory, use `--data_dir data`. When running from the `sharpness/` directory, use `--data_dir ../data`.

## Output Format

Results are saved as CSV files containing:

### Standard Metrics
- `accuracy`, `precision`, `recall`, `f1`: Classification performance
- `auc`: Area under ROC curve
- `avg_precision`: Average precision score
- `original_support`, `adversarial_support`: Class counts

### Timing Information
- `cal_time_total_s`: Total calibration time
- `test_time_total_s`: Total evaluation time
- `cal_time_per_sample_s`: Per-sample calibration time
- `test_time_per_sample_s`: Per-sample evaluation time

### Experiment Metadata
- `detection_method`: Always "fosc"
- `analysis_type`: Always "sharpness"
- `dataset`, `model`, `attack`: Experiment configuration
- `random_seed`: Seed used for reproducibility

### Generalization Experiments
Additional fields for cross-domain experiments:
- `experiment_type`: Type of generalization study
- `domain_type`: "in_domain" or "out_of_domain"
- `test_dataset`, `test_model`, `test_attack`: Test configuration
- `train_datasets`, `train_models`, `train_attacks`: Training configuration

## Model Support

The detector supports both local fine-tuned models and pre-trained models from HuggingFace:

### Local Models (if available)
- `models/roberta_{dataset}/`
- `models/deberta_{dataset}/`

### HuggingFace Models (fallback)
- **RoBERTa**: `textattack/roberta-base-{dataset}`
- **DeBERTa**: `microsoft/deberta-base` (requires fine-tuning)

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{zheng-etal-2023-detecting,
    title = "Detecting Adversarial Samples through Sharpness of Loss Landscape",
    author = "Zheng, Rui  and
      Dou, Shihan  and
      Zhou, Yuhao  and
      Liu, Qin  and
      Gui, Tao  and
      Zhang, Qi  and
      Wei, Zhongyu  and
      Huang, Xuanjing  and
      Zhang, Menghan",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.717/",
    doi = "10.18653/v1/2023.findings-acl.717",
    pages = "11282--11298"
}

```