# Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance

This repository contains the implementation and experimental code for adversarial text detection methods, including our novel **Guided Perturbation Sensitivity (GPS)** approach and implementations of two state-of-the-art baseline methods.

## Overview

Adversarial text detection is a critical security challenge in natural language processing. This repository provides:

- **GPS**: Our adversarial detection method using guided perturbation sensitivity analysis
- **Sharpness-based Detection**: Implementation of loss landscape sharpness analysis (Zheng et al., ACL 2023)
- **TextShield**: Implementation of Adaptive Word Importance detection (Shen et al., ICLR 2023)

All methods are evaluated on three datasets (AG News, IMDB, Yelp) with two model architectures (RoBERTa, DeBERTa) against three attack methods (TextFooler, BERT-Attack, DeepWordBug).

## Repository Structure

```
submission/
├── data/                     # Adversarial datasets by dataset/model/attack
├── models/                   # Fine-tuned victim models
├── output/                   # Experimental results
├── gps/                      # GPS method implementation
├── sharpness/                # Sharpness-based detection
├── textshield/               # TextShield (AWI) implementation
└── requirements.txt          # Dependencies
```

## Detection Methods

### 🔬 GPS (Guided Perturbation Sensitivity)
Our method that analyzes word-level sensitivity patterns through guided perturbations based on attribution methods.

**Key Features:**
- Multiple attribution strategies (Gradients, Attention-Rollout, GRAD-SAM, Random Selection)
- Progressive perturbation analysis
- BiLSTM-based classification with attention mechanisms
- Feature extraction pipeline for useage of external classifiers

**Usage:**
```bash
# Run GPS with attention rollout on AG News
python -m gps --model_name roberta_ag_news --data_dir data/ag_news/roberta/textfooler --use_attention --top_n 20

# Extract features only (for external classifiers)
python -m gps.extract_features --model_name roberta_ag_news --data_dir data/ag_news/roberta/textfooler --use_attention --top_n 20
```

### 📊 Sharpness-based Detection
Implementation of loss landscape sharpness analysis using First-Order Stationary Condition (FOSC). Original implementation from [https://github.com/ruizheng20/sharpness_detection](https://github.com/ruizheng20/sharpness_detection).

**Key Features:**
- FOSC and Frank-Wolfe gap convergence metrics
- Adaptive step size optimization
- Cross-domain generalization experiments
- Multiple perturbation norms (L2, L∞)

**Usage:**
```bash
# Standard detection
python sharpness/sharpness_detector.py --dataset imdb --model roberta --attack textfooler --data_dir data --output_dir output

# Cross-domain generalization
python sharpness/sharpness_detector.py --experiment_type cross_dataset --train_datasets yelp --test_datasets imdb --data_dir data
```

### 🛡️ TextShield (AWI)
Implementation of Adaptive Word Importance-based adversarial detection.

**Key Features:**
- Multiple AWI computation methods (Vanilla Gradient, Integrated Gradients, Guided Backpropagation, LRP)
- Multi-channel feature representation
- Optimized batch processing with caching
- BiLSTM-based detection architecture (ours)

**Usage:**
```bash
# Extract AWI features
python textshield/extract_awi_features.py --awi_method all --base_model_dir models/ --base_data_dir data/

# Train and evaluate detector
python textshield/train_evaluate_detector.py --dataset imdb --model roberta --attack textfooler --features_dir data --strategy awi
```

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone [<repository-url>](https://github.com/ReDASers/representation-stability)
   cd submission
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download additional NLTK data (if needed):**
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```

### Running Experiments

**Option 1: Individual Experiments**
```bash
# GPS with different strategies
python -m gps --model_name roberta_imdb --data_dir data/imdb/roberta/textfooler --use_attention --top_n 20
python -m gps --model_name roberta_imdb --data_dir data/imdb/roberta/textfooler --use_saliency --top_n 20

# Sharpness detection
python sharpness/sharpness_detector.py --dataset imdb --model roberta --attack textfooler --data_dir data

# TextShield
python textshield/extract_awi_features.py --awi_method all --base_model_dir models/ --base_data_dir data/
python textshield/train_evaluate_detector.py --dataset imdb --model roberta --attack textfooler --strategy awi
```

**Option 2: Batch Experiments**
```bash
# Run all GPS experiments
gps/scripts/run_all_experiments.bat

# Run all sharpness experiments  
sharpness/run_sharpness.bat

# Run all TextShield experiments
textshield/run_textshield.bat
```

## Data Format

The repository expects adversarial datasets in CSV format with the following structure:

```
data/{dataset}/{model}/{attack}/
├── calibration_data.csv      # Training data
└── test_data.csv            # Test data
```

**Required CSV columns:**
- `original_text`: Original (benign) text samples
- `adversarial_text`: Corresponding adversarial examples

## Models

The system supports both local fine-tuned models and HuggingFace models:

**Local Models** (if available):
- `models/roberta_{dataset}/`
- `models/deberta_{dataset}/`

**HuggingFace Fallback:**
- RoBERTa: `textattack/roberta-base-{dataset}`
- DeBERTa: `microsoft/deberta-base`

## Results and Evaluation

Results are automatically saved to the `output/` directory with several metrics:

- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Timing Information**: Training and evaluation time per sample
- **Method-specific Metrics**: 
  - GPS: Ranking metrics (MRR, MAP, NDCG), overlap analysis
  - Sharpness: FOSC convergence, step size adaptation
  - TextShield: Multi-channel AWI feature statistics

## Citation

If you use this code, please cite our work:

```bibtex
@inproceedings{your-paper-2024,
    title={Your Paper Title},
    author={Your Name and Co-authors},
    booktitle={Conference/Journal Name},
    year={2024}
}
```

**Baseline Methods:**

```bibtex
@inproceedings{zheng-etal-2023-detecting,
    title = "Detecting Adversarial Samples through Sharpness of Loss Landscape",
    author = "Zheng, Rui and others",
    booktitle = "Findings of ACL 2023",
    year = "2023"
}

@inproceedings{shen2023textshield,
    title={TextShield: Beyond Successfully Detecting Adversarial Sentences in text classification},
    author={Lingfeng Shen and Ze Zhang and Haiyun Jiang and Ying Chen},
    booktitle={ICLR 2023},
    year={2023}
}
```

## Requirements

- Python 3.8+
- PyTorch 2.7.1+
- Transformers 4.54.0+
- See `requirements.txt` for complete dependencies

## Hardware Requirements

- **GPU**: Recommended for efficient processing (CUDA-compatible)
- **Memory**: At least 16GB RAM for large model experiments
- **Storage**: ~10GB for models and datasets

## Contributing

For questions or issues, please open a GitHub issue or contact the authors.

## License


This code is provided for academic research purposes.
