# Representation Stability (RS) for Adversarial Text Detection

RS is a novel adversarial text detection system that analyzes word-level sensitivity patterns through guided perturbations based on different importance heuristics. The system detects adversarial examples by examining how model predictions change when words are systematically masked according to attribution methods.

## Architecture

The codebase is organized into modules for maintainability and extensibility:

```
gps/
├── main.py                     # Main entry point
├── extract_features.py         # Feature extraction only (no training)
├── core/                       # Core algorithms and pipelines
│   ├── pipeline.py             # End-to-end sensitivity analysis pipeline
│   └── sensitivity.py          # Sensitivity computation and feature extraction
├── analysis/                   # Analysis and attribution methods
│   ├── attribution.py          # Word importance attribution (attention, grad, etc.)
│   └── perturbation_eval.py    # Perturbation overlap analysis and ranking metrics
├── models/                     # Detection models
│   └── bilstm.py              # BiLSTM classifier with attention mechanisms
├── training/                   # Training and evaluation utilities
│   └── detector.py            # Model training and performance evaluation
├── utils/                      # Preprocessing utilities
│   └── text_processing.py     # Text tokenization and preprocessing
└── scripts/                    # Experiment runners
    ├── run_all_experiments.bat
    ├── run_ag_news.bat
    ├── run_imdb.bat
    ├── run_yelp.bat
    └── run_topk.bat
```

## Method Overview

RS operates through four key stages:

1. **Word Selection**: Uses attribution methods (Gradients, Attention-Rollout, GRAD-SAM, Random Selection) to rank word importance
2. **Progressive Perturbation**: Systematically masks top-k words and computes embedding changes
3. **Sensitivity Feature Extraction**: Extracts distributional and sequential features from sensitivity patterns
4. **Classification**: Uses BiLSTM with attention to distinguish adversarial from benign examples

## Word Selection Strategies

- **Gradient Attribution**: Computes Vanilla Gradients
- **Attention Rollout**: Tracks attention flow through transformer layers
- **Integrated Gradients**: Computes attribution through gradient integration
- **Gradient×Attention**: Combines gradient and attention information (Grad-SAM)
- **Random Baseline**: Random word selection for comparison

## Model Configuration

RS supports both default pre-trained models and custom models:

- **Default Models**: Uses fine-tuned models from the `redasers` HuggingFace community (redasers/roberta_ag_news, redasers/deberta_imdb, etc.)
  - Available models: https://huggingface.co/collections/redasers/representation-stability-689396925331dfaddaf59f09
- **Custom Models**: Specify any HuggingFace model ID or local model path using `--model_path`

## Usage

Run experiments on specific datasets:

```bash
# AG News dataset with RoBERTa (uses default redasers/roberta_ag_news from HuggingFace)
python -m gps --model_name roberta_ag_news --data_dir data/ag_news/roberta/textfooler --use_attention --top_n 20

# IMDB dataset with DeBERTa using gradient attribution (uses default redasers/deberta_imdb from HuggingFace)
python -m gps --model_name deberta_imdb --data_dir data/imdb/deberta/bert-attack --use_saliency --top_n 20

# Use Grad-SAM strategy (uses default redasers/roberta_yelp from HuggingFace)
python -m gps --model_name roberta_yelp --data_dir data/yelp/roberta/deepwordbug --use_gradient_attention --top_n 20

# Use a custom model from HuggingFace
python -m gps --model_name custom_model --model_path your_username/your_model --data_dir data/imdb/roberta/textfooler --use_attention --top_n 20

# Use a local model directory
python -m gps --model_name local_model --model_path ./path/to/your/local/model --data_dir data/imdb/roberta/textfooler --use_saliency --top_n 20
```

Or use the provided batch scripts for comprehensive experiments:

```bash
# Run all strategies on AG News
gps/scripts/run_ag_news.bat

# Run all strategies on IMDB  
gps/scripts/run_imdb.bat

# Run all strategies on Yelp
gps/scripts/run_yelp.bat

# Run saliency strategy with multiple TOP_N values (5-50) on bert-attack across all datasets
gps/scripts/run_topk.bat

# Run all experiments across datasets and strategies
gps/scripts/run_all_experiments.bat
```

## Feature Extraction Only

To extract RS features without training detection models (for use with external classifiers):

```bash
# Extract features only - outputs will be saved as JSON files (uses default redasers model)
python -m gps.extract_features --model_name roberta_ag_news --data_dir data/ag_news/roberta/textfooler --use_attention --top_n 20

# Extract features using a custom model
python -m gps.extract_features --model_name custom_model --model_path your_username/your_model --data_dir data/ag_news/roberta/textfooler --use_attention --top_n 20
```

This creates:
- `calibration_features.json`: Training features with labels (0=original, 1=adversarial)  
- `test_features.json`: Test features with labels
- Metadata including strategy, parameters, and sample counts

## Key Parameters

- `--model_name`: Target model identifier (e.g., `roberta_ag_news`, `deberta_imdb`)
- `--model_path`: Custom model path (HuggingFace model ID or local path). Optional - defaults to `redasers/{model_name}`
- `--data_dir`: Path to adversarial attack data
- `--top_n`: Number of top-ranked words to analyze (default: 20)
- `--distance_metric`: Embedding distance metric (`cosine`, `euclidean`, `manhattan`)
- `--detection_method`: Detection model type (`bilstm`)

## Output

The system generates:

- **Sensitivity features**: Sequential and distributional patterns from word removal
- **Detection performance**: Accuracy, precision, recall, F1-score, AUC
- **Ranking metrics**: Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), NDCG
- **Overlap analysis**: Comparison between attribution methods and ground-truth perturbations

Results are saved to the `output/` directory with detailed CSV reports and performance metrics.

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@misc{tuck2025assessingrepresentationstabilitytransformer,
      title={Assessing Representation Stability for Transformer Models}, 
      author={Bryan E. Tuck and Rakesh M. Verma},
      year={2025},
      eprint={2508.11667},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.11667}, 
}
```
