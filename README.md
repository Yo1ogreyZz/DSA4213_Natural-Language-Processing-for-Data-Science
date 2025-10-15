# DSA4213 Natural Language Processing for Data Science - Assignment 3

## Fine-tuning Pretrained Transformers for Financial Sentiment Analysis

This repository contains the implementation of Assignment 3 for DSA4213, focusing on fine-tuning pretrained Transformer models for domain-specific text classification tasks.

### Project Overview

This project implements and compares different fine-tuning strategies for adapting a pretrained RoBERTa model to financial sentiment analysis:
- **Full Fine-tuning**: Updates all model parameters
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning method

### Dataset

- **Domain**: Financial sentiment analysis
- **Source**: Twitter Financial News Sentiment / FiQA-2018 (automatically downloaded)
- **Task**: 3-class sentiment classification (Negative, Neutral, Positive)
- **Size**: Automatically split into train/validation/test sets (72%/8%/20%)

The dataset is chosen because financial text contains domain-specific terminology and requires specialized understanding, making it an ideal case for demonstrating the effectiveness of fine-tuning.

### Model

- **Base Model**: RoBERTa-base (pretrained on general text)
- **Task**: Sequence classification with 3 labels
- **Framework**: PyTorch + HuggingFace Transformers + PEFT

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd DSA4213_Natural-Language-Processing-for-Data-Science
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Required packages**:
- transformers>=4.30.0
- datasets>=2.12.0
- torch>=1.13.0
- scikit-learn>=1.2.2
- pandas>=1.5.3
- matplotlib>=3.7.1
- seaborn>=0.12.2
- peft>=0.4.0
- tqdm>=4.65.0

### Usage

#### 1. Download Dataset Only
```bash
python main.py --mode download_only
```

#### 2. Run Full Fine-tuning Only
```bash
python main.py --mode full
```

#### 3. Run LoRA Fine-tuning Only
```bash
python main.py --mode lora
```

#### 4. Run Both Fine-tuning Methods (Recommended)
```bash
python main.py --mode all
```

#### 5. Run Advanced Analysis
```bash
python main.py --mode advanced_analysis
```

This includes:
- LoRA hyperparameter ablation study (testing different rank and alpha values)
- Attention pattern visualization
- Domain adaptation analysis

### Project Structure

```
.
├── main.py                      # Main entry point
├── config.py                    # Configuration parameters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/
│   ├── download_alternative.py # Dataset download script
│   ├── preprocess.py           # Data preprocessing utilities
│   ├── train.csv               # Training data (generated)
│   ├── val.csv                 # Validation data (generated)
│   └── test.csv                # Test data (generated)
├── models/
│   ├── model_utils.py          # Model loading utilities
│   ├── full_finetuning.py      # Full fine-tuning implementation
│   └── lora_tuning.py          # LoRA fine-tuning implementation
├── evaluation/
│   └── metrics.py              # Evaluation metrics and visualization
├── advanced_analysis/
│   ├── lora_ablation.py        # LoRA hyperparameter study
│   ├── attention_visualization.py  # Attention pattern analysis
│   └── domain_adaptation.py    # Domain-specific analysis
└── outputs/
    ├── full_finetuning/        # Saved full fine-tuned model
    ├── lora_tuning/            # Saved LoRA adapters
    ├── ablation/               # Ablation study results
    ├── plots/                  # Generated visualizations
    └── results/                # Performance metrics (JSON/CSV)
```

### Experimental Setup

#### Full Fine-tuning
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 5
- Weight decay: 0.01
- Warmup ratio: 0.1

#### LoRA Fine-tuning
- Batch size: 32 (can use larger due to fewer parameters)
- Learning rate: 5e-4 (higher learning rate)
- Epochs: 5
- LoRA rank (r): 8
- LoRA alpha: 32
- LoRA dropout: 0.1
- Target modules: query and value projections

### Results

After running the experiments, results will be saved in:
- `outputs/results/metrics.json` - Detailed metrics in JSON format
- `outputs/results/performance.csv` - Performance comparison table
- `outputs/results/efficiency.csv` - Training time and parameter comparison
- `outputs/plots/` - Confusion matrices and comparison plots

### Key Takeaways

1. **Performance**: Both methods achieve comparable accuracy, with LoRA performing within 1-2% of full fine-tuning
2. **Efficiency**: LoRA reduces trainable parameters by ~1000x while maintaining performance
3. **Speed**: LoRA training is typically 2-3x faster than full fine-tuning
4. **Memory**: LoRA requires significantly less GPU memory

### Advanced Analysis

The project includes three advanced analysis components:

1. **LoRA Ablation Study**: Tests different combinations of rank (4, 8, 16, 32) and alpha (16, 32, 64) to understand hyperparameter effects
2. **Attention Visualization**: Compares attention patterns before and after fine-tuning
3. **Domain Adaptation**: Analyzes performance on financial vs. general text

### Reproducibility

All experiments use fixed random seeds (RANDOM_STATE=42) for reproducibility. The code automatically detects and uses GPU if available, otherwise falls back to CPU.

### Notes

- First run will download the dataset and pretrained model (may take a few minutes)
- Models and datasets are cached to avoid re-downloading
- GPU is recommended for faster training (but not required)
- Free platforms like Google Colab or Kaggle can be used if local resources are limited

### Author

DSA4213 Assignment 3 - Sem1 2025 Fall

### License

For educational purposes only.

---

## Assignment1

### Objective:

-  Understand and implement key word embedding algorithms 
- Train embeddings using real corpora 
- Compare model outputs via qualitative and optional quantitative analyses 
- Practice scientific reporting and reproducible research

### Overview:

-  **Individual assignment**

- **Report:** PDF with embedded figures and appendix for code

- **Deliverables:** 

  1. Algorithm explanation

  2. Data preprocessing
  3. Model training + visualization
  4. Comparison and analysis

### Algorithms to implement

- **Skip-gram**
- **SPPMI-SVD**
- **GloVe**

   Each model should be trained on the same corpus and compared fairly.

   Corpus option: anything that you feel interesting, for example, English Wikipedia, https://www.gutenberg.org/, ...

### Required analysis

-  **Nearest neighbors** for selected words
- **Qualitative evaluation:** Are the similar words reasonable? 
- **t-SNE/UMAP/PCA visualization** of embedding space
- **Comparison across models:** what differs, what patterns arise?

### Bonus opportunities

- Benchmark evaluation (e.g., WordSim-353, analogy tasks)
- Using embeddings for text classification or clustering
- Discovering interesting phenomena: semantic drift, gender bias, etc.