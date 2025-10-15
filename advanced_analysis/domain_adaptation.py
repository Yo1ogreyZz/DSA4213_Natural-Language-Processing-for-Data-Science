import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import sys

sys.path.append('..')
from config import Config
from evaluation.metrics import evaluate_model
from data.preprocess import FinancialPhraseDataset


def analyze_domain_adaptation():
    """Analyze how well the model adapts to financial domain-specific terms"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_df = pd.read_csv(os.path.join(Config.DATA_DIR, "test.csv"))

    # Define financial terms to look for
    financial_terms = [
        "revenue", "profit", "dividend", "stock", "market", "investment",
        "quarterly", "fiscal", "earnings", "growth", "loss", "debt",
        "assets", "liability", "portfolio", "shares", "trading", "volatility",
        "equity", "capital", "cash flow", "merger", "acquisition", "ipo"
    ]

    # Identify sentences containing financial terms
    financial_samples = []
    general_samples = []

    for idx, row in test_df.iterrows():
        sentence = row["sentence"].lower()
        contains_financial_term = any(term in sentence for term in financial_terms)

        if contains_financial_term:
            financial_samples.append(row)
        else:
            general_samples.append(row)

    print(f"Financial domain sentences: {len(financial_samples)}")
    print(f"General sentences: {len(general_samples)}")

    # Skip if no general samples
    if len(general_samples) == 0:
        print("No general samples found, skipping domain adaptation analysis")
        return

    # Convert to DataFrames
    financial_df = pd.DataFrame(financial_samples)
    general_df = pd.DataFrame(general_samples)

    # Create temporary files for dataset creation
    financial_path = os.path.join(Config.DATA_DIR, "financial_subset.csv")
    general_path = os.path.join(Config.DATA_DIR, "general_subset.csv")

    financial_df.to_csv(financial_path, index=False)
    general_df.to_csv(general_path, index=False)

    # Load models
    models = {}
    tokenizers = {}

    # Original pre-trained model
    print("Loading pretrained model...")
    tokenizers["pretrained"] = AutoTokenizer.from_pretrained(Config.MODEL_NAME, cache_dir=Config.CACHE_DIR)
    models["pretrained"] = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=3,
        cache_dir=Config.CACHE_DIR
    ).to(device)

    # Full fine-tuned model
    full_path = os.path.join(Config.OUTPUT_DIR, "full_finetuning")
    if os.path.exists(full_path):
        print("Loading full fine-tuned model...")
        try:
            tokenizers["full_finetuning"] = AutoTokenizer.from_pretrained(full_path)
            models["full_finetuning"] = AutoModelForSequenceClassification.from_pretrained(full_path).to(device)
        except Exception as e:
            print(f"Could not load full fine-tuned model: {e}")

    # LoRA fine-tuned model
    lora_path = os.path.join(Config.OUTPUT_DIR, "lora_tuning")
    if os.path.exists(lora_path):
        print("Loading LoRA fine-tuned model...")
        try:
            tokenizers["lora_tuning"] = AutoTokenizer.from_pretrained(lora_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                Config.MODEL_NAME,
                num_labels=3,
                cache_dir=Config.CACHE_DIR
            )
            models["lora_tuning"] = PeftModel.from_pretrained(base_model, lora_path).to(device)
        except Exception as e:
            print(f"Could not load LoRA fine-tuned model: {e}")

    # Evaluate each model on both subsets
    results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        tokenizer = tokenizers[model_name]

        # Create datasets
        financial_dataset = FinancialPhraseDataset(financial_path, tokenizer)
        general_dataset = FinancialPhraseDataset(general_path, tokenizer)

        # Create dataloaders
        financial_loader = torch.utils.data.DataLoader(financial_dataset, batch_size=16, shuffle=False)
        general_loader = torch.utils.data.DataLoader(general_dataset, batch_size=16, shuffle=False)

        # Evaluate
        financial_metrics = evaluate_model(model, financial_loader, device)
        general_metrics = evaluate_model(model, general_loader, device)

        results[model_name] = {
            "financial": financial_metrics,
            "general": general_metrics
        }

        print(f"{model_name} - Financial accuracy: {financial_metrics['accuracy']:.4f}, F1: {financial_metrics['f1']:.4f}")
        print(f"{model_name} - General accuracy: {general_metrics['accuracy']:.4f}, F1: {general_metrics['f1']:.4f}")

    # Clean up temporary files
    if os.path.exists(financial_path):
        os.remove(financial_path)
    if os.path.exists(general_path):
        os.remove(general_path)

    # Analyze results
    visualize_domain_results(results)
    return results


def visualize_domain_results(results):
    """Visualize the results of domain adaptation analysis"""
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "domain_analysis"), exist_ok=True)

    # Create DataFrame for easier plotting
    data = []
    for model_name, model_results in results.items():
        for domain_type, metrics in model_results.items():
            for metric_name, value in metrics.items():
                if metric_name in ["accuracy", "precision", "recall", "f1"]:
                    data.append({
                        "Model": model_name,
                        "Domain": domain_type,
                        "Metric": metric_name,
                        "Value": value
                    })

    df = pd.DataFrame(data)

    # Plot for each metric
    for metric in ["accuracy", "f1", "precision", "recall"]:
        metric_df = df[df["Metric"] == metric]

        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x="Model", y="Value", hue="Domain", data=metric_df)

        plt.title(f"{metric.capitalize()} across Different Domains")
        plt.xlabel("Model")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.legend(title="Domain")
        plt.xticks(rotation=15)

        # Add value labels on bars
        for p in chart.patches:
            height = p.get_height()
            if height > 0:
                chart.text(p.get_x() + p.get_width() / 2., height + 0.01,
                          f'{height:.3f}',
                          ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "domain_analysis", f"{metric}_by_domain.png"), dpi=150)
        plt.close()

    # Create a summary comparison plot
    plt.figure(figsize=(14, 8))

    # Group by model and calculate improvement
    summary_data = []
    for model_name in results.keys():
        fin_acc = results[model_name]["financial"]["accuracy"]
        gen_acc = results[model_name]["general"]["accuracy"]
        improvement = fin_acc - gen_acc

        summary_data.append({
            "Model": model_name,
            "Financial": fin_acc,
            "General": gen_acc,
            "Improvement": improvement
        })

    summary_df = pd.DataFrame(summary_data)

    x = range(len(summary_df))
    width = 0.35

    plt.bar([i - width/2 for i in x], summary_df["Financial"], width, label="Financial", alpha=0.8)
    plt.bar([i + width/2 for i in x], summary_df["General"], width, label="General", alpha=0.8)

    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Domain Adaptation Analysis: Financial vs General Text")
    plt.xticks(x, summary_df["Model"], rotation=15)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "domain_analysis", "domain_comparison_summary.png"), dpi=150)
    plt.close()

    print("\nDomain adaptation analysis completed!")
    print(f"Plots saved to {os.path.join(Config.OUTPUT_DIR, 'domain_analysis')}")


if __name__ == "__main__":
    analyze_domain_adaptation()