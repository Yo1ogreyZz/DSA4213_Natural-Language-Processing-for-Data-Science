import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append('..')
from config import Config


def evaluate_model(model, dataloader, device):
    """Evaluate model on the provided dataloader"""
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(predicted_labels)
            true_labels.extend(batch["labels"].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted"
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions,
        "true_labels": true_labels
    }

    return metrics


def plot_confusion_matrix(true_labels, predictions, model_name):
    """Plot and save confusion matrix"""
    classes = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "plots"), exist_ok=True)
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "plots", f"{model_name}_confusion_matrix.png"))
    plt.close()


def compare_models(full_metrics, lora_metrics):
    """Compare and visualize performance of different fine-tuning strategies"""
    metrics = ["accuracy", "precision", "recall", "f1"]
    models = ["Full Fine-tuning", "LoRA"]

    # Prepare data for plotting
    data = {
        "Full Fine-tuning": [full_metrics[m] for m in metrics],
        "LoRA": [lora_metrics[m] for m in metrics]
    }

    # Create bar plot
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width / 2, data["Full Fine-tuning"], width, label="Full Fine-tuning")
    plt.bar(x + width / 2, data["LoRA"], width, label="LoRA")

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Performance Comparison: Full Fine-tuning vs LoRA")
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save plot
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "plots"), exist_ok=True)
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "plots", "model_comparison.png"))
    plt.close()

    # Print comparison table
    print("\nModel Performance Comparison:")
    print(f"{'Metric':<10} | {'Full Fine-tuning':<15} | {'LoRA':<15}")
    print("-" * 45)

    for metric in metrics:
        print(f"{metric:<10} | {full_metrics[metric]:.4f}{' ' * 11} | {lora_metrics[metric]:.4f}")