import os
import torch
import time
import argparse
import json
import pandas as pd
from transformers import AutoModelForSequenceClassification
from config import Config
# Change the import to use the alternative dataset
from data.download_alternative import download_alternative_dataset
from models.full_finetuning import train_full_finetuning
from models.lora_tuning import train_with_lora
from evaluation.metrics import plot_confusion_matrix, compare_models
from advanced_analysis.lora_ablation import run_lora_ablation_study
from advanced_analysis.attention_visualization import compare_pre_post_finetuning_attention
from advanced_analysis.domain_adaptation import analyze_domain_adaptation

# Import psutil if available, otherwise define a simple memory tracking function
try:
    import psutil


    def get_memory_usage():
        return psutil.Process().memory_info().rss / (1024 * 1024)  # MB
except ImportError:
    def get_memory_usage():
        return 0  # Fallback if psutil is not available


def setup_directories():
    """Create necessary directories"""
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "results"), exist_ok=True)


def save_results(full_metrics, lora_metrics, full_time, lora_time, full_params, lora_params):
    """Save metrics to JSON file"""
    results = {
        "performance": {
            "full_finetuning": {
                "accuracy": full_metrics["accuracy"],
                "precision": full_metrics["precision"],
                "recall": full_metrics["recall"],
                "f1": full_metrics["f1"]
            },
            "lora": {
                "accuracy": lora_metrics["accuracy"],
                "precision": lora_metrics["precision"],
                "recall": lora_metrics["recall"],
                "f1": lora_metrics["f1"]
            }
        },
        "efficiency": {
            "full_finetuning_time": full_time,
            "lora_time": lora_time,
            "speedup": full_time / lora_time if lora_time > 0 else "N/A"
        },
        "parameters": {
            "full_finetuning_params": full_params,
            "lora_params": lora_params,
            "reduction_factor": full_params / lora_params if lora_params > 0 else "N/A"
        }
    }

    with open(os.path.join(Config.OUTPUT_DIR, "results", "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Also save as CSV for easier analysis
    performance_df = pd.DataFrame({
        "Metric": ["accuracy", "precision", "recall", "f1"],
        "Full Fine-tuning": [
            full_metrics["accuracy"],
            full_metrics["precision"],
            full_metrics["recall"],
            full_metrics["f1"]
        ],
        "LoRA": [
            lora_metrics["accuracy"],
            lora_metrics["precision"],
            lora_metrics["recall"],
            lora_metrics["f1"]
        ]
    })

    efficiency_df = pd.DataFrame({
        "Metric": ["Training Time (s)", "Trainable Parameters"],
        "Full Fine-tuning": [full_time, full_params],
        "LoRA": [lora_time, lora_params]
    })

    performance_df.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "performance.csv"), index=False)
    efficiency_df.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "efficiency.csv"), index=False)


def count_trainable_parameters(model_path, model_type):
    """Count trainable parameters in a model"""
    try:
        if model_type == "full":
            # For full fine-tuning, all parameters are trainable
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:  # LoRA
            # For LoRA, only adapter parameters are trainable
            model_file = os.path.join(model_path, "adapter_model.bin")
            if not os.path.exists(model_file):
                return 0

            state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            return sum(p.numel() for name, p in state_dict.items() if "lora" in name.lower())
    except Exception as e:
        print(f"Error counting parameters: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Pretrained Transformers")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "full", "lora", "download_only", "advanced_analysis"],
        help="Training mode: all, full (only full fine-tuning), lora (only LoRA), or download_only, or advanced_analysis"
    )

    args = parser.parse_args()

    # Set up directories
    setup_directories()

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download and prepare dataset
    print("Step 1: Downloading and preparing dataset...")
    try:
        # Use the alternative dataset
        download_alternative_dataset()
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return

    if args.mode == "download_only":
        print("Dataset downloaded. Exiting as requested.")
        return

    # Track memory usage
    memory_usage_start = get_memory_usage()

    full_metrics = {}
    lora_metrics = {}
    full_time = 0
    lora_time = 0

    # Full fine-tuning
    if args.mode in ["all", "full"]:
        print("\nStep 2: Starting full fine-tuning...")
        start_time = time.time()
        try:
            full_metrics = train_full_finetuning()
            full_time = time.time() - start_time
            print(f"Full fine-tuning completed in {full_time:.2f} seconds")

            # Plot confusion matrix for full fine-tuning
            plot_confusion_matrix(
                full_metrics["true_labels"],
                full_metrics["predictions"],
                "Full_Finetuning"
            )
        except Exception as e:
            print(f"Error during full fine-tuning: {e}")
            full_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "predictions": [], "true_labels": []}

    # LoRA fine-tuning
    if args.mode in ["all", "lora"]:
        print("\nStep 3: Starting LoRA fine-tuning...")
        start_time = time.time()
        try:
            lora_metrics = train_with_lora()
            lora_time = time.time() - start_time
            print(f"LoRA fine-tuning completed in {lora_time:.2f} seconds")

            # Plot confusion matrix for LoRA
            plot_confusion_matrix(
                lora_metrics["true_labels"],
                lora_metrics["predictions"],
                "LoRA"
            )
        except Exception as e:
            print(f"Error during LoRA fine-tuning: {e}")
            lora_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "predictions": [], "true_labels": []}

    # Count parameters
    try:
        full_params = count_trainable_parameters(
            os.path.join(Config.OUTPUT_DIR, "full_finetuning"),
            "full"
        )

        lora_params = count_trainable_parameters(
            os.path.join(Config.OUTPUT_DIR, "lora_tuning"),
            "lora"
        )
    except Exception as e:
        print(f"Error counting parameters: {e}")
        full_params = 0
        lora_params = 0

    # Compare models if both were trained
    if args.mode == "all" and full_metrics and lora_metrics:
        print("\nStep 4: Comparing models...")
        try:
            compare_models(full_metrics, lora_metrics)

            # Get final memory usage
            memory_usage_end = get_memory_usage()
            memory_increase = memory_usage_end - memory_usage_start

            # Save results
            save_results(full_metrics, lora_metrics, full_time, lora_time, full_params, lora_params)

            # Print time comparison
            print(f"\nTraining time comparison:")
            print(f"Full fine-tuning: {full_time:.2f} seconds")
            print(f"LoRA fine-tuning: {lora_time:.2f} seconds")
            if lora_time > 0:
                print(f"Speed-up with LoRA: {full_time / lora_time:.2f}x")

            # Print parameter comparison
            print(f"\nTrainable parameter comparison:")
            print(f"Full fine-tuning: {full_params:,} parameters")
            print(f"LoRA fine-tuning: {lora_params:,} parameters")
            if lora_params > 0:
                print(f"Parameter reduction with LoRA: {full_params / lora_params:.2f}x")

            # Print memory usage
            print(f"\nMemory usage:")
            print(f"Increase during execution: {memory_increase:.2f} MB")
        except Exception as e:
            print(f"Error comparing models: {e}")

    # Run advanced analysis
    if args.mode in ["all", "advanced_analysis"]:
        print("\nStep 5: Running advanced analysis...")

        try:
            # LoRA hyperparameter ablation study
            print("\n5.1: Running LoRA hyperparameter ablation study...")
            run_lora_ablation_study()
        except Exception as e:
            print(f"Error during LoRA ablation study: {e}")

        try:
            # Attention visualization
            print("\n5.2: Visualizing attention patterns...")
            compare_pre_post_finetuning_attention()
        except Exception as e:
            print(f"Error during attention visualization: {e}")

        try:
            # Domain adaptation analysis
            print("\n5.3: Analyzing domain adaptation...")
            analyze_domain_adaptation()
        except Exception as e:
            print(f"Error during domain adaptation analysis: {e}")


if __name__ == "__main__":
    main()