import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys

sys.path.append('..')
from config import Config
from models.lora_tuning import train_with_lora


def run_lora_ablation_study():
    """Perform ablation study on LoRA hyperparameters (rank and alpha)"""
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "ablation"), exist_ok=True)
    results = {}

    for rank in Config.ABLATION_RANKS:
        for alpha in Config.ABLATION_ALPHAS:
            config_name = f"r{rank}_alpha{alpha}"
            print(f"\n--- Starting LoRA ablation study for {config_name} ---")

            # Train with current hyperparameters
            metrics = train_with_lora(
                rank=rank,
                alpha=alpha,
                output_suffix=f"_ablation_{config_name}"
            )

            # Save results
            results[config_name] = {
                "rank": rank,
                "alpha": alpha,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"]
            }

            # Save intermediate results to file
            with open(os.path.join(Config.OUTPUT_DIR, "ablation", "results.json"), "w") as f:
                json.dump(results, f, indent=4)

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.to_csv(os.path.join(Config.OUTPUT_DIR, "ablation", "results.csv"))

    # Visualize results
    visualize_ablation_results(results)

    return results


def visualize_ablation_results(results):
    """Create visualizations for ablation study results"""
    ranks = Config.ABLATION_RANKS
    alphas = Config.ABLATION_ALPHAS

    # Create a grid for each metric
    metrics = ["accuracy", "f1", "precision", "recall"]

    for metric in metrics:
        # Prepare data for heatmap
        data = np.zeros((len(ranks), len(alphas)))

        for i, rank in enumerate(ranks):
            for j, alpha in enumerate(alphas):
                config_name = f"r{rank}_alpha{alpha}"
                if config_name in results:
                    data[i, j] = results[config_name][metric]

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt=".4f", xticklabels=alphas, yticklabels=ranks, cmap="viridis")
        plt.xlabel("Alpha")
        plt.ylabel("Rank")
        plt.title(f"LoRA {metric.capitalize()} with Different Ranks and Alphas")
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "ablation", f"lora_{metric}_heatmap.png"))
        plt.close()

    # Plot rank vs. performance
    plt.figure(figsize=(12, 8))

    for metric in metrics:
        values = []
        for rank in ranks:
            # Average over all alphas for this rank
            metric_values = [results[f"r{rank}_alpha{alpha}"][metric] for alpha in alphas if
                             f"r{rank}_alpha{alpha}" in results]
            values.append(np.mean(metric_values))

        plt.plot(ranks, values, marker='o', label=metric)

    plt.xlabel("Rank")
    plt.ylabel("Performance")
    plt.title("Effect of LoRA Rank on Performance (averaged over all alphas)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "ablation", "rank_effect.png"))
    plt.close()

    # Plot alpha vs. performance
    plt.figure(figsize=(12, 8))

    for metric in metrics:
        values = []
        for alpha in alphas:
            # Average over all ranks for this alpha
            metric_values = [results[f"r{rank}_alpha{alpha}"][metric] for rank in ranks if
                             f"r{rank}_alpha{alpha}" in results]
            values.append(np.mean(metric_values))

        plt.plot(alphas, values, marker='o', label=metric)

    plt.xlabel("Alpha")
    plt.ylabel("Performance")
    plt.title("Effect of LoRA Alpha on Performance (averaged over all ranks)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "ablation", "alpha_effect.png"))
    plt.close()


if __name__ == "__main__":
    run_lora_ablation_study()