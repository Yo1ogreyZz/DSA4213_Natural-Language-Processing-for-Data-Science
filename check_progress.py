import os
import time
import json

def check_training_progress():
    """Check the progress of training by examining output files"""

    print("=" * 60)
    print("Training Progress Monitor")
    print("=" * 60)

    # Check if LoRA model is saved
    lora_path = "./outputs/lora_tuning"
    if os.path.exists(lora_path):
        print("✅ LoRA model directory exists")
        files = os.listdir(lora_path)
        print(f"   Files: {len(files)} files saved")
        if "adapter_model.safetensors" in files:
            print("   ✅ LoRA adapter saved successfully")
    else:
        print("⏳ LoRA model not yet saved")

    # Check results
    results_path = "./outputs/results"
    if os.path.exists(results_path):
        print("✅ Results directory exists")
        if os.path.exists(os.path.join(results_path, "metrics.json")):
            with open(os.path.join(results_path, "metrics.json"), "r") as f:
                metrics = json.load(f)
            print("   ✅ Metrics saved:")
            if "performance" in metrics:
                perf = metrics["performance"]
                if "lora" in perf:
                    print(f"      LoRA Accuracy: {perf['lora']['accuracy']:.4f}")
                    print(f"      LoRA F1: {perf['lora']['f1']:.4f}")
    else:
        print("⏳ Results not yet available")

    # Check plots
    plots_path = "./outputs/plots"
    if os.path.exists(plots_path):
        plot_files = [f for f in os.listdir(plots_path) if f.endswith('.png')]
        print(f"✅ {len(plot_files)} plot(s) generated")
    else:
        print("⏳ Plots not yet generated")

    print("=" * 60)

if __name__ == "__main__":
    check_training_progress()

