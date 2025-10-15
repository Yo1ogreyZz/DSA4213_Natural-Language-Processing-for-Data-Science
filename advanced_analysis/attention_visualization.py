import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import sys

sys.path.append('..')
from config import Config


def visualize_attention_weights(model, tokenizer, example_sentences, model_name, layer_idx=-1):
    """
    Visualize attention weights for the given model and example sentences

    Args:
        model: The model to visualize
        tokenizer: The tokenizer for the model
        example_sentences: List of example sentences for visualization
        model_name: Name of the model for saving plots
        layer_idx: Which transformer layer to visualize (-1 for the last layer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create output directory
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "attention_viz"), exist_ok=True)

    for i, sentence in enumerate(example_sentences):
        # Tokenize input
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=Config.MAX_LENGTH).to(device)

        # Get attention weights
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Get attention from specified layer (default: last layer)
        attention = outputs.attentions[layer_idx].squeeze(0)  # Shape: [num_heads, seq_len, seq_len]

        # Average over attention heads
        attention_avg = attention.mean(dim=0).cpu().numpy()  # Shape: [seq_len, seq_len]

        # Get tokens for visualization
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        # Trim to actual tokens (removing padding)
        seq_len = min(len(tokens), attention_avg.shape[0])
        attention_avg = attention_avg[:seq_len, :seq_len]
        tokens = tokens[:seq_len]

        # Plot attention heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention_avg,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            annot=False,
            square=True
        )
        plt.title(f"Attention Weights - {model_name} - Example {i + 1}")
        plt.xlabel("Target Token")
        plt.ylabel("Source Token")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "attention_viz", f"{model_name}_attention_example_{i + 1}.png"), dpi=150)
        plt.close()

        print(f"Saved attention visualization for {model_name} - Example {i + 1}")


def compare_pre_post_finetuning_attention():
    """Compare attention patterns before and after fine-tuning"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example financial sentences
    example_sentences = [
        "The company reported a significant increase in quarterly profits.",
        "Investors are concerned about the falling stock prices due to market volatility.",
        "The merger is expected to boost revenue in the next fiscal year."
    ]

    # Load pretrained model
    print("Loading pretrained model...")
    pretrained_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, cache_dir=Config.CACHE_DIR)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=3,
        cache_dir=Config.CACHE_DIR
    )

    print("Visualizing pretrained model attention patterns...")
    visualize_attention_weights(pretrained_model, pretrained_tokenizer, example_sentences, "pretrained")

    # Visualize full fine-tuned model if it exists
    full_model_path = os.path.join(Config.OUTPUT_DIR, "full_finetuning")
    if os.path.exists(full_model_path):
        print("Loading full fine-tuned model...")
        try:
            full_tokenizer = AutoTokenizer.from_pretrained(full_model_path)
            full_model = AutoModelForSequenceClassification.from_pretrained(full_model_path)

            print("Visualizing full fine-tuned model attention patterns...")
            visualize_attention_weights(full_model, full_tokenizer, example_sentences, "full_finetuning")
        except Exception as e:
            print(f"Could not load full fine-tuned model: {e}")
    else:
        print(f"Full fine-tuned model not found at {full_model_path}")

    # Visualize LoRA fine-tuned model if it exists
    lora_model_path = os.path.join(Config.OUTPUT_DIR, "lora_tuning")
    if os.path.exists(lora_model_path):
        print("Loading LoRA fine-tuned model...")
        try:
            lora_tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
            # Load base model first, then apply LoRA adapters
            base_model = AutoModelForSequenceClassification.from_pretrained(
                Config.MODEL_NAME,
                num_labels=3,
                cache_dir=Config.CACHE_DIR
            )
            lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

            print("Visualizing LoRA fine-tuned model attention patterns...")
            visualize_attention_weights(lora_model, lora_tokenizer, example_sentences, "lora_tuning")
        except Exception as e:
            print(f"Could not load LoRA fine-tuned model: {e}")
    else:
        print(f"LoRA fine-tuned model not found at {lora_model_path}")

    print("\nAttention visualization completed!")


if __name__ == "__main__":
    compare_pre_post_finetuning_attention()