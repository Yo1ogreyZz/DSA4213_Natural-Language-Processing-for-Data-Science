import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys

sys.path.append('..')
from config import Config


def get_model_and_tokenizer():
    """Load pretrained RoBERTa model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=Config.CACHE_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=3,  # Negative, Neutral, Positive
        cache_dir=Config.CACHE_DIR
    )

    return model, tokenizer


def save_model(model, tokenizer, model_type):
    """Save the fine-tuned model and tokenizer"""
    output_dir = os.path.join(Config.OUTPUT_DIR, model_type)
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")