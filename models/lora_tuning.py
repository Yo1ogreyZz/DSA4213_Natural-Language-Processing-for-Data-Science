import os
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
import sys

sys.path.append('..')
from config import Config
from data.preprocess import create_dataloaders
from models.model_utils import get_model_and_tokenizer, save_model
from evaluation.metrics import evaluate_model


def train_with_lora(rank=Config.LORA_RANK, alpha=Config.LORA_ALPHA,
                    lr=Config.LORA_LEARNING_RATE, output_suffix=""):
    """Train the model using LoRA (Low-Rank Adaptation) approach"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get base model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=Config.LORA_DROPOUT,
        # Apply LoRA to query and value projection matrices in attention
        target_modules=["query", "value"]
    )

    # Apply LoRA adapters to the model
    model = get_peft_model(model, peft_config)
    model = model.to(device)

    # Print trainable parameters info
    model.print_trainable_parameters()

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=Config.LORA_BATCH_SIZE
    )

    # Setup optimizer - only train the LoRA parameters
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=Config.LORA_WEIGHT_DECAY
    )

    # Setup learning rate scheduler
    num_training_steps = Config.LORA_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(Config.LORA_WARMUP_RATIO * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Training loop
    progress_bar = tqdm(range(num_training_steps))
    best_val_accuracy = 0
    model_type = f"lora_tuning{output_suffix}"

    for epoch in range(Config.LORA_EPOCHS):
        # Training
        model.train()
        train_loss = 0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        val_metrics = evaluate_model(model, val_dataloader, device)
        val_accuracy = val_metrics["accuracy"]

        print(f"Epoch {epoch + 1}/{Config.LORA_EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, tokenizer, model_type)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_dataloader, device)

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")

    return test_metrics


if __name__ == "__main__":
    train_with_lora()