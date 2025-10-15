import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys

sys.path.append('..')
from config import Config


class FinancialPhraseDataset(Dataset):
    """Dataset class for Financial PhraseBank"""

    def __init__(self, data_path, tokenizer, max_length=Config.MAX_LENGTH):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["sentence"]
        label = self.data.iloc[idx]["label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Remove batch dimension added by the tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(tokenizer, batch_size):
    """Create DataLoaders for train, validation, and test sets"""

    # Create datasets
    train_dataset = FinancialPhraseDataset(
        data_path=f"{Config.DATA_DIR}/train.csv",
        tokenizer=tokenizer
    )

    val_dataset = FinancialPhraseDataset(
        data_path=f"{Config.DATA_DIR}/val.csv",
        tokenizer=tokenizer
    )

    test_dataset = FinancialPhraseDataset(
        data_path=f"{Config.DATA_DIR}/test.csv",
        tokenizer=tokenizer
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader