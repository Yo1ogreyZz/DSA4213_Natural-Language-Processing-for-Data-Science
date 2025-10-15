import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import sys

sys.path.append('..')
from config import Config


def download_alternative_dataset():
    """Download an alternative financial sentiment dataset and split it into train/val/test sets"""
    print("Downloading alternative financial sentiment dataset...")

    # Try loading FiQA dataset
    try:
        dataset = load_dataset("pauri32/fiqa-2018")

        # Convert to pandas DataFrame for easier handling
        df = pd.DataFrame(dataset["train"])

        # Keep only necessary columns and rename for compatibility
        df = df[["sentence", "sentiment_score"]]

        # Convert sentiment scores to categorical labels
        # Typically in FiQA: negative < -0.2, positive > 0.2, neutral otherwise
        def score_to_label(score):
            if score < -0.2:
                return "negative"
            elif score > 0.2:
                return "positive"
            else:
                return "neutral"

        df["sentiment"] = df["sentiment_score"].apply(score_to_label)

    except Exception as e:
        print(f"Error loading FiQA dataset: {e}")
        print("Falling back to Twitter Financial News Sentiment dataset...")

        # Fall back to Twitter Financial News Sentiment dataset
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

        # Convert to pandas DataFrame for easier handling
        df = pd.DataFrame(dataset["train"])

        # Map sentiment labels
        sentiment_mapping = {
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive"
        }
        df["sentiment"] = df["label"].map(sentiment_mapping)
        df["sentence"] = df["text"]  # rename column for consistency

    # Map sentiment to numeric labels
    df["label"] = df["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})

    # Display dataset statistics
    print(f"Dataset size: {len(df)} samples")
    print(f"Label distribution:\n{df['sentiment'].value_counts()}")

    # Split the dataset into train, validation, and test sets
    train_df, test_df = train_test_split(
        df, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=df["label"]
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=Config.VAL_SIZE / (1 - Config.TEST_SIZE),
        random_state=Config.RANDOM_STATE,
        stratify=train_df["label"]
    )

    # Create data directory if it doesn't exist
    os.makedirs(Config.DATA_DIR, exist_ok=True)

    # Save the splits to CSV files
    train_df.to_csv(os.path.join(Config.DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(Config.DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(Config.DATA_DIR, "test.csv"), index=False)

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Data saved to {Config.DATA_DIR}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    download_alternative_dataset()