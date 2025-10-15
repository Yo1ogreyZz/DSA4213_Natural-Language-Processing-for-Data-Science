class Config:
    # Dataset configurations
    DATASET_NAME = "financial_phrasebank"
    DATASET_SUBSET = "sentences_allagree"  # Only sentences where all annotators agreed
    DATA_DIR = "./data"
    MAX_LENGTH = 128
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    # Model configurations
    MODEL_NAME = "roberta-base"
    OUTPUT_DIR = "./outputs"
    CACHE_DIR = "./cache"

    # Training configurations - Full fine-tuning
    FULL_BATCH_SIZE = 16
    FULL_LEARNING_RATE = 2e-5
    FULL_EPOCHS = 5
    FULL_WEIGHT_DECAY = 0.01
    FULL_WARMUP_RATIO = 0.1

    # Training configurations - LoRA
    LORA_BATCH_SIZE = 32  # Can use larger batch size with LoRA
    LORA_LEARNING_RATE = 5e-4  # Higher learning rate for LoRA
    LORA_EPOCHS = 5
    LORA_WEIGHT_DECAY = 0.01
    LORA_WARMUP_RATIO = 0.1

    # LoRA specific configurations
    LORA_RANK = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1

    # Evaluation
    METRICS = ["accuracy", "precision", "recall", "f1"]

    # LoRA ablation study parameters
    ABLATION_RANKS = [4, 8, 16, 32]
    ABLATION_ALPHAS = [16, 32, 64]