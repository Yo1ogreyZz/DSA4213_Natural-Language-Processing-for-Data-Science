#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:15:58 2025

@author: doudou
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
import numpy as np
import nltk


# Load the text of 红楼梦
with open("红楼梦.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Preview the first 500 characters
print(text[:500])

# Tokenize the text at the character level
chars = list(text)  # Each character is treated as a token

# Build vocabulary
vocab = sorted(set(chars))  # Unique characters
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

print(f"Vocabulary size: {len(vocab)}")

# Convert characters to indices
encoded_text = [char_to_idx[char] for char in chars]

# Convert to PyTorch tensor
data = torch.tensor(encoded_text, dtype=torch.long)


# Define the RNN-based language model
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)  # Embedding layer
        output, hidden = self.rnn(x, hidden)  # RNN layer
        output = self.fc(output)  # Fully connected layer
        return output, hidden

# Generate batches of data
def get_batch(data, seq_len, batch_size):
    for i in range(0, len(data) - seq_len, seq_len):
        x = data[i:i+seq_len]
        y = data[i+1:i+seq_len+1]
        yield x.view(-1, seq_len), y.view(-1, seq_len)

# Hyperparameters
vocab_size = len(vocab)
embed_size = 128
hidden_size = 256
num_layers = 2
seq_len = 100
batch_size = 32
num_epochs = 5
learning_rate = 0.001

# Model, loss, and optimizer
model = RNNLanguageModel(vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with progress bar
for epoch in range(num_epochs):
    hidden = None  # Initialize hidden state
    total_loss = 0
    progress_bar = tqdm(get_batch(data, seq_len, batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for x_batch, y_batch in progress_bar:
        optimizer.zero_grad()
        output, hidden = model(x_batch, hidden)
        hidden = hidden.detach()  # Detach hidden state for the next batch
        loss = criterion(output.view(-1, vocab_size), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Update the progress bar with the current loss
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss:.4f}")



# Generate text
def generate_text(model, start_text, max_len=200):
    model.eval()
    # Convert starting text to indices and create input tensor
    input_ids = torch.tensor([char_to_idx[char] for char in start_text], dtype=torch.long).unsqueeze(0)  # Shape: (1, len(start_text))
    generated_text = list(start_text)  # Store the generated characters
    hidden = None  # Initialize hidden state

    for _ in range(max_len):
        output, hidden = model(input_ids, hidden)  # Forward pass
        next_char_id = output[:, -1, :].argmax(dim=-1).item()  # Take the last step's output
        next_char = idx_to_char[next_char_id]  # Convert to character
        generated_text.append(next_char)
        input_ids = torch.tensor([[next_char_id]], dtype=torch.long)  # Update input with the predicted char
        #if next_char == "。":  # Stop generating at a full stop
            #break

    return ''.join(generated_text)

start_text = ["贾", "宝", "玉", "和"]
print("Generated Text:", generate_text(model, start_text))
start_text = ["林", "黛", "玉", "不"]
print("Generated Text:", generate_text(model, start_text))

