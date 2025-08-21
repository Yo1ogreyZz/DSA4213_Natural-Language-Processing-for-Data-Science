#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:46:12 2025

@author: doudou

conda create -n torch_env python=3.10 -y
conda activate torch_env           
pip install torch==2.0.1 torchtext==0.15.2 torchdata==0.6.1
conda install -c conda-forge portalocker
"""
 
 
 
"""
Step 1: Import Libraries
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

# Download necessary NLTK resources
nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')



"""
Step 2: Prepare the Reuters Dataset
Load, tokenize, and preprocess the Reuters dataset.
"""
# Load the Reuters dataset
categories = reuters.categories()
documents = reuters.fileids()
print(f"Number of documents: {len(documents)}")

"""
reuters.categories(): Lists the topic categories available in the Reuters dataset.
reuters.fileids(): Returns the IDs of all documents in the dataset. Each document corresponds to a news article.
"""

# Tokenize and preprocess
def preprocess_reuters(doc_ids):
    data = []
    for doc_id in doc_ids:
        tokens = word_tokenize(reuters.raw(doc_id).lower())
        data.append(tokens)
    return data
"""
Why Tokenize?
Tokenization breaks text into individual words (tokens), which are the basic units the model processes.

Why Lowercase?
Lowercasing ensures that The and the are treated as the same word, simplifying the vocabulary.

reuters.raw(doc_id): Retrieves the raw text of a document.
"""


# Split into training and test sets
train_docs = [doc for doc in documents if doc.startswith('training/')]
test_docs = [doc for doc in documents if doc.startswith('test/')]

# Limit the number of documents for training and testing
# If we use CPU to train the full data, it can take more than one hour
# But you should try the full dataset after the course 
train_docs = [doc for doc in documents if doc.startswith('training/')][:500]  # Use only 1000 training documents
test_docs = [doc for doc in documents if doc.startswith('test/')][:100]       # Use only 100 testing documents

# Preprocess the reduced dataset
train_data = preprocess_reuters(train_docs)
test_data = preprocess_reuters(test_docs)


# Build vocabulary
counter = Counter(token for sentence in train_data for token in sentence)
vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(), start=4)}
vocab.update({"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3})
inv_vocab = {idx: word for word, idx in vocab.items()}
"""
Vocabulary: A mapping of words to unique IDs.

<unk>: Represents unknown words (words not in the training data).
<pad>: Used to pad sequences to the same length for batching.
<bos>: Marks the beginning of a sentence.
<eos>: Marks the end of a sentence.
counter: Counts how many times each word appears in the data.
"""

# Numericalize data
def numericalize(data, vocab):
    numericalized = []
    for sentence in data:
        numericalized.append([vocab.get("<bos>")] + [vocab.get(word, vocab["<unk>"]) for word in sentence] + [vocab.get("<eos>")])
    return numericalized
#Numericalization: Converts words into their corresponding IDs based on the vocabulary

train_data = numericalize(train_data, vocab)
test_data = numericalize(test_data, vocab)

print(f"Vocabulary size: {len(vocab)}")
#Vocabulary size: 12621
#Vocabulary size: 42070


"""
Step 3: Define Dataset and DataLoader
Create a custom PyTorch dataset for batching.
"""
class ReutersDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data    #List of tokenized, numericalized sentences
        self.seq_len = seq_len # Fixed sequence length for training

    def __len__(self):
        return sum(len(sentence) - self.seq_len for sentence in self.data)

    def __getitem__(self, idx):
        for sentence in self.data:
            if idx < len(sentence) - self.seq_len:
                x = sentence[idx:idx + self.seq_len]
                y = sentence[idx + 1:idx + self.seq_len + 1]
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
            idx -= len(sentence) - self.seq_len
        raise IndexError
"""
1. __init__ Method
The __init__ method initializes the dataset.

Parameters:

data: A list of sentences, where each sentence is a list of token IDs (e.g., [2, 15, 8, 39, 3] for <bos>, the, company, announced, <eos>).
seq_len: The fixed length of the input and output sequences.
What It Does:

Stores the data and sequence length as instance variables (self.data, self.seq_len).
2. __len__ Method
The __len__ method defines the total number of samples in the dataset.

What It Computes:

Each sentence can provide len(sentence) - seq_len input-output pairs.
The method sums this value across all sentences.
Why It’s Needed:

PyTorch uses __len__ to determine how many times to call __getitem__ during training.
3. __getitem__ Method
The __getitem__ method retrieves a single sample (input-output pair) from the dataset.

Parameters:
idx: The index of the sample to retrieve.
What It Does:
Iterates Over Sentences:

Loops through the sentences to find the one containing the desired sample index.
Extracts Input and Output Sequences:

The input sequence (x) is sentence[idx:idx + seq_len].
The output sequence (y) is sentence[idx + 1:idx + seq_len + 1].
Returns PyTorch Tensors:

Converts the input (x) and output (y) into tensors with dtype=torch.long.
Why It Subtracts idx:
If the index idx exceeds the current sentence's possible samples, it decrements idx to account for the samples already skipped.
"""        
"""
Why Use a Custom Dataset?
PyTorch expects data to be organized as a subclass of Dataset.
This dataset breaks each sentence into overlapping sequences of length seq_len.
"""


# Hyperparameters
seq_len = 30
batch_size = 32

# Create datasets
train_dataset = ReutersDataset(train_data, seq_len)
test_dataset = ReutersDataset(test_data, seq_len)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
#DataLoader: Automatically batches and shuffles data for training.

"""
Step 4: Build the RNN Model
"""
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embed_size)
        output, hidden = self.rnn(embedded, hidden)  # RNN output and hidden state
        output = self.fc(output)  # Shape: (batch_size, seq_len, vocab_size)
        return output, hidden


"""
Step 5: Train the Model
"""
# Model hyperparameters
vocab_size = len(vocab)
embed_size = 128
hidden_size = 256
num_layers = 2
"""
Embedding Layer: Converts word IDs into dense vectors.
RNN Layer: Processes the sequence of embeddings to capture temporal dependencies.
Fully Connected Layer: Maps RNN outputs to vocabulary size, predicting the next word.
"""

# Initialize model, loss, and optimizer
model = RNNLanguageModel(vocab_size, embed_size, hidden_size, num_layers, pad_idx=vocab["<pad>"])
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=0.001)
"""
CrossEntropyLoss: Measures how far the predicted probabilities are from the true word.
Adam: An optimization algorithm that adjusts learning rates automatically.
"""

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train() #This sets the model to training mode. Some layers, like dropout or batch normalization, behave differently during training and evaluation.
    total_loss = 0 #Initialize the Total Loss
    for x, y in tqdm(train_loader):
        optimizer.zero_grad() #Gradients from the previous batch are accumulated by default in PyTorch. Clearing them ensures gradients are only computed for the current batch.
        output, _ = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward() #Computes gradients for all model parameters based on the loss. These gradients indicate how the model should adjust its weights to reduce the loss.
        optimizer.step() #Updates the model’s weights using the gradients computed in the backward pass.
        total_loss += loss.item() #Extracts the scalar value of the loss (detached from the computational graph) to track the total loss for this epoch.
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

"""
train_loader: Provides batches of data during training.

x: Input sequences (e.g., [2, 15, 8]).
y: Target sequences (e.g., [15, 8, 39]).
tqdm: Displays a progress bar for the loop to monitor the training progress.
Why Use a DataLoader?

It automatically batches data and shuffles it to improve generalization.
"""

"""
output, _ = model(x): 
The input batch x is passed through the model to compute predictions (output).
output is a tensor of shape (batch_size, seq_len, vocab_size):
batch_size: Number of sequences in the batch.
seq_len: Length of each sequence.
vocab_size: Size of the vocabulary (number of possible word predictions).

 loss = criterion(output.view(-1, vocab_size), y.view(-1)): 
output.view(-1, vocab_size):
Reshapes the model's output to (batch_size * seq_len, vocab_size), where each row corresponds to a word's predicted probabilities.
y.view(-1):
Flattens the target tensor to (batch_size * seq_len) so it aligns with the predictions.
criterion:
The loss function (e.g., CrossEntropyLoss) compares the predicted probabilities with the true word IDs (y).

"""
    
"""
100%|██████████| 3924/3924 [03:42<00:00, 17.63it/s]
Epoch 1, Loss: 3.3994
100%|██████████| 3924/3924 [03:38<00:00, 17.92it/s]
Epoch 2, Loss: 1.1176
100%|██████████| 3924/3924 [03:40<00:00, 17.77it/s]
Epoch 3, Loss: 0.6861
100%|██████████| 3924/3924 [03:39<00:00, 17.87it/s]
Epoch 4, Loss: 0.5600
100%|██████████| 3924/3924 [03:42<00:00, 17.62it/s]
Epoch 5, Loss: 0.5004
"""
"""
Step 6: Evaluate the Model
"""
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            #x, y = x.to('cuda'), y.to('cuda')
            output, _ = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)
"""
Evaluation: Computes the average loss on unseen data without updating weights.
"""
test_loss = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}")


"""
Step 7: Generate Text 
"""
def generate_text(model, start_seq, max_len=1000):
    model.eval()
    idx_seq = [vocab.get(word, vocab["<unk>"]) for word in start_seq]
    x = torch.tensor([idx_seq], dtype=torch.long)
    hidden = None
    result = start_seq
    for _ in range(max_len):
        output, hidden = model(x, hidden)
        next_token = output.argmax(dim=-1)[:, -1].item()
        if next_token == vocab["<eos>"]:
            break
        result.append(inv_vocab[next_token])
        x = torch.tensor([[next_token]], dtype=torch.long)
    return " ".join(result)
#Text Generation: Starts with a given sequence and predicts the next word iteratively using the model.
start_sequence = ["the"]
generated_text = generate_text(model, start_sequence,max_len=1000)
print(f"Generated Text: {generated_text}")

start_sequence = ["Now"]
generated_text = generate_text(model, start_sequence,max_len=100)
print(f"Generated Text: {generated_text}")
#Generated Text: Now , with one billion dlrs , a government spokesman said . he said thailand will pay for 25 pct of the cost of the cranes in u.s. dlrs and the rest by sales of rice , textiles and other commodities to yugoslavia . the bangkok shipowners and agents association has appealed to the government to scrap the purchase plan . it said the new facility has allowed it to take full advantage of an early payment discount of about 13 mln dlrs in principal and interest which was negotiated with its banking syndicate in connection with an april . despite