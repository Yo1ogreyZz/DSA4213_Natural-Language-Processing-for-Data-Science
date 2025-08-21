#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 21:59:53 2025

@author: doudou
"""

#Example 1
# Import necessary libraries
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Step 1: Prepare a small dataset
# Each sentence is a list of words (tokenized)
sentences = [
    ["I", "love", "natural", "language", "processing"],
    ["Word2Vec", "is", "a", "popular", "model", "for", "embeddings"],
    ["We", "can", "learn", "word", "relationships"],
    ["Word", "embeddings", "capture", "semantic", "meanings"],
    ["I", "enjoy", "teaching", "Word2Vec", "to", "students"]
]

# Step 2: Train a Word2Vec model
# The min_count parameter ensures words appearing less than 1 time are ignored
model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, sg=1)

# Step 3: Explore the Word2Vec model
print("Vocabulary in the model:")
print(list(model.wv.index_to_key))

# Step 4: Get the vector representation of a word
print("\nVector representation of the word 'Word2Vec':")
print(model.wv['Word2Vec'])

# Step 5: Find most similar words
print("\nWords most similar to 'language':")
print(model.wv.most_similar("language"))


#Example 2 Training Word2Vec on a Simple Online Text Corpus (e.g., Movie Reviews)
import nltk
from nltk.corpus import movie_reviews
from gensim.models import Word2Vec

# Step 1: Download and load the dataset
nltk.download('movie_reviews')
sentences = [list(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]

# Step 2: Train a Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=5, min_count=5, sg=1)

# Step 3: Explore the trained model
print("Vocabulary in the model:")
print(list(model.wv.index_to_key)[:10])  # Display top 10 words in the vocabulary

# Step 4: Check most similar words
print("\nWords most similar to 'good':")
print(model.wv.most_similar("good"))

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Get the embeddings and corresponding words
words = list(model.wv.index_to_key[20:100])  # Limit to 50 words for clarity
embeddings = model.wv[words]

# Use PCA to reduce dimensionality
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot the embeddings
plt.figure(figsize=(12, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], marker='o')

for i, word in enumerate(words):
    plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=10)

plt.title("Word Embeddings from Movie Reviews (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()


from sklearn.manifold import TSNE

# Use t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
reduced_embeddings = tsne.fit_transform(embeddings)

# Plot the embeddings
plt.figure(figsize=(14, 10))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], marker='o')

for i, word in enumerate(words):
    plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8)

plt.title("Word Embeddings from Wikipedia Dataset (t-SNE)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()


#Example 3: Training Word2Vec on Wikipedia Articles
import gensim.downloader as api

# Step 1: Load a pre-tokenized Wikipedia dataset
dataset = api.load("text8")  # A subset of cleaned Wikipedia data
print("Dataset loaded.")

# Step 2: Train a Word2Vec model
model = Word2Vec(dataset, vector_size=100, window=5, min_count=10, sg=0)

# Step 3: Explore the trained model
print("Vocabulary size:", len(model.wv.index_to_key))

# Step 4: Find most similar words
print("\nWords most similar to 'king':")
print(model.wv.most_similar("king"))




#Co-occurrence Matrix Method
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

# Example dataset
sentences = [
    ["I", "love", "natural", "language", "processing"],
    ["Word2Vec", "is", "a", "popular", "model", "for", "embeddings"],
    ["We", "can", "learn", "word", "relationships"],
    ["Word", "embeddings", "capture", "semantic", "meanings"],
    ["I", "enjoy", "teaching", "Word2Vec", "to", "students"]
]

# Step 1: Build the vocabulary
vocab = {word: idx for idx, word in enumerate(set(word for sentence in sentences for word in sentence))}
print(vocab)

# Step 2: Build the co-occurrence matrix
vocab_size = len(vocab)
print(vocab_size)

co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

# Step 3: Define the window size
window_size = 2  # Only consider words within 2 positions on either side

# Step 4: Populate the co-occurrence matrix with the window
for sentence in sentences:
    sentence_length = len(sentence)
    for idx, word in enumerate(sentence):
        word_idx = vocab[word]
        
        # Define the context window
        start = max(0, idx - window_size)
        end = min(sentence_length, idx + window_size + 1)
        
        # Update co-occurrence counts for words in the window
        for context_idx in range(start, end):
            if idx != context_idx:  # Skip the word itself
                context_word_idx = vocab[sentence[context_idx]]
                co_matrix[word_idx, context_word_idx] += 1
                
# Step 3: Reduce dimensionality using SVD
svd = TruncatedSVD(n_components=2)
reduced_embeddings = svd.fit_transform(co_matrix)

# Step 4: Visualize
words = list(vocab.keys())
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
plt.title("Co-occurrence Matrix Embeddings (SVD)")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.show()


#PMI-SVD Method
# Step 1: Calculate PMI matrix
co_occurrence_sum = np.sum(co_matrix)
p_word = np.sum(co_matrix, axis=1) / co_occurrence_sum
pmi_matrix = np.zeros_like(co_matrix)

for i in range(vocab_size):
    for j in range(vocab_size):
        if co_matrix[i, j] > 0:
            pmi = np.log((co_matrix[i, j] / co_occurrence_sum) / (p_word[i] * p_word[j]))
            pmi_matrix[i, j] = max(pmi, 0)  # Positive PMI

# Step 2: Apply SVD on the PMI matrix
import numpy as np
U, Sigma, Vt = np.linalg.svd(pmi_matrix, full_matrices=False)

# Extract the top-k components
k = 2  # Number of dimensions
U_k = U[:, :k]                # First k columns of U
Sigma_k = np.diag(Sigma[:k])  # Top k singular values as a diagonal matrix
V_k = Vt[:k, :]               # First k rows of V^T

# Compute U_k Sigma_k^{1/2}
Sigma_k_sqrt = np.sqrt(Sigma_k)
pmi_embeddings = U_k @ Sigma_k_sqrt


# Step 3: Visualize
plt.figure(figsize=(10, 8))
plt.scatter(pmi_embeddings[:, 0], pmi_embeddings[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (pmi_embeddings[i, 0], pmi_embeddings[i, 1]))
plt.title("PMI-SVD Word Embeddings")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.show()
