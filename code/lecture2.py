#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:07:25 2025

@author: doudou
"""

import nltk
from nltk.corpus import brown
from nltk.util import ngrams
from collections import Counter, defaultdict
import random

# Step 1: Download the Brown Corpus
nltk.download('brown')
nltk.download('punkt')

# Step 2: Load and Preprocess Data
# Use a subset of categories (e.g., 'news') from the Brown Corpus
sentences = brown.sents(categories='news')
print(f"Number of sentences: {len(sentences)}")

# Tokenize and flatten the corpus
tokens = [word.lower() for sentence in sentences for word in sentence]

# Step 3: Generate Trigrams
trigrams = list(ngrams(tokens, 3))

# Step 4: Count Frequencies and Compute Probabilities
# Trigram and bigram frequencies
trigram_counts = Counter(trigrams)
bigrams = list(ngrams(tokens, 2))
bigram_counts = Counter(bigrams)

# Calculate conditional probabilities
trigram_probabilities = defaultdict(lambda: defaultdict(float))
for (w1, w2, w3), count in trigram_counts.items():
    trigram_probabilities[(w1, w2)][w3] = count / bigram_counts[(w1, w2)]

# Step 5: Generate Text Using the Trigram Model
def generate_text(trigram_probabilities, start_words, length=15):
    w1, w2 = start_words
    generated = [w1, w2]
    for _ in range(length):
        next_word_candidates = trigram_probabilities[(w1, w2)]
        if not next_word_candidates:
            break
        w3 = random.choices(
            list(next_word_candidates.keys()), 
            list(next_word_candidates.values())
        )[0]
        generated.append(w3)
        w1, w2 = w2, w3
    return ' '.join(generated)

# Generate a sentence starting with "the government"
start_words = ('the', 'government')
generated_text = generate_text(trigram_probabilities, start_words)
print("Generated Text:", generated_text)

# Step 6: Compute Sentence Probability
def sentence_probability(sentence, trigram_probabilities):
    tokens = nltk.word_tokenize(sentence.lower())
    trigrams = list(ngrams(tokens, 3))
    prob = 1.0
    for w1, w2, w3 in trigrams:
        prob *= trigram_probabilities[(w1, w2)].get(w3, 1e-6)  # Smoothing for unseen words
    return prob

sentence = "the government announced new policies"
print("Probability of sentence:", sentence_probability(sentence, trigram_probabilities))
