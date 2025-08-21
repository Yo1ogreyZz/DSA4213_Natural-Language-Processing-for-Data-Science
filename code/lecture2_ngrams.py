#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:19:16 2025

@author: doudou
"""

import nltk
from nltk.corpus import reuters
from nltk.util import ngrams
from collections import Counter, defaultdict
import random

# Step 1: Download and Load the Reuters Corpus
nltk.download('reuters')
#nltk.download('punkt')
nltk.download('punkt_tab')

# Load sentences from the Reuters Corpus
sentences = reuters.sents()
print(f"Number of sentences in Reuters Corpus: {len(sentences)}")

# Flatten sentences and tokenize into lowercase words
tokens = [word.lower() for sentence in sentences for word in sentence]

# Step 2: Generate Trigrams
trigrams = list(ngrams(tokens, 3))

# Step 3: Count Trigrams and Bigrams
trigram_counts = Counter(trigrams)
bigrams = list(ngrams(tokens, 2))
bigram_counts = Counter(bigrams)

# Step 4: Calculate Trigram Probabilities
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

# Example: Generate a sentence starting with "the company"
start_words = ('the', 'company')
generated_text = generate_text(trigram_probabilities, start_words,length=50)
print("Generated Text:", generated_text)

start_words = ('today', 'the')
generated_text = generate_text(trigram_probabilities, start_words)
print("Generated Text:", generated_text)


# Step 6: Compute the Probability of a Sentence
def sentence_probability(sentence, trigram_probabilities):
    tokens = nltk.word_tokenize(sentence.lower())
    trigrams = list(ngrams(tokens, 3))
    prob = 1.0
    for w1, w2, w3 in trigrams:
        prob *= trigram_probabilities[(w1, w2)].get(w3, 1e-6)  # Smoothing for unseen trigrams
    return prob

# Example: Compute the probability of a test sentence
test_sentence = "the company announced strong quarterly profits"
print("Probability of Sentence:", sentence_probability(test_sentence, trigram_probabilities))



"""
Improvements and Extensions
Smoothing:
Implement Kneser-Ney Smoothing or Laplace Smoothing for better handling of unseen trigrams.
Performance Optimization:
Use sparse matrix representations (e.g., scipy.sparse) to handle large co-occurrence matrices efficiently.
"""