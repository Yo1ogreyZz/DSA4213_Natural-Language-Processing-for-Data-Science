# Assignment1 for DSA4213
# A0329409A Zhang Jingxuan
# Word2Vec(skip-gram)
# Import packages needed
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datasets import load_dataset
# import fitz #PyMuPDF
import re
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# doc = fitz.open("Speech and Language Processing.pdf.pdf")
# corpus = ""
# for page in doc:
#     corpus += page.get_text()
# Import corpus and simply processing

# Other methods that can be used for small corpora
# from pdfminer.high_level import extract_text
# corpus = extract_text("Speech and Language Processing.pdf")
# -------------------------------------------------------
# import pdfplumber
# corpus = ""
# with pdfplumber.open("Speech and Language Processing.pdf") as pdf:
#     for page in pdf.pages:
#         corpus += page.extract_text() or ""

corpus = load_dataset("wikitext", "wikitext-2-v1")
all_lines = []
for split in ["train", "validation", "test"]:
    all_lines.extend(corpus[split]["text"])

def tokenize_line(line: str):
    tokens = re.findall(r"[a-zA-Z]+", line.lower())
    return [w for w in tokens if w and w not in STOPWORDS]

# text = re.sub(r'[^a-zA-Z.\s]', '', corpus).lower()
# words = [s.split() for s in text.split(".") if s.strip()]
# Filter out the words in the corpus and convert all of them to lowercase

STOPWORDS = set("""
a an the and or of in on to for with without within through at by from into over under
is are was were be been being am do does did doing have has had having this that these those
it its as not no nor so such too very can could would should may might must will shall
we you they he she i me him her them my your our their his hers ours yours theirs et al
a b c d e f g h i j k l m n o p q r s t u v w x y z
""".split())
# filtered_words = [[w for w in sent if w not in STOPWORDS] for sent in words]
# Remove the meaningless prepositions from the vocabulary list

sentences = [tok for tok in (tokenize_line(l) for l in all_lines) if tok]

model = Word2Vec(
    sentences = sentences, # iterable of tokenized sentences
    vector_size = 150, # embedding dimensions
    window = 5, # context window size on each side
    min_count = 2, # ignore words with total frequency < 3
    sg = 1, # using Skip-Gram, 0 is CBOW
    negative = 12, # negative samples per positive
    sample = 1e-3, # subsampling for very frequent words
    workers = 4, # CPU threads
    epochs = 10, # training passes over the corpus
    hs = 0, # 0 is using negative sampling; 1 is using hierarchical softmax
    ns_exponent = 0.75, # negative sampling distribution exponent
    compute_loss = False # monitor loss or not
)
# Training model

print(list(model.wv.index_to_key)[:10]) # The top ten most frequently occurring words
print(len(model.wv.index_to_key)) # Total number of words
print(model.wv.most_similar("language", topn = 10))
# Select "systems" as the main word and view the ten most similar words.

vocab = model.wv.index_to_key
show_vocab = vocab[100:200]
X = np.array([model.wv[w] for w in show_vocab])

pca = PCA(n_components = 2, random_state=4213)
XY = pca.fit_transform(X)
# Use PCA to reduce the dimensions of the selected top 200 words

plt.figure(figsize=(10,8))
plt.scatter(XY[:, 0], XY[:, 1], s = 20)
for i, w in enumerate(show_vocab):
    plt.annotate(w, xy=(XY[i, 0], XY[i, 1]), fontsize=9, alpha=0.8)
plt.title("Word2Vec (Skip-gram) — PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
# Use plt for plotting and display