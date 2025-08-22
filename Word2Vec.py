# Assignment1 for DSA4213
# A0329409A Zhang Jingxuan
# Word2Vec (Skip-gram)

# Imports & Global Settings
import json
import os
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from gensim.models import Word2Vec
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score

from datasets import load_dataset

# reproducibility
SEED = 4213
random.seed(SEED)
np.random.seed(SEED)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Data Preprocessing]
STOPWORDS = set("""
a an the and or of in on to for with without within through at by from into over under
is are was were be been being am do does did doing have has had having this that these those
it its as not no nor so such too very can could would should may might must will shall
we you they he she i me him her them my your our their his hers ours yours theirs et al
which but unk also after who during other when all th first one two new old three more while
only most later up about four five between among some out there than before where
a b c d e f g h i j k l m n o p q r s t u v w x y z
""".split())

def tokenize_line(line: str):
    tokens = re.findall(r"[a-zA-Z]+", line.lower())
    return [w for w in tokens if w and w not in STOPWORDS]

def corpus_stats(tokenized_sentences):
    num_sent = len(tokenized_sentences)
    num_tok = sum(len(s) for s in tokenized_sentences)
    vocab = set(w for s in tokenized_sentences for w in s)
    print(f"[Corpus] sentences={num_sent:,}, tokens={num_tok:,}, |V|={len(vocab):,}")

print("[Load] WikiText-2 ...")
t0 = time.time()
corpus = load_dataset("wikitext", "wikitext-2-v1")
all_lines = []
for split in ["train", "validation", "test"]:
    all_lines.extend(corpus[split]["text"])
print(f"[Load] Raw lines: {len(all_lines):,} in {time.time()-t0:.2f}s")

sentences = [tok for tok in (tokenize_line(l) for l in all_lines) if tok]
corpus_stats(sentences)

# Model Training
w2v_cfg = dict(
    sentences=sentences,
    vector_size=300,
    window=10,
    min_count=2,
    sg=1,
    negative=15,
    sample=1e-3,
    workers=6,
    epochs=15,
    hs=0,
    ns_exponent=0.75,
    compute_loss=True
)
print("[Train] SGNS config:", json.dumps({k:v for k,v in w2v_cfg.items() if k!='sentences'}, indent=2))
t1 = time.time()
model = Word2Vec(**w2v_cfg)
loss_per_token = model.get_latest_training_loss() / sum(len(s) for s in sentences)
print(f"[Train] Done in {time.time()-t1:.2f}s; final loss (relative): {model.get_latest_training_loss():.2f}; Average loss per token: {loss_per_token}")

os.makedirs("artifacts", exist_ok=True)
model.save("artifacts/w2v_sgns_wikitext2.model")
model.wv.save_word2vec_format("artifacts/w2v_sgns_wikitext2.txt")

print("[Vocab] Top-10:", list(model.wv.index_to_key)[:10])
print("[Vocab] Size:", len(model.wv.index_to_key))

# 3. Visualization — PCA / t-SNE / UMAP
BUCKETS_MANUAL = {
    "cities": ["tokyo","berlin","london","beijing","shanghai","athens","frankfurt","lausanne"],
    "academia": ["mathematics","physics","chemistry","biology","sociology","stanford","thesis","doctoral"],
    "royalty": ["king","queen","duke","duchess","prince","princess","emperor","empress"],
    "baseball": ["yankees","twins","rangers","era","saves","all","star","closer"],
    "vg_music": ["valkyria","chronicles","sega","soundtrack","orchestra","theme","opening","production"]
}

SEED_BUCKETS = {
    "cities": ["tokyo","berlin","london","beijing"],
    "academia": ["mathematics","physics","thesis","doctoral"],
    "royalty": ["king","queen","duke","duchess"],
    "baseball": ["yankees","twins","rangers","era"],
    "vg_music": ["valkyria","sega","orchestra","theme"]
}

def collect_bucket_vectors(model, buckets):
    words, labels = [], []
    for bucket_name, word_list in buckets.items():
        for w in word_list:
            if w in model.wv:
                words.append(w)
                labels.append(bucket_name)
    X = np.stack([model.wv[w] for w in words], axis=0)
    return words, labels, X

def seed_expand_buckets(model, seed_dict, per_seed=40, exclude_overlap=True):
    buckets = {}
    used = set()
    for name, seeds in seed_dict.items():
        cand = []
        for s in seeds:
            if s in model.wv:
                cand.append(s)
                for nb, _ in model.wv.most_similar(s, topn=per_seed):
                    cand.append(nb)
        uniq = []
        for w in cand:
            if (not exclude_overlap) or (w not in used):
                if w in model.wv and w not in uniq:
                    uniq.append(w)
        for w in uniq:
            used.add(w)
        buckets[name] = uniq
    return buckets

def plot_2d(X2d, words, labels, title="Word Embeddings 2D Plot"):
    plt.figure(figsize=(9,7))
    uniq_labels = sorted(set(labels))
    for lab in uniq_labels:
        idx = [i for i,l in enumerate(labels) if l==lab]
        plt.scatter(X2d[idx,0], X2d[idx,1], s=35, label=lab, alpha=0.7)
        for i in idx:
            plt.annotate(words[i], (X2d[i,0], X2d[i,1]), fontsize=9, alpha=0.85)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Manual Buckets
wordsM, labelsM, XM = collect_bucket_vectors(model, BUCKETS_MANUAL)

# PCA
XYp = PCA(n_components=2, random_state=SEED).fit_transform(XM)
plot_2d(XYp, wordsM, labelsM, "SGNS — PCA (Manual Buckets)")

# t-SNE
XYt = TSNE(n_components=2, perplexity=20, learning_rate=200, max_iter=2000, random_state=SEED, init="pca").fit_transform(XM)
plot_2d(XYt, wordsM, labelsM, "SGNS — t-SNE (Manual Buckets)")

# UMAP
reducer = umap.UMAP(
    n_components=2,
    random_state=SEED,
    n_neighbors=15,
    min_dist=0.1,
    n_jobs=1
)
XYu = reducer.fit_transform(XM)
plot_2d(XYu, wordsM, labelsM, "SGNS — UMAP (Manual Buckets)")

# Seed-Expanded Buckets
# PCA
auto_buckets = seed_expand_buckets(model, SEED_BUCKETS, per_seed=25)
wordsA, labelsA, XA = collect_bucket_vectors(model, auto_buckets)
XYa = PCA(n_components=2, random_state=SEED).fit_transform(XA)
plot_2d(XYa, wordsA, labelsA, "SGNS — PCA (Seed-Expanded Buckets)")

# t-SNE
XYtA = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1500,
            random_state=SEED, init="pca").fit_transform(XA)
plot_2d(XYtA, wordsA, labelsA, "SGNS — t-SNE (Seed-Expanded Buckets)")

# UMAP
reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=25, min_dist=0.1, n_jobs=1)
reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=25, min_dist=0.1, n_jobs=1)
XYuA = reducer.fit_transform(XA)
plot_2d(XYuA, wordsA, labelsA, "SGNS — UMAP (Seed-Expanded Buckets)")

# Nearest Neighbors — Multi-class + Frequency-Stratified + Purity Score
def build_freq_rank(model):
    return {w: i for i, w in enumerate(model.wv.index_to_key)}

def bucket_centroids(model, buckets):
    cents = {}
    for name, ws in buckets.items():
        vecs = [model.wv[w] for w in ws if w in model.wv]
        if vecs:
            cents[name] = np.mean(np.stack(vecs, 0), axis=0)
    return cents

def pick_bucket_representatives(model, buckets, per_bucket=6, freq_bins=(0.0, 0.2, 0.6, 1.0)):
    rank = build_freq_rank(model); V = len(model.wv.index_to_key)
    cents = bucket_centroids(model, buckets)
    probe_words, label_map = [], {}
    n_bins = len(freq_bins) - 1
    per_bin = max(1, per_bucket // n_bins)
    for name, ws in buckets.items():
        ws = [w for w in ws if w in model.wv]
        if len(ws) == 0 or name not in cents: continue
        c = cents[name]
        dists = {w: np.linalg.norm(model.wv[w] - c) for w in ws}
        bins_words = [[] for _ in range(n_bins)]
        for w in ws:
            r = rank[w] / V
            for bi in range(n_bins):
                if freq_bins[bi] <= r < freq_bins[bi+1]:
                    bins_words[bi].append(w); break
        chosen = []
        for bi in range(n_bins):
            cand_sorted = sorted(bins_words[bi], key=lambda w: dists[w])
            chosen.extend(cand_sorted[:per_bin])
        if len(chosen) < per_bucket:
            rest = sorted(ws, key=lambda w: dists[w])
            for w in rest:
                if w not in chosen:
                    chosen.append(w)
                if len(chosen) >= per_bucket: break
        chosen = chosen[:per_bucket]
        probe_words.extend(chosen)
        for w in chosen: label_map[w] = name
    return probe_words, label_map

def show_neighbors(model, probe_words, topn=10):
    report = {}
    for w in probe_words:
        if w in model.wv:
            report[w] = model.wv.most_similar(w, topn=topn)
        else:
            report[w] = []
    for w, nbs in list(report.items())[:20]:
        print(f"\n[{w}]")
        for nb, sim in nbs:
            print(f"  {nb:<20} {sim:.3f}")
    return report

def neighbor_purity(model, probe_words, label_map, topn=10):
    per_word = {}
    for w in probe_words:
        if w not in model.wv: continue
        my_lab = label_map.get(w, None)
        if my_lab is None: continue
        nbs = model.wv.most_similar(w, topn=topn)
        same = sum(1 for nb,_ in nbs if label_map.get(nb, None) == my_lab)
        per_word[w] = same / topn
    avg = (sum(per_word.values()) / len(per_word)) if per_word else 0.0
    return avg, per_word

probe_words, label_map = pick_bucket_representatives(
    model, BUCKETS_MANUAL, per_bucket=6, freq_bins=(0.0, 0.2, 0.6, 1.0)
)
print(f"[Neighbors] Probes: {len(probe_words)} over {len(set(label_map.values()))} buckets.")
_ = show_neighbors(model, probe_words, topn=10)
avg_purity, per_word_purity = neighbor_purity(model, probe_words, label_map, topn=10)
print(f"[Neighbors] Category-aware purity@10 = {avg_purity:.3f}")

# 5. Bonus-1: Word Similarity (WordSim-353 / SimLex-999)
def eval_word_similarity(model, path, sep=",", has_header=True, w1_col=0, w2_col=1, score_col=2):
    gold, pred = [], []
    import csv
    with open(path, newline='', encoding="utf8") as f:
        rdr = csv.reader(f, delimiter=sep)
        if has_header: next(rdr, None)
        for row in rdr:
            w1, w2, s = row[w1_col].strip().lower(), row[w2_col].strip().lower(), float(row[score_col])
            if w1 in model.wv and w2 in model.wv:
                gold.append(s); pred.append(model.wv.similarity(w1, w2))
    if not gold:
        print("[WordSim] No overlapping pairs (OOV too many?)")
        return None
    sp = spearmanr(gold, pred).correlation
    pe = pearsonr(gold, pred)[0] if isinstance(pearsonr(gold, pred), tuple) else pearsonr(gold, pred).statistic
    print(f"[WordSim] n={len(gold)} | Spearman={sp:.3f} | Pearson={pe:.3f}")
    return dict(n=len(gold), spearman=float(sp), pearson=float(pe))

eval_word_similarity(model,
    "datasets/wordsim353/combined.csv",
    sep=",", has_header=True, w1_col=0, w2_col=1, score_col=2)
eval_word_similarity(model,
    "datasets/SimLex-999/SimLex-999.txt",
    sep="\t", has_header=True, w1_col=0, w2_col=1, score_col=3)

# 6. Bonus-2: Analogy (3CosAdd / 3CosMul)
def analogy_3cosadd(model, a, a_star, b, topn=3, exclude=None):
    try:
        res = model.wv.most_similar(positive=[b, a_star], negative=[a], topn=topn+5)
    except KeyError:
        return []
    out = []
    for w, sim in res:
        if w not in {a, a_star, b} and (exclude is None or w not in exclude):
            out.append((w,sim))
            if len(out) >= topn: break
    return out

def analogy_3cosmul(model, a, a_star, b, vocab_limit=None, topn=3, exclude=None):
    try:
        va = model.wv[a]; va_ = model.wv[a_star]; vb = model.wv[b]
    except KeyError:
        return []
    words = model.wv.index_to_key if vocab_limit is None else model.wv.index_to_key[:vocab_limit]
    sims = []
    na = np.linalg.norm(va); na_ = np.linalg.norm(va_); nb = np.linalg.norm(vb)
    for w in words:
        if w in {a, a_star, b} or (exclude is not None and w in exclude): continue
        vw = model.wv[w]; nw = np.linalg.norm(vw) + 1e-9
        cos1 = (np.dot(vw, va_)/(nw*na_ + 1e-9))
        cos2 = (np.dot(vw, vb )/(nw*nb  + 1e-9))
        cos3 = (np.dot(vw, va )/(nw*na  + 1e-9))
        score = (cos1 * cos2) / (cos3 + 1e-9)
        sims.append((w, score))
    sims.sort(key=lambda x: -x[1])
    return sims[:topn]

def eval_analogies(model, quadruples, method="3cosadd", topn=1):
    correct = total = 0
    for a, a_star, b, b_star in quadruples:
        pred = analogy_3cosadd(model, a, a_star, b, topn) if method=="3cosadd" else analogy_3cosmul(model, a, a_star, b, topn=topn)
        if not pred: continue
        total += 1
        if b_star in [w for w,_ in pred]: correct += 1
    acc = correct/total if total else 0.0
    print(f"[Analogy] {method} Acc@{topn} = {acc:.3f} (n={total})")

quadruples_demo = [
    ("king","queen","duke","duchess"),
    ("tokyo","yokohama","berlin","frankfurt"),
    ("physics","chemistry","biology","mathematics"),
    ("closer","saves","starter","innings")
]
eval_analogies(model, quadruples_demo, method="3cosadd", topn=1)
eval_analogies(model, quadruples_demo, method="3cosmul", topn=1)

# 7. Bonus
def simple_tokenize(s):
    return re.findall(r"[a-zA-Z]+", s.lower())

def doc_embed_average(model, docs, tfidf=None):
    if tfidf is None:
        vecs = []
        for doc in docs:
            toks = simple_tokenize(doc)
            arr = [model.wv[w] for w in toks if w in model.wv]
            vecs.append(np.mean(arr, axis=0) if arr else np.zeros(model.vector_size))
        return np.stack(vecs, 0)
    else:
        X = tfidf.transform(docs)
        vocab = tfidf.vocabulary_; inv_vocab = {i:t for t,i in vocab.items()}
        vecs = []
        for i in range(X.shape[0]):
            row = X.getrow(i); idxs = row.indices; data = row.data
            parts, weights = [], []
            for j, w in zip(idxs, data):
                token = inv_vocab[j]
                if token in model.wv:
                    parts.append(model.wv[token]*w); weights.append(w)
            v = (np.sum(parts, 0) / (np.sum(weights)+1e-9)) if parts else np.zeros(model.vector_size)
            vecs.append(v)
        return np.stack(vecs, 0)

def eval_20ng_small(model):
    try:
        cats = ["comp.graphics","sci.space","rec.sport.baseball","talk.politics.mideast"]
        train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers","footers","quotes"))
        test  = fetch_20newsgroups(subset="test",  categories=cats, remove=("headers","footers","quotes"))
    except Exception as e:
        print(f"[20NG] Could not fetch dataset (offline?): {e}")
        return
    tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z]+", lowercase=True, min_df=3)
    tfidf.fit(train.data)
    Xtr = doc_embed_average(model, train.data, tfidf=tfidf)
    Xte = doc_embed_average(model, test.data,  tfidf=tfidf)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Xtr, train.target)
    pred = clf.predict(Xte)
    acc = accuracy_score(test.target, pred)
    f1  = f1_score(test.target, pred, average="macro")
    print(f"[20NG-CLS] Acc={acc:.3f}, Macro-F1={f1:.3f}")

    X_all = np.vstack([Xtr, Xte])
    y_all = np.concatenate([train.target, test.target])
    kmeans = KMeans(n_clusters=len(cats), random_state=SEED, n_init=10)
    y_pred = kmeans.fit_predict(X_all)
    ari = adjusted_rand_score(y_all, y_pred)
    nmi = normalized_mutual_info_score(y_all, y_pred)
    print(f"[20NG-CLU] ARI={ari:.3f}, NMI={nmi:.3f}")

# 20NG
eval_20ng_small(model)