# Assignment1 for DSA4213
# A0329409A Zhang Jingxuan
# Glove

# Imports & Global Settings
import json
import os
import random
import re
import time
from collections import Counter
import sys

import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from gensim.models import KeyedVectors
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

plt.ioff()

# Redirect print output
class Tee(object):
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = Tee(sys.stdout, open("Outputs_Glove.txt", "w", encoding="utf-8"))

# Save all plots
save_dir = "plots/Glove"
os.makedirs(save_dir, exist_ok=True)

_plot_count = 0
def autosave_show(*args, **kwargs):
    global _plot_count
    for num in list(plt.get_fignums()):
        fig = plt.figure(num)
        _plot_count += 1
        fname = os.path.join(save_dir, f"plot_{_plot_count:03d}.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"[Plot saved to {fname}]")
        plt.close(fig)

plt.show = autosave_show

# Data Preprocessing
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
def build_vocab(sentences, min_count=2):
    freq = Counter(w for s in sentences for w in s)
    words = [w for w,c in freq.items() if c >= min_count]
    words.sort(key=lambda w: (-freq[w], w))
    w2i = {w:i for i,w in enumerate(words)}
    return words, w2i, freq

def build_cooccurrence(sentences, w2i, window=10):
    from collections import defaultdict
    cooc = defaultdict(float)
    for s in sentences:
        idxs = [w2i[w] for w in s if w in w2i]
        L = len(idxs)
        for pos, i in enumerate(idxs):
            w = random.randint(1, window)
            start = max(0, pos - w)
            end   = min(L, pos + w + 1)
            for pos_c in range(start, end):
                if pos_c == pos: continue
                j = idxs[pos_c]
                dist = abs(pos - pos_c)
                cooc[(i,j)] += 1.0 / dist
    return [(i, j, X) for (i,j), X in cooc.items()]

class GloVeModel:
    def __init__(self, vector_size, vocab_words, seed=4213):
        self.vector_size = vector_size
        self.vocab_words = vocab_words
        self.V = len(vocab_words)
        rng = np.random.RandomState(seed)

        self.W  = (rng.rand(self.V, vector_size) - 0.5) / vector_size
        self.C  = (rng.rand(self.V, vector_size) - 0.5) / vector_size
        self.bW = np.zeros(self.V, dtype=np.float32)
        self.bC = np.zeros(self.V, dtype=np.float32)

        self.gW  = np.ones_like(self.W)
        self.gC  = np.ones_like(self.C)
        self.gbW = np.ones_like(self.bW)
        self.gbC = np.ones_like(self.bC)
        self._latest_train_loss = 0.0
        self.wv = None

    def get_latest_training_loss(self):
        return float(self._latest_train_loss)

    def save(self, path):
        if self.wv is None:
            raise RuntimeError("Call save after training & building KeyedVectors.")
        self.wv.save(path)

def glove_train(sentences, vector_size=300, window=10, min_count=2,
                epochs=15, x_max=100.0, alpha=0.75, lr=0.025, seed=4213, report_every=100000):
    words, w2i, freq = build_vocab(sentences, min_count=min_count)
    print(f"[GloVe] Vocab size after min_count={min_count}: {len(words):,}")
    entries = build_cooccurrence(sentences, w2i, window=window)
    print(f"[GloVe] Co-occurrence entries: {len(entries):,}")

    model = GloVeModel(vector_size, words, seed=seed)

    rng = np.random.RandomState(seed)
    for ep in range(1, epochs+1):
        t_ep = time.time()
        rng.shuffle(entries)
        total_loss = 0.0

        lr_epoch = max(0.0001, lr * (1 - ep / epochs))

        for step, (i, j, Xij) in enumerate(entries, 1):
            if Xij < x_max:
                w_ij = (Xij / x_max) ** alpha
            else:
                w_ij = 1.0

            dot = np.dot(model.W[i], model.C[j])
            pred = dot + model.bW[i] + model.bC[j]
            logX = np.log(max(Xij, 1e-10))
            diff = pred - logX
            loss = w_ij * (diff ** 2)
            total_loss += loss


            grad = 2.0 * w_ij * diff

            grad_Wi = grad * model.C[j]
            grad_Cj = grad * model.W[i]

            model.gW[i] += grad_Wi * grad_Wi
            model.gC[j] += grad_Cj * grad_Cj

            model.W[i]  -= (lr_epoch / np.sqrt(model.gW[i])) * grad_Wi
            model.C[j]  -= (lr_epoch / np.sqrt(model.gC[j])) * grad_Cj

            model.gbW[i] += grad * grad
            model.gbC[j] += grad * grad
            model.bW[i]  -= (lr_epoch / np.sqrt(model.gbW[i])) * grad
            model.bC[j]  -= (lr_epoch / np.sqrt(model.gbC[j])) * grad

            if report_every and (step % report_every == 0):
                avg = total_loss / report_every
                print(f"[GloVe][epoch {ep}] step={step:,}/{len(entries):,}  avg_loss≈{avg:.4f}")
                total_loss = 0.0

        print(f"[GloVe] epoch {ep}/{epochs} done in {time.time()-t_ep:.2f}s")

    model._latest_train_loss = float(total_loss)

    E = model.W + model.C
    kv = KeyedVectors(vector_size=vector_size)
    kv.add_vectors(words, E)

    model.wv = kv
    return model


glove_cfg = dict(
    vector_size=300,
    window=10,
    min_count=2,
    epochs=15,
    x_max=100.0,
    alpha=0.75,
    lr=0.025,
    seed=SEED
)

print("[Train] GloVe config:", json.dumps({k:v for k,v in glove_cfg.items()}, indent=2))
t1 = time.time()
model = glove_train(sentences, **glove_cfg)
loss_per_token = model.get_latest_training_loss() / max(1, sum(len(s) for s in sentences))
print(f"[Train] Done in {time.time()-t1:.2f}s; final loss (relative, last-chunk): {model.get_latest_training_loss():.2f}; Average loss per token: {loss_per_token}")

os.makedirs("artifacts", exist_ok=True)
model.save("artifacts/glove_wikitext2.kv")
model.wv.save_word2vec_format("artifacts/glove_wikitext2.txt")

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

def plot_2d(X2d, words, labels, title="Word Embeddings 2D Plot", show_words=True):
    plt.figure(figsize=(9,7))
    uniq_labels = sorted(set(labels))
    for lab in uniq_labels:
        idx = [i for i,l in enumerate(labels) if l==lab]
        plt.scatter(X2d[idx,0], X2d[idx,1], s=35, label=lab, alpha=0.7)
        if show_words:
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
plot_2d(XYp, wordsM, labelsM, "GLOVE — PCA (Manual Buckets)", True)

# t-SNE
XYt = TSNE(n_components=2, perplexity=20, learning_rate=200, max_iter=2000, random_state=SEED, init="pca").fit_transform(XM)
plot_2d(XYt, wordsM, labelsM, "GLOVE — t-SNE (Manual Buckets)", True)

# UMAP
reducer = umap.UMAP(
    n_components=2,
    random_state=SEED,
    n_neighbors=15,
    min_dist=0.1,
    n_jobs=1
)
XYu = reducer.fit_transform(XM)
plot_2d(XYu, wordsM, labelsM, "GLOVE — UMAP (Manual Buckets)", True)

# Seed-Expanded Buckets
# PCA
auto_buckets = seed_expand_buckets(model, SEED_BUCKETS, per_seed=25)
wordsA, labelsA, XA = collect_bucket_vectors(model, auto_buckets)
XYa = PCA(n_components=2, random_state=SEED).fit_transform(XA)
plot_2d(XYa, wordsA, labelsA, "GLOVE — PCA (Seed-Expanded Buckets)", False)

# t-SNE
XYtA = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1500,
            random_state=SEED, init="pca").fit_transform(XA)
plot_2d(XYtA, wordsA, labelsA, "GLOVE — t-SNE (Seed-Expanded Buckets)", False)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=25, min_dist=0.1, n_jobs=1)
XYuA = reducer.fit_transform(XA)
plot_2d(XYuA, wordsA, labelsA, "GLOVE — UMAP (Seed-Expanded Buckets)", False)


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

# 7. Bonus-3: 20 Newsgroups 文本聚类/分类 (doc embedding = TF-IDF 加权均值)]
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