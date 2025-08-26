# Report——Word embedding training

> Name: Zhang Jingxuan   Number: A0326409A

## 1. Introduction

This task compare three classic word‑embedding methods under aligned settings on WikiText‑2: **Word2Vec Skip‑gram with Negative Sampling (SGNS)**, **SPPMI‑SVD** (explicit factorization), and **GloVe** (global weighted least squares).

------

## 2. Data & Preprocessing

- **Corpus**：WikiText‑2
- **Stats**：`sentences=28,369`, `tokens=1,094,036`, `|V|=27,091`
- **Pipeline**：Regex （`[A-Za-z]+`）→ Lowercase → Delete STOPWORDS；`min_count=2` Align word list
- **Effect**：Remove noise, unify the vocabulary, and ensure fair comparison among the three models
- **Sliding window co‑occurrence**： `W=10`，if `1 ≤ |pos(w)−pos(c)| ≤ W` then it is counted as a co-occurrence

------

## 3. Models & Parameters

## 3. Model and parameters

### 3.1 SGNS (Skip‑gram with Negative Sampling)

**Config**
 `vector_size=300`, `window=10`, `min_count=2`, `sg=1`, `negative=15`, `sample=1e-3`,
 `workers=6`, `epochs=15`, `hs=0`, `ns_exponent=0.75`, `compute_loss=True`.

**The Basis and Function of the Setup**

- 300 dimension: Classic equilibrium point; Window 10: Suitable for encyclopedic style (more semantic/theme-oriented).
- `negative = 15`: Stronger contrast signal; `ns_exponent = 0.75`: Optimal negative sampling distribution based on experience;
- `sample=1e-3`: High-frequency downsampling, reducing the dominance of stop words.

**Loss**（For the positive sample (w, c) and the K negative samples $c'_k$）：
$$
\mathcal{L}_{\text{SGNS}}(w,c)= -\log \sigma(\mathbf{u}_w^\top\mathbf{v}_c)
-\sum_{k=1}^{K}\log \sigma\!\left(-\mathbf{u}_w^\top\mathbf{v}_{c'_k}\right),\quad 
\sigma(x)=\frac{1}{1+e^{-x}}.
$$
**Negative Sampling Distribution**
$$
P_n(c)\propto f(c)^{\alpha},\ \alpha=0.75.
$$
**High-frequency downsamplin**
$$
P(\text{discard } w)=1-\sqrt{\frac{t}{f(w)}},\ \ t=10^{-3}.
$$

------

### 3.2 SPPMI‑SVD (Shifted Positive PMI + Truncated SVD)

**Config**
 `vector_size=300`, `window=10`, `min_count=2`, `subsample_t=1e-3`,
 `shift_k=15.0`, `svd_power=0.5`, `seed=4213`.

**The Basis and Function of the Setup**

- Align the window/dimension/word list with SGNS.
- Setting `shift_k = 15` aligns the $\text{SPPMI} = \max(\text{PMI} - \log K, 0)$ with the negative sampling approximation of SGNS.
- `svd_power = 0.5`: Singular value weighting, reducing the dominance of head components.

**Co‑occurrence & PMI**
$$
P(w,c)=\frac{X_{wc}}{X_{**}},\quad P(w)=\frac{X_{w*}}{X_{**}},\quad P(c)=\frac{X_{*c}}{X_{**}}.
$$
**Truncated SVD & reweighting**
$$
M=\text{SPPMI}\approx U_d\Sigma_d V_d^\top,\quad 
E_w=U_d\Sigma_d^{p},\ E_c=V_d\Sigma_d^{p},\ p=\texttt{svd\_power}=0.5.
$$

------

### 3.3 GloVe (Global Vectors)

**Config**
 `vector_size=300`, `window=10`, `min_count=2`, `epochs=15`,
 `x_max=100.0`, `alpha=0.75`, `lr=0.025`, `seed=4213`.

**The Basis and Function of the Setup**

- Align the dimensions / windows / rounds with other models.
- $x_{\max}, \alpha$ are empirical values in the paper: used to balance the weights of low-frequency and high-frequency; `lr = 0.025` adopts AdaGrad style learning rate decay.

**Weighted least squares**
$$
J=\sum_{i,j} f(X_{ij})\left(\mathbf{w}_i^{\top}\tilde{\mathbf{w}}_j+b_i+\tilde b_j-\log X_{ij}\right)^2,
\quad
f(x)=\begin{cases}
(\frac{x}{x_{\max}})^{\alpha} & x<x_{\max}\\
1 & x\ge x_{\max}
\end{cases}.
$$
**Gradient (used for AdaGrad update)**（Set  $e_{ij}=\mathbf{w}_i^\top\tilde{\mathbf{w}}_j+b_i+\tilde b_j-\log X_{ij}$）：
$$
\frac{\partial J}{\partial \mathbf{w}_i}= \sum_j 2f(X_{ij})\,e_{ij}\,\tilde{\mathbf{w}}_j,\quad
\frac{\partial J}{\partial \tilde{\mathbf{w}}_j}= \sum_i 2f(X_{ij})\,e_{ij}\,\mathbf{w}_i,
$$
After training, $\mathbf{w}_i+\tilde{\mathbf{w}}_i$ is used as the word vector.

------

## 4. Evaluation & Visualization Math

- **Cosine**：$\cos(\mathbf{a},\mathbf{b})=\dfrac{\mathbf{a}^\top\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$.
- **Analogy**（3CosAdd）：$\arg\max_{w}\cos\big(\mathbf{w},\,\mathbf{b}-\mathbf{a}+\mathbf{c}\big)$；（3CosMul Similarly）.
- **PCA**：Project onto the first two principal components of the covariance matrix.
- **t-SNE**: Minimize $\mathrm{KL}(P\|Q)$; **UMAP**: Minimize cross-entropy.
- **TF-IDF + Logistic Regression**：$\mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\log\frac{N}{1+\mathrm{df}(t)}$。
- **K‑Means**：$\min\sum_i\|\mathbf{x}_i-\mu_{z_i}\|^2$。
- **ARI/NMI、Pearson/Spearman**：The standard definition is the same as that in the class/implementation of the code evaluation.

------

## 5. Results

### 5.1 Word Similarity

| Model         | WordSim‑353 (Spearman / Pearson) | SimLex‑999 (Spearman / Pearson) |
| ------------- | -------------------------------- | ------------------------------- |
| **SGNS**      | **0.524 / 0.484**                | **0.239 / 0.225**               |
| **SPPMI‑SVD** | 0.376 / 0.390                    | 0.147 / 0.155                   |
| **GloVe**     | 0.078 / 0.080                    | 0.063 / 0.076                   |

### 5.2 Word Analogy

- Three models：`3CosAdd Acc@1 = 0.000`，`3CosMul Acc@1 = 0.000`（n=4 demo）。

### 5.3 Downstream — Text Classification & Clustering

| Model         | 20NG‑CLS Acc / Macro‑F1 | 20NG‑CLU ARI / NMI |
| ------------- | ----------------------- | ------------------ |
| **SGNS**      | **0.855 / 0.854**       | **0.456 / 0.452**  |
| **SPPMI‑SVD** | 0.832 / 0.832           | 0.176 / 0.241      |
| **GloVe**     | 0.477 / 0.468           | 0.034 / 0.043      |

------

## 6. Fairness

- **The same corpus and word list**：WikiText‑2，`min_count=2`。
- **The same core hyperparameter**：`vector_size=300`，`window=10`，`epochs=15`。
- **Align the "Noise Processing/Weight" strategy**：
  - SGNS：`sample=1e-3`、Negative sampling `negative=15`；
  - SPPMI‑SVD：Downsample before constructing the matrix；`shift_k = 15` aligns with the $\log K$ of SGNS;
  - GloVe: $f(x)$ suppresses high-frequency components while not excessively amplifying low-frequency ones.
- **Same evaluation pipeline**: TF-IDF classification, KMeans clustering, PCA/t-SNE/UMAP visualization, same correlation coefficient implementation.

------

## 7. Three dimensionality reduction visualization methods (PCA, t-SNE, UMAP)

### 7.1 PCA (Principal Component Analysis)

PCA is a **linear dimensionality reduction method**, which projects the data by finding the direction with the maximum variance, thereby maximizing the information retention.
Mathematically, given the centered data matrix $X \in \mathbb{R}^{n \times d}$, PCA solves the eigenvalue problem:
$$
C = \frac{1}{n} X^\top X
$$
Here, $C$ represents the covariance matrix, $v$ is the eigenvector (principal component direction), and $\lambda$ is the eigenvalue.
Select the $k$ largest eigenvalues and their corresponding eigenvectors to obtain the dimensionality reduction matrix $V_k$, and then perform linear projection:
$$
Z = X V_k
$$
**Effect：**

- Maintain the global variance structure.
- The local structure may be stretched or compressed, and the clusters often take a "elongated/flat" shape.

------

### 7.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE focuses on local similarities. It models the similarity between high-dimensional points as a probability distribution, and then minimizes the difference in the distribution in the low-dimensional space.

- In high dimensions, the similarity between points $i$ and $j$ is defined as a conditional probability:

$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

- symmetrization:

$$
p_{ij} = \frac{p_{i|j} + p_{j|i}}{2n}
$$

- Low-dimensional space uses t-distribution to model similarity:

$$
q_{ij} = \frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k \neq l}(1+\|y_k-y_l\|^2)^{-1}}
$$

- The objective function is to minimize the KL divergence:

$$
\text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

**Effect：**

- Place a strong emphasis on the local neighborhood.
- The clusters will be "spaced out" from each other, but the distance between the clusters should not be over-interpreted.

------

### 7.3 UMAP (Uniform Manifold Approximation and Projection)

UMAP is based on the **"manifold assumption + graph theory"**: it assumes that high-dimensional data is distributed on a low-dimensional manifold. The steps are as follows:

1. Construct the adjacency graph ($k$-NN graph).
2. Estimate the fuzzy simplicial set in high dimensions, and define the local similarity:

$$
p_{ij} = \exp\left(-\frac{d(x_i,x_j) - \rho_i}{\sigma_i}\right)
$$

Here, $\rho_i$ represents the nearest neighbor distance, and $\sigma_i$ controls the scale.

3. Define a similar distribution $q_{ij}$ in the low-dimensional space:

$$
q_{ij} = \frac{1}{1 + a\|y_i-y_j\|^{2b}}
$$

4. By minimizing the cross-entropy loss:

$$
\mathcal{L} = \sum_{i<j}\left[p_{ij}\log\frac{p_{ij}}{q_{ij}} + (1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}\right]
$$

**Effect：**

- While maintaining both local and some global structures.
- It is faster and more stable than t-SNE, and often can reveal the relative relationships between clusters.

------

## 8. The selection of Bucket and the reasons behind it

**1. annual_buckets**

- A fixed baseline set of categories (e.g., cities/academia/royalty/sports) used across models/visualizers to ensure fairness and comparability.

**2. Seed‑Expanded Buckets**

- Start with seed terms, expand by each model’s own top‑N neighbors. This tests whether the model’s similarity space is semantically coherent; bad neighbors ⇒ messy expanded clusters.

------

### 8.1 Make a comparison of three dimensions based on the graph you provided

> Conclusion first: In your batch of graphs, **the clusters of GloVe are the least stable** (all three methods - PCA/t-SNE/UMAP - show clustering, outliers, and "threading" phenomena), **SGNS is second** (overall separable, but with relatively blurred boundaries), and **SPPMI-SVD's clustering is the clearest** (especially in the royalty and cities categories).

#### A. Same model, different clusters (considering the stability of the model for the boundaries between classes)

- **SGNS**：
  - Cities and academia are relatively stable; royalty has clusters but the boundaries are slightly blurry; sports (such as "yankees" and "saves") are clearly in the context of baseball in the seed-expanded graph, but there is still a small amount of overlap with other categories.
- **SPPMI‑SVD**：
  - The "cities/royalty" category was classified the best (both "annual" and "seed-expanded" are clear islands), and the "academia" category was also relatively stable; this indicates that the explicit global statistics combined with SVD resulted in strong intra-class consistency in the space.
- **GloVe**：
  - **Three types of overfitting**: There are a large number of personal names and miscellaneous words in the "cities" category; modern common words are mixed in the "royalty" category; there are many scattered words and obvious outliers in the "sports" category. **This, together with your numerical evaluation, indicates that this GloVe model has not learned reliable semantic neighborhoods. **

#### B. Same model, same cluster, different visualization algorithms (depending on the sensitivity of the algorithm)

- **PCA**: The boundaries of the three models are generally **vague**; SPPMI-SVD still clearly shows distinct subgroups; GloVe in PCA is more like "clouds and fog", lacking clear boundaries.
- **t-SNE**: All three models are more "separated"; the cities/royalty in SPPMI-SVD are the clearest; SGNS is second; although GloVe shows some small clusters, the words within each cluster are diverse and the clusters still intermingle between each other. **
- **UMAP**: Overall, it is the most capable of "illustrating through images". SPPMI-SVD presents a **stable island-like structure**; SGNS has distinct boundaries; **GloVe on UMAP remains a "cloud of confusion", with severe mixing of small clusters locally**.

#### C. Different models, same cluster, same visualization (horizontal knife ratio)

Take "cities" as an example (any visualization consistency): 

- **SPPMI-SVD**: The city name is strongly coupled with city-related/Olympic-related words such as IOC/arena/bid → Forming a compact and clear cluster.
- **SGNS**: There are place names/organization names/event names around the city name, along with a few miscellaneous words → Can be divided but the boundaries are generally not clear. -

- **GloVe**: When a large number of names or common general words appear around the city name, and the similarity is extremely close (0.98 - 0.99) → The cluster is unreliable; seed-expanded directly "distorts" the entire cluster.

------

### 8.2 Similarity (Top-N Nearest Neighbors)

**cities**

- **tokyo**
  - SGNS：`baku(0.587), yokohama(0.528), sportaccord(0.519), nippon(0.506), nexon(0.501)`
  - SPPMI‑SVD：`bid(0.768), venues(0.751), rio(0.747), athens(0.741), bidding(0.733)`
  - **GloVe**：`ben(0.990), motor(0.990), wright(0.990), del(0.990), kevin(0.990)` ← **Person Name / Miscellaneous Items**
- **london**
  - SGNS：`islington(0.457), heidelberg(0.456), piccadilly(0.446), esmond(0.444), cornhill(0.438)`
  - SPPMI‑SVD：`house(0.781), theatre(0.764), travelled(0.761), invited(0.754), attended(0.753)`
  - **GloVe**：*(Negative response)*

**academia**

- **physics**
  - SGNS：`cowan(0.578), physicists(0.553), buggy(0.546), udell(0.536), biology(0.529)`
  - SPPMI‑SVD：`reines(0.794), mathematics(0.756), creutz(0.748), laboratory(0.740), alamos(0.737)`
  - **GloVe**：`santa(0.985), shop(0.985), barbara(0.985), mental(0.985), matt(0.985)` ← **Irrelevant**
- **thesis**
  - SGNS：*(Negative response)*
  - SPPMI‑SVD：`doctoral(0.826), phd(0.824), innis(0.771), economics(0.770), zionism(0.750)`
  - **GloVe**：`del(0.988), ben(0.988), kevin(0.987), dr(0.987), iv(0.987)` ← **Irrelevant / Confusing**

**royalty**

- **duchess**
  - SGNS：`guildford(0.678), paget(0.665), northumberland(0.640), lisle(0.613), dudley(0.586)`
  - SPPMI‑SVD：`dudley(0.880), guildford(0.876), northumberland(0.859), countess(0.817), duke(0.791)`
  - **GloVe**：`pack(0.971), plus(0.970), max(0.970), karl(0.970), von(0.970)` ← **Clearly off-topic**
- **king**
  - SGNS：`overlord(0.513), diarmata(0.511), ragnaill(0.511), ua(0.506), bagrat(0.502)`
  - SPPMI‑SVD：`son(0.784), grandfather(0.755), kings(0.754), grandson(0.744), vassal(0.733)`
  - **GloVe**：*(Negative response)*

**sports（棒球）**

- **yankees**
  - SGNS：`frazee(0.616), braves(0.610), wever(0.598), sox(0.594), pitching(0.590)`
  - SPPMI‑SVD：`sox(0.936), dodgers(0.917), pitcher(0.912), braves(0.909), mlb(0.909)`
  - **GloVe**：`england(0.987), defeat(0.987), defeated(0.986), manager(0.985), conference(0.985)` ← **mismatch between contexts**
- **saves**(Basketball statistics item)
  - SGNS：`batters(0.594), shutout(0.583), strikeouts(0.569), registering(0.565), orioles(0.565)`
  - SPPMI‑SVD：`strikeouts(0.900), mlb(0.889), dodgers(0.889), postseason(0.884), strikeout(0.882)`
  - **GloVe**：`record(0.989), era(0.989), consecutive(0.986), wins(0.986), jordan(0.985)` ← **Integrate with names of people**

> It can be seen that: **The neighbors of GloVe are mostly names/common words, and the similarity is generally between 0.98 and 0.99 (at the 'top level').** This is exactly consistent with the overlapping and outliers shown in the graph - indicating that the vector space of GloVe has **degraded/collapsed** (the similarity distribution is distorted) - thereby "badly affecting" the expanded words generated by seed expansion, and finally presenting fragmented and overlapping clusters on t-SNE/UMAP.

------

### 8.3 Which model is more reliable?

- **Regarding this experiment (your actual picture and log)**:
  - **Most reliable: SPPMI-SVD (Clustering Interpretability)** - In the annual and seed-expanded sets of graphs, **clusters such as 'cities/royalty' have the clearest boundaries**, and the Top-N nearest neighbors closely adhere to the semantic domain (e.g., *tokyo→bid/venues/rio/ioc*; *duchess→countess/duke*).
  - **Robust for the task: SGNS (Downstream/Overall Availability)** - The graph is divisible, and the majority of the nearest neighbors are semantically related (although there is a small amount of noise). Combined with your numerical evaluation (classification/clustering metrics are leading), it indicates **high availability**.
  - **The least reliable this time: GloVe** - In the graph, all three methods show obvious aliasing; the "personification/generalization" of the nearest neighbors is severe and the similarity is at its maximum. This is consistent with the numerical evaluation (WordSim/SimLex, 20NG-CLS/CLU are significantly lagging), indicating that **this training did not converge to a meaningful semantic space**.

> Summary: **Image + Nearest Neighbors + Metrics** are integrated into one: This time your GloVe performance was poor; overall reliability **SPPMI-SVD (interpretability) ≥ SGNS (practicality) > GloVe (this time)**.

------

### 8.4 How to restore GloVe to its normal performance

- Moderately increase the learning rate/iteration (`lr 0.05 - 0.1`, `epochs 25 - 50`), or use a fixed lr + warmup;
- Grid search for parameters `x_max ∈ [50, 200], α ∈ [0.5, 0.9]`, avoiding underestimation of low-frequency components;
- Apply numerical stability to `log X` (`log(max(X, 1e-8))`), perform mean subtraction and L2 normalization after training, and then calculate similarity/visualize;
- Increasing the corpus size (at the level of WikiText-103) can significantly alleviate the sparsity/noise issues in global regression.

------

## 9. Discussion

### 9.1 Parallel Comparison

- **SGNS**：
  - **Advantage**: The word similarity (between the two datasets) and downstream classification/clustering metrics are **significantly superior**.
  - **Disadvantage**: The analogy task still fails when using a small corpus.
- **SPPMI‑SVD**：
  - **Advantages**: The visualization of clusters is clear and the interpretation is strong (frequent co-occurrence classes such as royalty/cities are clearly aggregated).
  - **Disadvantages**: The numerical indicators are generally lagging behind, especially the clustering (ARI/NMI) values are low; SVD truncation only retains a portion of the variance.
- **GloVe**：
  - **This result**: The word similarity, downstream classification and clustering **significantly** fall behind the other two methods.
  - **Possible reasons**:
    1. The corpus is relatively small (~1M tokens), and GloVe, as a global regression model, is sensitive to sparsity in small datasets;
    2. `lr=0.025` + linear decay might be **too small**, and it may not converge adequately (although AdaGrad is used, the upper limit of the learning rate is restricted);
    3. The weighted sum with $x_{\max}=100$ and $\alpha=0.75$ might **over-suppress low-frequency words** under this data distribution;
    4. Although the co-occurrence construction and traversal order are randomly shuffled, the **symmetric/ diagonal processing**, **numerical stabilization terms**, etc., can significantly affect convergence;
    5. The word vector output adopts $W+C$ (correctly), but if **no centering/normalization is performed**, it may affect the cosine similarity and the performance of downstream linear models.

### 9.2 Targeted improvement suggestions (while maintaining a fair comparison basis)

- **Expanding the corpus**: Using WikiText-103 or larger general corpora can significantly enhance the performance of analogy and global statistical methods.
- **Capacity and Training**:
  - SGNS：`epochs`→30，`negative`→20–25；
  - SPPMI‑SVD：The "svd_power" grid (0, 0.25, 0.5, 1.0), or perform small-step SGNS fine-tuning on the SVD vectors;
  - GloVe：
    - Learning rate: `lr` → 0.05 - 0.1 (use AdaGrad if applicable), or use a fixed `lr` with warm-up;
    - Weights: $x_{\max}\in[50,200]$, $\alpha\in[0.5,0.9]$ grid;
    - Numerical stability: `log(max(X, 1e-8))`, evaluate after zero-meaning and L2 normalization of vectors/bias.
- **Evaluation Consistency**: The outputs of the three models are uniformly processed through "mean subtraction + L2 normalization" before conducting the cosine evaluation, which can reduce the influence of scale differences.

------

## 10. Parameter‑by‑Parameter

- `vector_size=300`: Dimension capacity; ↑ Dimension → Expressiveness ↑ but training/inference cost ↑.
- `window=10`: Context range; Large window focuses on theme/semantics, small window focuses on syntax.
- `min_count=2`: Remove extremely low-frequency noise, stabilize statistics.
- `negative=15` (SGNS): Negative sample number; ↑ Improves discrimination but increases time consumption ↑.
- `ns_exponent=0.75`: Negative sampling distribution; Optimal empirical range $[0.5,1]$.
- `sample=1e-3`: Downsample high-frequency; Avoid dominance of stop words.
- `shift_k=15` (SPPMI): Align with `negative` of SGNS ($\log K$ shift).
- `svd_power=0.5`: Re-weight singular values; $p$ smaller leads to more balanced distribution, larger emphasizes principal components.
- `x_max=100, \alpha=0.75` (GloVe): Weight function; `x_max` ↑ Strengthens high-frequency, `α` ↑ Strengthens low-frequency suppression.
- `lr=0.025` (GloVe): This time relatively conservative; may underfit on small data.
- `epochs=15`: Total number of rounds; If aiming for the best performance, it can be appropriately increased.

------

## 11. Conclusion

**SGNS** clearly leads in similarity and downstream tasks on WikiText‑2; **SPPMI‑SVD** offers good interpretability but weaker metrics; **GloVe** underperforms with the current small‑corpus and settings. Practical remedies include larger corpora and targeted tuning (especially for GloVe’s learning rate and weighting).

------

## 12. Appendix

#### 12.1 Training result & plot:

##### SGNS:

```
[Load] WikiText-2 ...
[Load] Raw lines: 44,836 in 11.07s
[Corpus] sentences=28,369, tokens=1,094,036, |V|=27,091
[Train] SGNS config: {
  "vector_size": 300,
  "window": 10,
  "min_count": 2,
  "sg": 1,
  "negative": 15,
  "sample": 0.001,
  "workers": 6,
  "epochs": 15,
  "hs": 0,
  "ns_exponent": 0.75,
  "compute_loss": true
}
[Train] Done in 114.54s; final loss (relative): 53321704.00; Average loss per token: 48.73852779981646
[Vocab] Top-10: ['time', 'game', 'city', 'song', 'year', 'made', 'season', 'war', 'north', 'used']
[Vocab] Size: 27091
[Plot saved to plots/Word2Vec\plot_001.png]
[Plot saved to plots/Word2Vec\plot_002.png]
[Plot saved to plots/Word2Vec\plot_003.png]
[Plot saved to plots/Word2Vec\plot_004.png]
[Plot saved to plots/Word2Vec\plot_005.png]
[Plot saved to plots/Word2Vec\plot_006.png]
[Neighbors] Probes: 30 over 5 buckets.

[london]
  islington            0.457
  heidelberg           0.456
  piccadilly           0.446
  esmond               0.444
  cornhill             0.438
  cramer               0.437
  mayson               0.434
  makropoulos          0.433
  placards             0.427
  cheevers             0.424

[tokyo]
  baku                 0.587
  yokohama             0.528
  sportaccord          0.519
  nippon               0.506
  nexon                0.501
  benz                 0.501
  doha                 0.500
  budokan              0.498
  bella                0.490
  applicant            0.487

[athens]
  sportaccord          0.672
  baku                 0.582
  bella                0.555
  horizonte            0.548
  olympian             0.546
  vaulting             0.542
  beit                 0.540
  culminates           0.536
  chabad               0.530
  sankat               0.530

[shanghai]
  tiebreak             0.525
  benz                 0.524
  compatriot           0.502
  wawrinka             0.499
  taiwanese            0.497
  sponsoring           0.496
  mitsuyo              0.494
  indoors              0.493
  quarterfinals        0.493
  marat                0.491

[lausanne]
  baku                 0.710
  applicant            0.709
  ipc                  0.707
  moutawakel           0.700
  latvia               0.695
  yokohama             0.695
  horizonte            0.687
  sportaccord          0.686
  johannesburg         0.668
  bella                0.666

[frankfurt]
  henriette            0.581
  og                   0.571
  sseldorf             0.550
  heidelberg           0.537
  thorst               0.534
  kleine               0.525
  chabad               0.523
  wirth                0.516
  salons               0.511
  gast                 0.509

[physics]
  cowan                0.578
  physicists           0.553
  buggy                0.546
  udell                0.536
  biology              0.529
  metallurgy           0.514
  btec                 0.507
  farrar               0.501
  atkin                0.501
  alamos               0.500

[chemistry]
  biology              0.541
  aziridines           0.528
  corey                0.521
  chaykovsky           0.510
  btec                 0.502
  nascarella           0.498
  methylene            0.497
  methanide            0.483
  votta                0.481
  reacting             0.478

[stanford]
  cmu                  0.663
  kowalski             0.659
  intrepid             0.630
  rollefson            0.628
  papert               0.615
  hoffmann             0.608
  mit                  0.601
  faculties            0.601
  professions          0.600
  atomics              0.599

[biology]
  btec                 0.679
  silico               0.602
  carpentry            0.594
  vocational           0.591
  vivo                 0.590
  proteomics           0.579
  math                 0.575
  electrophoresis      0.561
  mediated             0.555
  hecht                0.555

[sociology]
  phd                  0.715
  dissertation         0.672
  anthropology         0.668
  doctoral             0.655
  ethnographic         0.634
  asylums              0.601
  math                 0.598
  erving               0.593
  ethnography          0.587
  faculties            0.581

[mathematics]
  btec                 0.687
  lilavati             0.650
  professions          0.612
  mathematicians       0.601
  arithmetic           0.597
  geography            0.597
  textbooks            0.584
  curriculum           0.583
  dvaita               0.580
  bachelor             0.569

[duke]
  villiers             0.529
  devonport            0.503
  cecilius             0.497
  tatum                0.483
  renaming             0.482
  qufu                 0.482
  maximilien           0.467
  guildford            0.456
  duchess              0.455
  ascendancy           0.455

[king]
  overlord             0.513
  diarmata             0.511
  ragnaill             0.511
  ua                   0.508
  donnchad             0.504
  kinsman              0.504
  bagrat               0.502
  murchad              0.502
  meic                 0.500
  sennacherib          0.493

[empress]
  erard                0.649
  azzam                0.646
  genoa                0.644
  kiev                 0.643
  joinville            0.640
  roupen               0.636
  ayyubid              0.635
  nobleman             0.631
  kino                 0.629
  maud                 0.626

[duchess]
  guildford            0.678
  paget                0.665
  northumberland       0.640
  lisle                0.613
  dudley               0.586
  dudleys              0.580
  cheevers             0.577
  countess             0.571
  mengden              0.567
  ranulf               0.562

[prince]
  au                   0.550
  hailing              0.458
  jacmel               0.452
  serge                0.437
  fronti               0.415
  citibank             0.408
  port                 0.393
  sans                 0.392
  neserkauhor          0.387
  spithead             0.385

[emperor]
  huizong              0.612
  xizong               0.611
  qinzong              0.598
  valerian             0.580
  abdicated            0.572
  taizong              0.569
  caizhou              0.561
  gallienus            0.559
  qufu                 0.555
  overlords            0.552

[saves]
  batters              0.594
  shutout              0.583
  strikeouts           0.569
  registering          0.565
  orioles              0.565
  waivers              0.555
  nlcs                 0.552
  tendinitis           0.544
  nathan               0.541
  alds                 0.538

[yankees]
  frazee               0.616
  braves               0.610
  wever                0.598
  sox                  0.594
  pitching             0.590
  alds                 0.586
  mlb                  0.583
  brewers              0.573
  orioles              0.573
  royals               0.571
[Neighbors] Category-aware purity@10 = 0.017
[WordSim] n=318 | Spearman=0.524 | Pearson=0.484
[WordSim] n=909 | Spearman=0.239 | Pearson=0.225
[Analogy] 3cosadd Acc@1 = 0.000 (n=4)
[Analogy] 3cosmul Acc@1 = 0.000 (n=4)
[20NG-CLS] Acc=0.855, Macro-F1=0.854
[20NG-CLU] ARI=0.456, NMI=0.452
```

![](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_001.png)

![plot_002](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_002.png)

![plot_003](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_003.png)

![plot_004](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_004.png)

![plot_005](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_005.png)

![plot_006](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_006.png)

##### SPPMI-SVD:

```
[Load] WikiText-2 ...
[Load] Raw lines: 44,836 in 10.94s
[Corpus] sentences=28,369, tokens=1,094,036, |V|=27,091
[Train] SPPMI-SVD config: {
  "vector_size": 300,
  "window": 10,
  "min_count": 2,
  "subsample_t": 0.001,
  "shift_k": 15.0,
  "svd_power": 0.5,
  "seed": 4213
}
[SPPMI] |V| after min_count=2: 27091
[SPPMI] Cooc built: nnz=8,201,880 in 6.25s
[SPPMI] SPPMI built: nnz=8,201,880 in 5.16s
[SPPMI] SVD done in 40.65s; explained_var=0.130
[Train] Done in 52.27s (SPPMI-SVD)
[Vocab] Top-10: ['time', 'game', 'city', 'song', 'year', 'made', 'season', 'war', 'north', 'used']
[Vocab] Size: 27091
[Plot saved to plots/SPPMI-SVD\plot_001.png]
[Plot saved to plots/SPPMI-SVD\plot_002.png]
[Plot saved to plots/SPPMI-SVD\plot_003.png]
[Plot saved to plots/SPPMI-SVD\plot_004.png]
[Plot saved to plots/SPPMI-SVD\plot_005.png]
[Plot saved to plots/SPPMI-SVD\plot_006.png]
[Neighbors] Probes: 30 over 5 buckets.

[tokyo]
  bid                  0.768
  venues               0.751
  rio                  0.747
  athens               0.741
  bidding              0.733
  baku                 0.726
  arena                0.722
  moscow               0.721
  ioc                  0.716
  japan                0.711

[london]
  house                0.781
  theatre              0.764
  travelled            0.761
  invited              0.754
  attended             0.753
  residence            0.745
  museum               0.739
  meeting              0.729
  nottingham           0.729
  albert               0.729

[athens]
  tokyo                0.741
  greece               0.721
  olympics             0.720
  paralympics          0.703
  olympic              0.694
  cities               0.686
  sportaccord          0.679
  beijing              0.678
  summer               0.671
  venues               0.671

[beijing]
  mongols              0.766
  delegation           0.762
  chad                 0.720
  hangzhou             0.718
  yuan                 0.717
  xia                  0.707
  prefectures          0.707
  burma                0.697
  liao                 0.694
  negotiations         0.691

[lausanne]
  ioc                  0.900
  applicant            0.895
  candidature          0.865
  janeiro              0.864
  ipc                  0.862
  rio                  0.858
  baku                 0.852
  doha                 0.841
  sportaccord          0.802
  nuzman               0.797

[berlin]
  prussian             0.787
  opera                0.773
  meyerbeer            0.770
  kapellmeister        0.757
  huguenots            0.743
  paris                0.740
  wagner               0.736
  beer                 0.722
  operas               0.713
  germany              0.710

[physics]
  reines               0.794
  mathematics          0.756
  creutz               0.748
  laboratory           0.740
  alamos               0.737
  research             0.725
  atomics              0.724
  science              0.722
  sciences             0.721
  economics            0.717

[chemistry]
  appreciated          0.699
  reaction             0.689
  felt                 0.675
  reactions            0.670
  pairing              0.669
  screen               0.666
  catalyst             0.660
  noting               0.657
  isn                  0.656
  biology              0.655

[mathematics]
  economics            0.810
  sciences             0.801
  btec                 0.785
  geography            0.779
  bachelor             0.758
  physics              0.756
  arithmetic           0.752
  psychology           0.749
  science              0.730
  graduate             0.728

[sociology]
  anthropology         0.833
  economics            0.824
  goffman              0.802
  sciences             0.798
  ethnographic         0.774
  doctoral             0.774
  phd                  0.765
  academic             0.758
  innis                0.750
  dissertation         0.747

[thesis]
  doctoral             0.826
  phd                  0.824
  innis                0.771
  economics            0.770
  zionism              0.750
  dissertation         0.750
  essay                0.726
  sociology            0.723
  princeton            0.711
  academic             0.710

[doctoral]
  dissertation         0.885
  phd                  0.831
  thesis               0.826
  academic             0.786
  zionism              0.778
  sociology            0.774
  princeton            0.754
  faculties            0.752
  scholarship          0.748
  graduate             0.736

[prince]
  edward               0.741
  duke                 0.737
  royal                0.725
  collapsed            0.703
  loyal                0.692
  au                   0.690
  albert               0.689
  imperial             0.683
  queen                0.681
  remained             0.676

[duke]
  edward               0.827
  earl                 0.823
  dudley               0.820
  lancaster            0.806
  northumberland       0.802
  duchess              0.791
  henry                0.763
  guildford            0.758
  cecil                0.752
  warwick              0.748

[empress]
  constantinople       0.717
  sicily               0.697
  italy                0.686
  crowned              0.685
  arrival              0.673
  coronation           0.666
  surrendered          0.666
  gibraltar            0.665
  invasion             0.664
  matilda              0.662

[duchess]
  dudley               0.880
  guildford            0.876
  northumberland       0.859
  countess             0.817
  duke                 0.791
  warwick              0.784
  lisle                0.780
  earl                 0.777
  anne                 0.762
  paget                0.760

[queen]
  northumberland       0.711
  charlotte            0.704
  lord                 0.686
  elizabeth            0.685
  prince               0.681
  spithead             0.679
  royal                0.677
  edward               0.672
  loyal                0.662
  albert               0.660

[king]
  son                  0.784
  grandfather          0.755
  kings                0.754
  grandson             0.740
  nephew               0.735
  succeeded            0.733
  vassal               0.733
  throne               0.728
  accession            0.727
  successor            0.719

[yankees]
  sox                  0.936
  dodgers              0.917
  pitcher              0.912
  braves               0.909
  mlb                  0.909
  phillies             0.906
  pitching             0.903
  nathan               0.898
  nl                   0.898
  giants               0.897

[saves]
  strikeouts           0.900
  mlb                  0.889
  dodgers              0.889
  postseason           0.884
  strikeout            0.882
  nathan               0.879
  pitchers             0.878
  batters              0.875
  phillies             0.874
  pitcher              0.873
[Neighbors] Category-aware purity@10 = 0.077
[WordSim] n=318 | Spearman=0.376 | Pearson=0.390
[WordSim] n=909 | Spearman=0.147 | Pearson=0.155
[Analogy] 3cosadd Acc@1 = 0.000 (n=4)
[Analogy] 3cosmul Acc@1 = 0.000 (n=4)
[20NG-CLS] Acc=0.832, Macro-F1=0.832
[20NG-CLU] ARI=0.176, NMI=0.241
```

![](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_001.png)

![plot_002](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_002.png)

![plot_003](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_003.png)

![plot_004](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_004.png)

![plot_005](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_005.png)

![plot_006](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_006.png)

##### GloVe:

```
[Load] WikiText-2 ...
[Load] Raw lines: 44,836 in 14.04s
[Corpus] sentences=28,369, tokens=1,094,036, |V|=27,091
[Train] GloVe config: {
  "vector_size": 300,
  "window": 10,
  "min_count": 2,
  "epochs": 15,
  "x_max": 100.0,
  "alpha": 0.75,
  "lr": 0.025,
  "seed": 4213
}
[GloVe] Vocab size after min_count=2: 27,091
[GloVe] Co-occurrence entries: 6,714,664
[GloVe][epoch 1] step=100,000/6,714,664  avg_loss≈0.0281
[GloVe][epoch 1] step=200,000/6,714,664  avg_loss≈0.0277
[GloVe][epoch 1] step=300,000/6,714,664  avg_loss≈0.0277
[GloVe][epoch 1] step=400,000/6,714,664  avg_loss≈0.0262
[GloVe][epoch 1] step=500,000/6,714,664  avg_loss≈0.0264
[GloVe][epoch 1] step=600,000/6,714,664  avg_loss≈0.0260
[GloVe][epoch 1] step=700,000/6,714,664  avg_loss≈0.0266
[GloVe][epoch 1] step=800,000/6,714,664  avg_loss≈0.0275
[GloVe][epoch 1] step=900,000/6,714,664  avg_loss≈0.0259
[GloVe][epoch 1] step=1,000,000/6,714,664  avg_loss≈0.0260
[GloVe][epoch 1] step=1,100,000/6,714,664  avg_loss≈0.0277
[GloVe][epoch 1] step=1,200,000/6,714,664  avg_loss≈0.0255
[GloVe][epoch 1] step=1,300,000/6,714,664  avg_loss≈0.0256
[GloVe][epoch 1] step=1,400,000/6,714,664  avg_loss≈0.0261
[GloVe][epoch 1] step=1,500,000/6,714,664  avg_loss≈0.0252
[GloVe][epoch 1] step=1,600,000/6,714,664  avg_loss≈0.0251
[GloVe][epoch 1] step=1,700,000/6,714,664  avg_loss≈0.0251
[GloVe][epoch 1] step=1,800,000/6,714,664  avg_loss≈0.0246
[GloVe][epoch 1] step=1,900,000/6,714,664  avg_loss≈0.0252
[GloVe][epoch 1] step=2,000,000/6,714,664  avg_loss≈0.0266
[GloVe][epoch 1] step=2,100,000/6,714,664  avg_loss≈0.0255
[GloVe][epoch 1] step=2,200,000/6,714,664  avg_loss≈0.0252
[GloVe][epoch 1] step=2,300,000/6,714,664  avg_loss≈0.0249
[GloVe][epoch 1] step=2,400,000/6,714,664  avg_loss≈0.0260
[GloVe][epoch 1] step=2,500,000/6,714,664  avg_loss≈0.0253
[GloVe][epoch 1] step=2,600,000/6,714,664  avg_loss≈0.0258
[GloVe][epoch 1] step=2,700,000/6,714,664  avg_loss≈0.0256
[GloVe][epoch 1] step=2,800,000/6,714,664  avg_loss≈0.0246
[GloVe][epoch 1] step=2,900,000/6,714,664  avg_loss≈0.0249
[GloVe][epoch 1] step=3,000,000/6,714,664  avg_loss≈0.0246
[GloVe][epoch 1] step=3,100,000/6,714,664  avg_loss≈0.0254
[GloVe][epoch 1] step=3,200,000/6,714,664  avg_loss≈0.0240
[GloVe][epoch 1] step=3,300,000/6,714,664  avg_loss≈0.0255
[GloVe][epoch 1] step=3,400,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 1] step=3,500,000/6,714,664  avg_loss≈0.0245
[GloVe][epoch 1] step=3,600,000/6,714,664  avg_loss≈0.0265
[GloVe][epoch 1] step=3,700,000/6,714,664  avg_loss≈0.0261
[GloVe][epoch 1] step=3,800,000/6,714,664  avg_loss≈0.0249
[GloVe][epoch 1] step=3,900,000/6,714,664  avg_loss≈0.0250
[GloVe][epoch 1] step=4,000,000/6,714,664  avg_loss≈0.0253
[GloVe][epoch 1] step=4,100,000/6,714,664  avg_loss≈0.0250
[GloVe][epoch 1] step=4,200,000/6,714,664  avg_loss≈0.0244
[GloVe][epoch 1] step=4,300,000/6,714,664  avg_loss≈0.0238
[GloVe][epoch 1] step=4,400,000/6,714,664  avg_loss≈0.0250
[GloVe][epoch 1] step=4,500,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 1] step=4,600,000/6,714,664  avg_loss≈0.0252
[GloVe][epoch 1] step=4,700,000/6,714,664  avg_loss≈0.0256
[GloVe][epoch 1] step=4,800,000/6,714,664  avg_loss≈0.0249
[GloVe][epoch 1] step=4,900,000/6,714,664  avg_loss≈0.0235
[GloVe][epoch 1] step=5,000,000/6,714,664  avg_loss≈0.0241
[GloVe][epoch 1] step=5,100,000/6,714,664  avg_loss≈0.0250
[GloVe][epoch 1] step=5,200,000/6,714,664  avg_loss≈0.0240
[GloVe][epoch 1] step=5,300,000/6,714,664  avg_loss≈0.0248
[GloVe][epoch 1] step=5,400,000/6,714,664  avg_loss≈0.0248
[GloVe][epoch 1] step=5,500,000/6,714,664  avg_loss≈0.0247
[GloVe][epoch 1] step=5,600,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 1] step=5,700,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 1] step=5,800,000/6,714,664  avg_loss≈0.0256
[GloVe][epoch 1] step=5,900,000/6,714,664  avg_loss≈0.0246
[GloVe][epoch 1] step=6,000,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 1] step=6,100,000/6,714,664  avg_loss≈0.0235
[GloVe][epoch 1] step=6,200,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 1] step=6,300,000/6,714,664  avg_loss≈0.0240
[GloVe][epoch 1] step=6,400,000/6,714,664  avg_loss≈0.0245
[GloVe][epoch 1] step=6,500,000/6,714,664  avg_loss≈0.0236
[GloVe][epoch 1] step=6,600,000/6,714,664  avg_loss≈0.0240
[GloVe][epoch 1] step=6,700,000/6,714,664  avg_loss≈0.0242
[GloVe] epoch 1/15 done in 113.71s
[GloVe][epoch 2] step=100,000/6,714,664  avg_loss≈0.0244
[GloVe][epoch 2] step=200,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 2] step=300,000/6,714,664  avg_loss≈0.0244
[GloVe][epoch 2] step=400,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 2] step=500,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 2] step=600,000/6,714,664  avg_loss≈0.0238
[GloVe][epoch 2] step=700,000/6,714,664  avg_loss≈0.0243
[GloVe][epoch 2] step=800,000/6,714,664  avg_loss≈0.0236
[GloVe][epoch 2] step=900,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 2] step=1,000,000/6,714,664  avg_loss≈0.0244
[GloVe][epoch 2] step=1,100,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 2] step=1,200,000/6,714,664  avg_loss≈0.0241
[GloVe][epoch 2] step=1,300,000/6,714,664  avg_loss≈0.0244
[GloVe][epoch 2] step=1,400,000/6,714,664  avg_loss≈0.0241
[GloVe][epoch 2] step=1,500,000/6,714,664  avg_loss≈0.0249
[GloVe][epoch 2] step=1,600,000/6,714,664  avg_loss≈0.0241
[GloVe][epoch 2] step=1,700,000/6,714,664  avg_loss≈0.0238
[GloVe][epoch 2] step=1,800,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 2] step=1,900,000/6,714,664  avg_loss≈0.0253
[GloVe][epoch 2] step=2,000,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 2] step=2,100,000/6,714,664  avg_loss≈0.0244
[GloVe][epoch 2] step=2,200,000/6,714,664  avg_loss≈0.0238
[GloVe][epoch 2] step=2,300,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 2] step=2,400,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 2] step=2,500,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 2] step=2,600,000/6,714,664  avg_loss≈0.0233
[GloVe][epoch 2] step=2,700,000/6,714,664  avg_loss≈0.0238
[GloVe][epoch 2] step=2,800,000/6,714,664  avg_loss≈0.0236
[GloVe][epoch 2] step=2,900,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 2] step=3,000,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 2] step=3,100,000/6,714,664  avg_loss≈0.0224
[GloVe][epoch 2] step=3,200,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 2] step=3,300,000/6,714,664  avg_loss≈0.0242
[GloVe][epoch 2] step=3,400,000/6,714,664  avg_loss≈0.0241
[GloVe][epoch 2] step=3,500,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 2] step=3,600,000/6,714,664  avg_loss≈0.0236
[GloVe][epoch 2] step=3,700,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 2] step=3,800,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 2] step=3,900,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 2] step=4,000,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 2] step=4,100,000/6,714,664  avg_loss≈0.0238
[GloVe][epoch 2] step=4,200,000/6,714,664  avg_loss≈0.0248
[GloVe][epoch 2] step=4,300,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 2] step=4,400,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 2] step=4,500,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 2] step=4,600,000/6,714,664  avg_loss≈0.0236
[GloVe][epoch 2] step=4,700,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 2] step=4,800,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 2] step=4,900,000/6,714,664  avg_loss≈0.0226
[GloVe][epoch 2] step=5,000,000/6,714,664  avg_loss≈0.0226
[GloVe][epoch 2] step=5,100,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 2] step=5,200,000/6,714,664  avg_loss≈0.0236
[GloVe][epoch 2] step=5,300,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 2] step=5,400,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 2] step=5,500,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 2] step=5,600,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 2] step=5,700,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 2] step=5,800,000/6,714,664  avg_loss≈0.0233
[GloVe][epoch 2] step=5,900,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 2] step=6,000,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 2] step=6,100,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 2] step=6,200,000/6,714,664  avg_loss≈0.0230
[GloVe][epoch 2] step=6,300,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 2] step=6,400,000/6,714,664  avg_loss≈0.0226
[GloVe][epoch 2] step=6,500,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 2] step=6,600,000/6,714,664  avg_loss≈0.0245
[GloVe][epoch 2] step=6,700,000/6,714,664  avg_loss≈0.0241
[GloVe] epoch 2/15 done in 105.16s
[GloVe][epoch 3] step=100,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 3] step=200,000/6,714,664  avg_loss≈0.0238
[GloVe][epoch 3] step=300,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 3] step=400,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 3] step=500,000/6,714,664  avg_loss≈0.0225
[GloVe][epoch 3] step=600,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 3] step=700,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 3] step=800,000/6,714,664  avg_loss≈0.0235
[GloVe][epoch 3] step=900,000/6,714,664  avg_loss≈0.0235
[GloVe][epoch 3] step=1,000,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 3] step=1,100,000/6,714,664  avg_loss≈0.0230
[GloVe][epoch 3] step=1,200,000/6,714,664  avg_loss≈0.0225
[GloVe][epoch 3] step=1,300,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 3] step=1,400,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 3] step=1,500,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 3] step=1,600,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 3] step=1,700,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 3] step=1,800,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 3] step=1,900,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 3] step=2,000,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 3] step=2,100,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 3] step=2,200,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 3] step=2,300,000/6,714,664  avg_loss≈0.0230
[GloVe][epoch 3] step=2,400,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 3] step=2,500,000/6,714,664  avg_loss≈0.0226
[GloVe][epoch 3] step=2,600,000/6,714,664  avg_loss≈0.0235
[GloVe][epoch 3] step=2,700,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 3] step=2,800,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 3] step=2,900,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 3] step=3,000,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 3] step=3,100,000/6,714,664  avg_loss≈0.0220
[GloVe][epoch 3] step=3,200,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 3] step=3,300,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 3] step=3,400,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 3] step=3,500,000/6,714,664  avg_loss≈0.0220
[GloVe][epoch 3] step=3,600,000/6,714,664  avg_loss≈0.0230
[GloVe][epoch 3] step=3,700,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 3] step=3,800,000/6,714,664  avg_loss≈0.0225
[GloVe][epoch 3] step=3,900,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 3] step=4,000,000/6,714,664  avg_loss≈0.0226
[GloVe][epoch 3] step=4,100,000/6,714,664  avg_loss≈0.0222
[GloVe][epoch 3] step=4,200,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 3] step=4,300,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 3] step=4,400,000/6,714,664  avg_loss≈0.0233
[GloVe][epoch 3] step=4,500,000/6,714,664  avg_loss≈0.0224
[GloVe][epoch 3] step=4,600,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 3] step=4,700,000/6,714,664  avg_loss≈0.0230
[GloVe][epoch 3] step=4,800,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 3] step=4,900,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 3] step=5,000,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 3] step=5,100,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 3] step=5,200,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 3] step=5,300,000/6,714,664  avg_loss≈0.0238
[GloVe][epoch 3] step=5,400,000/6,714,664  avg_loss≈0.0225
[GloVe][epoch 3] step=5,500,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 3] step=5,600,000/6,714,664  avg_loss≈0.0236
[GloVe][epoch 3] step=5,700,000/6,714,664  avg_loss≈0.0235
[GloVe][epoch 3] step=5,800,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 3] step=5,900,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 3] step=6,000,000/6,714,664  avg_loss≈0.0233
[GloVe][epoch 3] step=6,100,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 3] step=6,200,000/6,714,664  avg_loss≈0.0236
[GloVe][epoch 3] step=6,300,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 3] step=6,400,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 3] step=6,500,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 3] step=6,600,000/6,714,664  avg_loss≈0.0235
[GloVe][epoch 3] step=6,700,000/6,714,664  avg_loss≈0.0225
[GloVe] epoch 3/15 done in 104.55s
[GloVe][epoch 4] step=100,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 4] step=200,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 4] step=300,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 4] step=400,000/6,714,664  avg_loss≈0.0226
[GloVe][epoch 4] step=500,000/6,714,664  avg_loss≈0.0222
[GloVe][epoch 4] step=600,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 4] step=700,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 4] step=800,000/6,714,664  avg_loss≈0.0225
[GloVe][epoch 4] step=900,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 4] step=1,000,000/6,714,664  avg_loss≈0.0224
[GloVe][epoch 4] step=1,100,000/6,714,664  avg_loss≈0.0222
[GloVe][epoch 4] step=1,200,000/6,714,664  avg_loss≈0.0234
[GloVe][epoch 4] step=1,300,000/6,714,664  avg_loss≈0.0230
[GloVe][epoch 4] step=1,400,000/6,714,664  avg_loss≈0.0237
[GloVe][epoch 4] step=1,500,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 4] step=1,600,000/6,714,664  avg_loss≈0.0226
[GloVe][epoch 4] step=1,700,000/6,714,664  avg_loss≈0.0224
[GloVe][epoch 4] step=1,800,000/6,714,664  avg_loss≈0.0226
[GloVe][epoch 4] step=1,900,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 4] step=2,000,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 4] step=2,100,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 4] step=2,200,000/6,714,664  avg_loss≈0.0232
[GloVe][epoch 4] step=2,300,000/6,714,664  avg_loss≈0.0239
[GloVe][epoch 4] step=2,400,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 4] step=2,500,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 4] step=2,600,000/6,714,664  avg_loss≈0.0225
[GloVe][epoch 4] step=2,700,000/6,714,664  avg_loss≈0.0227
[GloVe][epoch 4] step=2,800,000/6,714,664  avg_loss≈0.0220
[GloVe][epoch 4] step=2,900,000/6,714,664  avg_loss≈0.0220
[GloVe][epoch 4] step=3,000,000/6,714,664  avg_loss≈0.0220
[GloVe][epoch 4] step=3,100,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 4] step=3,200,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 4] step=3,300,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 4] step=3,400,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 4] step=3,500,000/6,714,664  avg_loss≈0.0214
[GloVe][epoch 4] step=3,600,000/6,714,664  avg_loss≈0.0230
[GloVe][epoch 4] step=3,700,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 4] step=3,800,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 4] step=3,900,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 4] step=4,000,000/6,714,664  avg_loss≈0.0224
[GloVe][epoch 4] step=4,100,000/6,714,664  avg_loss≈0.0225
[GloVe][epoch 4] step=4,200,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 4] step=4,300,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 4] step=4,400,000/6,714,664  avg_loss≈0.0230
[GloVe][epoch 4] step=4,500,000/6,714,664  avg_loss≈0.0229
[GloVe][epoch 4] step=4,600,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 4] step=4,700,000/6,714,664  avg_loss≈0.0217
[GloVe][epoch 4] step=4,800,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 4] step=4,900,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 4] step=5,000,000/6,714,664  avg_loss≈0.0228
[GloVe][epoch 4] step=5,100,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 4] step=5,200,000/6,714,664  avg_loss≈0.0233
[GloVe][epoch 4] step=5,300,000/6,714,664  avg_loss≈0.0214
[GloVe][epoch 4] step=5,400,000/6,714,664  avg_loss≈0.0216
[GloVe][epoch 4] step=5,500,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 4] step=5,600,000/6,714,664  avg_loss≈0.0225
[GloVe][epoch 4] step=5,700,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 4] step=5,800,000/6,714,664  avg_loss≈0.0220
[GloVe][epoch 4] step=5,900,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 4] step=6,000,000/6,714,664  avg_loss≈0.0240
[GloVe][epoch 4] step=6,100,000/6,714,664  avg_loss≈0.0231
[GloVe][epoch 4] step=6,200,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 4] step=6,300,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 4] step=6,400,000/6,714,664  avg_loss≈0.0216
[GloVe][epoch 4] step=6,500,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 4] step=6,600,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 4] step=6,700,000/6,714,664  avg_loss≈0.0218
[GloVe] epoch 4/15 done in 105.73s
[GloVe][epoch 5] step=100,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 5] step=200,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 5] step=300,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 5] step=400,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 5] step=500,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 5] step=600,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 5] step=700,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 5] step=800,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=900,000/6,714,664  avg_loss≈0.0222
[GloVe][epoch 5] step=1,000,000/6,714,664  avg_loss≈0.0220
[GloVe][epoch 5] step=1,100,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 5] step=1,200,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 5] step=1,300,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 5] step=1,400,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=1,500,000/6,714,664  avg_loss≈0.0222
[GloVe][epoch 5] step=1,600,000/6,714,664  avg_loss≈0.0212
[GloVe][epoch 5] step=1,700,000/6,714,664  avg_loss≈0.0230
[GloVe][epoch 5] step=1,800,000/6,714,664  avg_loss≈0.0222
[GloVe][epoch 5] step=1,900,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 5] step=2,000,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=2,100,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 5] step=2,200,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 5] step=2,300,000/6,714,664  avg_loss≈0.0216
[GloVe][epoch 5] step=2,400,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 5] step=2,500,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 5] step=2,600,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=2,700,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 5] step=2,800,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 5] step=2,900,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 5] step=3,000,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 5] step=3,100,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 5] step=3,200,000/6,714,664  avg_loss≈0.0217
[GloVe][epoch 5] step=3,300,000/6,714,664  avg_loss≈0.0217
[GloVe][epoch 5] step=3,400,000/6,714,664  avg_loss≈0.0214
[GloVe][epoch 5] step=3,500,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=3,600,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 5] step=3,700,000/6,714,664  avg_loss≈0.0217
[GloVe][epoch 5] step=3,800,000/6,714,664  avg_loss≈0.0216
[GloVe][epoch 5] step=3,900,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 5] step=4,000,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=4,100,000/6,714,664  avg_loss≈0.0212
[GloVe][epoch 5] step=4,200,000/6,714,664  avg_loss≈0.0216
[GloVe][epoch 5] step=4,300,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=4,400,000/6,714,664  avg_loss≈0.0220
[GloVe][epoch 5] step=4,500,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 5] step=4,600,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 5] step=4,700,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=4,800,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 5] step=4,900,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 5] step=5,000,000/6,714,664  avg_loss≈0.0220
[GloVe][epoch 5] step=5,100,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 5] step=5,200,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 5] step=5,300,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 5] step=5,400,000/6,714,664  avg_loss≈0.0216
[GloVe][epoch 5] step=5,500,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=5,600,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 5] step=5,700,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 5] step=5,800,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 5] step=5,900,000/6,714,664  avg_loss≈0.0221
[GloVe][epoch 5] step=6,000,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 5] step=6,100,000/6,714,664  avg_loss≈0.0212
[GloVe][epoch 5] step=6,200,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=6,300,000/6,714,664  avg_loss≈0.0211
[GloVe][epoch 5] step=6,400,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 5] step=6,500,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 5] step=6,600,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 5] step=6,700,000/6,714,664  avg_loss≈0.0210
[GloVe] epoch 5/15 done in 106.43s
[GloVe][epoch 6] step=100,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 6] step=200,000/6,714,664  avg_loss≈0.0212
[GloVe][epoch 6] step=300,000/6,714,664  avg_loss≈0.0218
[GloVe][epoch 6] step=400,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 6] step=500,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 6] step=600,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 6] step=700,000/6,714,664  avg_loss≈0.0214
[GloVe][epoch 6] step=800,000/6,714,664  avg_loss≈0.0212
[GloVe][epoch 6] step=900,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 6] step=1,000,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 6] step=1,100,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 6] step=1,200,000/6,714,664  avg_loss≈0.0223
[GloVe][epoch 6] step=1,300,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 6] step=1,400,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 6] step=1,500,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 6] step=1,600,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 6] step=1,700,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 6] step=1,800,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 6] step=1,900,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 6] step=2,000,000/6,714,664  avg_loss≈0.0216
[GloVe][epoch 6] step=2,100,000/6,714,664  avg_loss≈0.0211
[GloVe][epoch 6] step=2,200,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 6] step=2,300,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 6] step=2,400,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 6] step=2,500,000/6,714,664  avg_loss≈0.0222
[GloVe][epoch 6] step=2,600,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 6] step=2,700,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 6] step=2,800,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 6] step=2,900,000/6,714,664  avg_loss≈0.0211
[GloVe][epoch 6] step=3,000,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 6] step=3,100,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 6] step=3,200,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 6] step=3,300,000/6,714,664  avg_loss≈0.0216
[GloVe][epoch 6] step=3,400,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 6] step=3,500,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 6] step=3,600,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 6] step=3,700,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 6] step=3,800,000/6,714,664  avg_loss≈0.0217
[GloVe][epoch 6] step=3,900,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 6] step=4,000,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 6] step=4,100,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 6] step=4,200,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 6] step=4,300,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 6] step=4,400,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 6] step=4,500,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 6] step=4,600,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 6] step=4,700,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 6] step=4,800,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 6] step=4,900,000/6,714,664  avg_loss≈0.0215
[GloVe][epoch 6] step=5,000,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 6] step=5,100,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 6] step=5,200,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 6] step=5,300,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 6] step=5,400,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 6] step=5,500,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 6] step=5,600,000/6,714,664  avg_loss≈0.0216
[GloVe][epoch 6] step=5,700,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 6] step=5,800,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 6] step=5,900,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 6] step=6,000,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 6] step=6,100,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 6] step=6,200,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 6] step=6,300,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 6] step=6,400,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 6] step=6,500,000/6,714,664  avg_loss≈0.0217
[GloVe][epoch 6] step=6,600,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 6] step=6,700,000/6,714,664  avg_loss≈0.0208
[GloVe] epoch 6/15 done in 104.29s
[GloVe][epoch 7] step=100,000/6,714,664  avg_loss≈0.0214
[GloVe][epoch 7] step=200,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 7] step=300,000/6,714,664  avg_loss≈0.0211
[GloVe][epoch 7] step=400,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 7] step=500,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 7] step=600,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 7] step=700,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 7] step=800,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 7] step=900,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 7] step=1,000,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 7] step=1,100,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 7] step=1,200,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 7] step=1,300,000/6,714,664  avg_loss≈0.0212
[GloVe][epoch 7] step=1,400,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 7] step=1,500,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 7] step=1,600,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 7] step=1,700,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 7] step=1,800,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 7] step=1,900,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 7] step=2,000,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 7] step=2,100,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 7] step=2,200,000/6,714,664  avg_loss≈0.0211
[GloVe][epoch 7] step=2,300,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 7] step=2,400,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 7] step=2,500,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 7] step=2,600,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 7] step=2,700,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 7] step=2,800,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 7] step=2,900,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 7] step=3,000,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 7] step=3,100,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 7] step=3,200,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 7] step=3,300,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 7] step=3,400,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 7] step=3,500,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 7] step=3,600,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 7] step=3,700,000/6,714,664  avg_loss≈0.0213
[GloVe][epoch 7] step=3,800,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 7] step=3,900,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 7] step=4,000,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 7] step=4,100,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 7] step=4,200,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 7] step=4,300,000/6,714,664  avg_loss≈0.0219
[GloVe][epoch 7] step=4,400,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 7] step=4,500,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 7] step=4,600,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 7] step=4,700,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 7] step=4,800,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 7] step=4,900,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 7] step=5,000,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 7] step=5,100,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 7] step=5,200,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 7] step=5,300,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 7] step=5,400,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 7] step=5,500,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 7] step=5,600,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 7] step=5,700,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 7] step=5,800,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 7] step=5,900,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 7] step=6,000,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 7] step=6,100,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 7] step=6,200,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 7] step=6,300,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 7] step=6,400,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 7] step=6,500,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 7] step=6,600,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 7] step=6,700,000/6,714,664  avg_loss≈0.0209
[GloVe] epoch 7/15 done in 104.10s
[GloVe][epoch 8] step=100,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 8] step=200,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 8] step=300,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 8] step=400,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 8] step=500,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 8] step=600,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 8] step=700,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 8] step=800,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 8] step=900,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 8] step=1,000,000/6,714,664  avg_loss≈0.0209
[GloVe][epoch 8] step=1,100,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 8] step=1,200,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 8] step=1,300,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 8] step=1,400,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 8] step=1,500,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 8] step=1,600,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 8] step=1,700,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 8] step=1,800,000/6,714,664  avg_loss≈0.0211
[GloVe][epoch 8] step=1,900,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 8] step=2,000,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 8] step=2,100,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 8] step=2,200,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 8] step=2,300,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 8] step=2,400,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 8] step=2,500,000/6,714,664  avg_loss≈0.0208
[GloVe][epoch 8] step=2,600,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 8] step=2,700,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 8] step=2,800,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 8] step=2,900,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 8] step=3,000,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 8] step=3,100,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 8] step=3,200,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 8] step=3,300,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 8] step=3,400,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 8] step=3,500,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 8] step=3,600,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 8] step=3,700,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 8] step=3,800,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 8] step=3,900,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 8] step=4,000,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 8] step=4,100,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 8] step=4,200,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 8] step=4,300,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 8] step=4,400,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 8] step=4,500,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 8] step=4,600,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 8] step=4,700,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 8] step=4,800,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 8] step=4,900,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 8] step=5,000,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 8] step=5,100,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 8] step=5,200,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 8] step=5,300,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 8] step=5,400,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 8] step=5,500,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 8] step=5,600,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 8] step=5,700,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 8] step=5,800,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 8] step=5,900,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 8] step=6,000,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 8] step=6,100,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 8] step=6,200,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 8] step=6,300,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 8] step=6,400,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 8] step=6,500,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 8] step=6,600,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 8] step=6,700,000/6,714,664  avg_loss≈0.0203
[GloVe] epoch 8/15 done in 107.00s
[GloVe][epoch 9] step=100,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 9] step=200,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 9] step=300,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 9] step=400,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 9] step=500,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 9] step=600,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 9] step=700,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 9] step=800,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 9] step=900,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 9] step=1,000,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 9] step=1,100,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 9] step=1,200,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 9] step=1,300,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 9] step=1,400,000/6,714,664  avg_loss≈0.0210
[GloVe][epoch 9] step=1,500,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 9] step=1,600,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 9] step=1,700,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 9] step=1,800,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 9] step=1,900,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 9] step=2,000,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 9] step=2,100,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 9] step=2,200,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 9] step=2,300,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 9] step=2,400,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 9] step=2,500,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 9] step=2,600,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 9] step=2,700,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 9] step=2,800,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 9] step=2,900,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 9] step=3,000,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 9] step=3,100,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 9] step=3,200,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 9] step=3,300,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 9] step=3,400,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 9] step=3,500,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 9] step=3,600,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 9] step=3,700,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 9] step=3,800,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 9] step=3,900,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 9] step=4,000,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 9] step=4,100,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 9] step=4,200,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 9] step=4,300,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 9] step=4,400,000/6,714,664  avg_loss≈0.0222
[GloVe][epoch 9] step=4,500,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 9] step=4,600,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 9] step=4,700,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 9] step=4,800,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 9] step=4,900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 9] step=5,000,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 9] step=5,100,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 9] step=5,200,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 9] step=5,300,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 9] step=5,400,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 9] step=5,500,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 9] step=5,600,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 9] step=5,700,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 9] step=5,800,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 9] step=5,900,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 9] step=6,000,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 9] step=6,100,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 9] step=6,200,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 9] step=6,300,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 9] step=6,400,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 9] step=6,500,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 9] step=6,600,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 9] step=6,700,000/6,714,664  avg_loss≈0.0193
[GloVe] epoch 9/15 done in 104.95s
[GloVe][epoch 10] step=100,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 10] step=200,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 10] step=300,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 10] step=400,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 10] step=500,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 10] step=600,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 10] step=700,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 10] step=800,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 10] step=900,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 10] step=1,000,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 10] step=1,100,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 10] step=1,200,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 10] step=1,300,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 10] step=1,400,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 10] step=1,500,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 10] step=1,600,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 10] step=1,700,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 10] step=1,800,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 10] step=1,900,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 10] step=2,000,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 10] step=2,100,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 10] step=2,200,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 10] step=2,300,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 10] step=2,400,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 10] step=2,500,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 10] step=2,600,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 10] step=2,700,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 10] step=2,800,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 10] step=2,900,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 10] step=3,000,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 10] step=3,100,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 10] step=3,200,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 10] step=3,300,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 10] step=3,400,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 10] step=3,500,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 10] step=3,600,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 10] step=3,700,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 10] step=3,800,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 10] step=3,900,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 10] step=4,000,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 10] step=4,100,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 10] step=4,200,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 10] step=4,300,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 10] step=4,400,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 10] step=4,500,000/6,714,664  avg_loss≈0.0206
[GloVe][epoch 10] step=4,600,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 10] step=4,700,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 10] step=4,800,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 10] step=4,900,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 10] step=5,000,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 10] step=5,100,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 10] step=5,200,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 10] step=5,300,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 10] step=5,400,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 10] step=5,500,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 10] step=5,600,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 10] step=5,700,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 10] step=5,800,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 10] step=5,900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 10] step=6,000,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 10] step=6,100,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 10] step=6,200,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 10] step=6,300,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 10] step=6,400,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 10] step=6,500,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 10] step=6,600,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 10] step=6,700,000/6,714,664  avg_loss≈0.0200
[GloVe] epoch 10/15 done in 104.80s
[GloVe][epoch 11] step=100,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 11] step=200,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 11] step=300,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 11] step=400,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 11] step=500,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 11] step=600,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 11] step=700,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 11] step=800,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 11] step=900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 11] step=1,000,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 11] step=1,100,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 11] step=1,200,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 11] step=1,300,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 11] step=1,400,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 11] step=1,500,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 11] step=1,600,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 11] step=1,700,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 11] step=1,800,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 11] step=1,900,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 11] step=2,000,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 11] step=2,100,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 11] step=2,200,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 11] step=2,300,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 11] step=2,400,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 11] step=2,500,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 11] step=2,600,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 11] step=2,700,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 11] step=2,800,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 11] step=2,900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 11] step=3,000,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 11] step=3,100,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 11] step=3,200,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 11] step=3,300,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 11] step=3,400,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 11] step=3,500,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 11] step=3,600,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 11] step=3,700,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 11] step=3,800,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 11] step=3,900,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 11] step=4,000,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 11] step=4,100,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 11] step=4,200,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 11] step=4,300,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 11] step=4,400,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 11] step=4,500,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 11] step=4,600,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 11] step=4,700,000/6,714,664  avg_loss≈0.0205
[GloVe][epoch 11] step=4,800,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 11] step=4,900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 11] step=5,000,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 11] step=5,100,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 11] step=5,200,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 11] step=5,300,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 11] step=5,400,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 11] step=5,500,000/6,714,664  avg_loss≈0.0189
[GloVe][epoch 11] step=5,600,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 11] step=5,700,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 11] step=5,800,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 11] step=5,900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 11] step=6,000,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 11] step=6,100,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 11] step=6,200,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 11] step=6,300,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 11] step=6,400,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 11] step=6,500,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 11] step=6,600,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 11] step=6,700,000/6,714,664  avg_loss≈0.0201
[GloVe] epoch 11/15 done in 105.05s
[GloVe][epoch 12] step=100,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 12] step=200,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 12] step=300,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 12] step=400,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 12] step=500,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=600,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 12] step=700,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 12] step=800,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 12] step=900,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=1,000,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=1,100,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 12] step=1,200,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=1,300,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 12] step=1,400,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=1,500,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 12] step=1,600,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 12] step=1,700,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 12] step=1,800,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 12] step=1,900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 12] step=2,000,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 12] step=2,100,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=2,200,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=2,300,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=2,400,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 12] step=2,500,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 12] step=2,600,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 12] step=2,700,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 12] step=2,800,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 12] step=2,900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 12] step=3,000,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 12] step=3,100,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 12] step=3,200,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=3,300,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=3,400,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=3,500,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 12] step=3,600,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 12] step=3,700,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=3,800,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 12] step=3,900,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 12] step=4,000,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=4,100,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 12] step=4,200,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=4,300,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 12] step=4,400,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 12] step=4,500,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 12] step=4,600,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 12] step=4,700,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=4,800,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=4,900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 12] step=5,000,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 12] step=5,100,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=5,200,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=5,300,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 12] step=5,400,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=5,500,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 12] step=5,600,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 12] step=5,700,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 12] step=5,800,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=5,900,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 12] step=6,000,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 12] step=6,100,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=6,200,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 12] step=6,300,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 12] step=6,400,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 12] step=6,500,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 12] step=6,600,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 12] step=6,700,000/6,714,664  avg_loss≈0.0198
[GloVe] epoch 12/15 done in 102.29s
[GloVe][epoch 13] step=100,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 13] step=200,000/6,714,664  avg_loss≈0.0190
[GloVe][epoch 13] step=300,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 13] step=400,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 13] step=500,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 13] step=600,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=700,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 13] step=800,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 13] step=900,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 13] step=1,000,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=1,100,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 13] step=1,200,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=1,300,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 13] step=1,400,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 13] step=1,500,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 13] step=1,600,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 13] step=1,700,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 13] step=1,800,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 13] step=1,900,000/6,714,664  avg_loss≈0.0190
[GloVe][epoch 13] step=2,000,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 13] step=2,100,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=2,200,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 13] step=2,300,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=2,400,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 13] step=2,500,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 13] step=2,600,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=2,700,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 13] step=2,800,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 13] step=2,900,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 13] step=3,000,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 13] step=3,100,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=3,200,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 13] step=3,300,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 13] step=3,400,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 13] step=3,500,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 13] step=3,600,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 13] step=3,700,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 13] step=3,800,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=3,900,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=4,000,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 13] step=4,100,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 13] step=4,200,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=4,300,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=4,400,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 13] step=4,500,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 13] step=4,600,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 13] step=4,700,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=4,800,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 13] step=4,900,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=5,000,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=5,100,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 13] step=5,200,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 13] step=5,300,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=5,400,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 13] step=5,500,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=5,600,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 13] step=5,700,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=5,800,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=5,900,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 13] step=6,000,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 13] step=6,100,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=6,200,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 13] step=6,300,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 13] step=6,400,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=6,500,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 13] step=6,600,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 13] step=6,700,000/6,714,664  avg_loss≈0.0194
[GloVe] epoch 13/15 done in 100.58s
[GloVe][epoch 14] step=100,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 14] step=200,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 14] step=300,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 14] step=400,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 14] step=500,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 14] step=600,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 14] step=700,000/6,714,664  avg_loss≈0.0186
[GloVe][epoch 14] step=800,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 14] step=900,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 14] step=1,000,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 14] step=1,100,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 14] step=1,200,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 14] step=1,300,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 14] step=1,400,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 14] step=1,500,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=1,600,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 14] step=1,700,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 14] step=1,800,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 14] step=1,900,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 14] step=2,000,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 14] step=2,100,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 14] step=2,200,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=2,300,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 14] step=2,400,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 14] step=2,500,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 14] step=2,600,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 14] step=2,700,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 14] step=2,800,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 14] step=2,900,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 14] step=3,000,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 14] step=3,100,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 14] step=3,200,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 14] step=3,300,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=3,400,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 14] step=3,500,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 14] step=3,600,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 14] step=3,700,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=3,800,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=3,900,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 14] step=4,000,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=4,100,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=4,200,000/6,714,664  avg_loss≈0.0204
[GloVe][epoch 14] step=4,300,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 14] step=4,400,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 14] step=4,500,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 14] step=4,600,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 14] step=4,700,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 14] step=4,800,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 14] step=4,900,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 14] step=5,000,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 14] step=5,100,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 14] step=5,200,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 14] step=5,300,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 14] step=5,400,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 14] step=5,500,000/6,714,664  avg_loss≈0.0189
[GloVe][epoch 14] step=5,600,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 14] step=5,700,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 14] step=5,800,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 14] step=5,900,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 14] step=6,000,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 14] step=6,100,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=6,200,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=6,300,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 14] step=6,400,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 14] step=6,500,000/6,714,664  avg_loss≈0.0189
[GloVe][epoch 14] step=6,600,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 14] step=6,700,000/6,714,664  avg_loss≈0.0196
[GloVe] epoch 14/15 done in 99.87s
[GloVe][epoch 15] step=100,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 15] step=200,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 15] step=300,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 15] step=400,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 15] step=500,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 15] step=600,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 15] step=700,000/6,714,664  avg_loss≈0.0189
[GloVe][epoch 15] step=800,000/6,714,664  avg_loss≈0.0189
[GloVe][epoch 15] step=900,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 15] step=1,000,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 15] step=1,100,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 15] step=1,200,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 15] step=1,300,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 15] step=1,400,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 15] step=1,500,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 15] step=1,600,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 15] step=1,700,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 15] step=1,800,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 15] step=1,900,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 15] step=2,000,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 15] step=2,100,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 15] step=2,200,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 15] step=2,300,000/6,714,664  avg_loss≈0.0189
[GloVe][epoch 15] step=2,400,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 15] step=2,500,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 15] step=2,600,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 15] step=2,700,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 15] step=2,800,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 15] step=2,900,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 15] step=3,000,000/6,714,664  avg_loss≈0.0198
[GloVe][epoch 15] step=3,100,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 15] step=3,200,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 15] step=3,300,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 15] step=3,400,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 15] step=3,500,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 15] step=3,600,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 15] step=3,700,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 15] step=3,800,000/6,714,664  avg_loss≈0.0192
[GloVe][epoch 15] step=3,900,000/6,714,664  avg_loss≈0.0190
[GloVe][epoch 15] step=4,000,000/6,714,664  avg_loss≈0.0189
[GloVe][epoch 15] step=4,100,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 15] step=4,200,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 15] step=4,300,000/6,714,664  avg_loss≈0.0190
[GloVe][epoch 15] step=4,400,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 15] step=4,500,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 15] step=4,600,000/6,714,664  avg_loss≈0.0201
[GloVe][epoch 15] step=4,700,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 15] step=4,800,000/6,714,664  avg_loss≈0.0195
[GloVe][epoch 15] step=4,900,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 15] step=5,000,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 15] step=5,100,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 15] step=5,200,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 15] step=5,300,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 15] step=5,400,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 15] step=5,500,000/6,714,664  avg_loss≈0.0200
[GloVe][epoch 15] step=5,600,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 15] step=5,700,000/6,714,664  avg_loss≈0.0203
[GloVe][epoch 15] step=5,800,000/6,714,664  avg_loss≈0.0197
[GloVe][epoch 15] step=5,900,000/6,714,664  avg_loss≈0.0207
[GloVe][epoch 15] step=6,000,000/6,714,664  avg_loss≈0.0193
[GloVe][epoch 15] step=6,100,000/6,714,664  avg_loss≈0.0196
[GloVe][epoch 15] step=6,200,000/6,714,664  avg_loss≈0.0199
[GloVe][epoch 15] step=6,300,000/6,714,664  avg_loss≈0.0188
[GloVe][epoch 15] step=6,400,000/6,714,664  avg_loss≈0.0191
[GloVe][epoch 15] step=6,500,000/6,714,664  avg_loss≈0.0194
[GloVe][epoch 15] step=6,600,000/6,714,664  avg_loss≈0.0202
[GloVe][epoch 15] step=6,700,000/6,714,664  avg_loss≈0.0194
[GloVe] epoch 15/15 done in 100.04s
[Train] Done in 1577.62s; final loss (relative, last-chunk): 286.94; Average loss per token: 0.0002622793947498343
[Vocab] Top-10: ['time', 'game', 'city', 'song', 'year', 'made', 'season', 'war', 'north', 'used']
[Vocab] Size: 27091
[Plot saved to plots/Glove\plot_001.png]
[Plot saved to plots/Glove\plot_002.png]
[Plot saved to plots/Glove\plot_003.png]
[Plot saved to plots/Glove\plot_004.png]
[Plot saved to plots/Glove\plot_005.png]
[Plot saved to plots/Glove\plot_006.png]
[Neighbors] Probes: 30 over 5 buckets.

[tokyo]
  ben                  0.990
  motor                0.990
  wright               0.990
  del                  0.990
  kevin                0.990
  jeff                 0.990
  succeed              0.990
  grass                0.990
  ron                  0.990
  ford                 0.989

[berlin]
  del                  0.990
  jeff                 0.989
  pete                 0.989
  marcus               0.989
  eric                 0.989
  ed                   0.989
  randy                0.989
  zero                 0.989
  larry                0.989
  greg                 0.989

[frankfurt]
  greg                 0.975
  rowland              0.975
  keith                0.975
  ed                   0.975
  eric                 0.975
  hunter               0.974
  dean                 0.974
  wright               0.974
  del                  0.974
  gwen                 0.974

[athens]
  kevin                0.981
  benjamin             0.981
  seal                 0.981
  sam                  0.981
  pack                 0.980
  jim                  0.980
  internal             0.980
  ben                  0.980
  julian               0.980
  lee                  0.980

[lausanne]
  rays                 0.847
  faith                0.845
  borders              0.845
  tim                  0.844
  rodriguez            0.844
  canon                0.844
  orange               0.844
  panama               0.844
  guidance             0.844
  col                  0.844

[beijing]
  sur                  0.981
  troop                0.980
  corn                 0.980
  stephen              0.980
  pace                 0.980
  flew                 0.980
  planes               0.980
  lowered              0.980
  hamilton             0.980
  ram                  0.980

[physics]
  santa                0.985
  shop                 0.985
  barbara              0.985
  mental               0.985
  matt                 0.985
  seal                 0.985
  external             0.985
  iv                   0.985
  sam                  0.985
  max                  0.985

[chemistry]
  del                  0.991
  stewart              0.991
  matt                 0.991
  greg                 0.991
  jeff                 0.991
  mart                 0.991
  ron                  0.991
  santa                0.991
  randy                0.991
  iv                   0.991

[thesis]
  del                  0.988
  ben                  0.988
  kevin                0.987
  dr                   0.987
  iv                   0.987
  barry                0.987
  greg                 0.987
  solely               0.987
  explain              0.987
  dean                 0.987

[biology]
  ian                  0.966
  sam                  0.966
  bobby                0.965
  rick                 0.965
  stock                0.965
  mental               0.965
  burton               0.965
  santa                0.965
  shaolin              0.965
  der                  0.965

[doctoral]
  ben                  0.959
  keith                0.958
  charlotte            0.958
  del                  0.958
  larry                0.958
  drummer              0.958
  jackie               0.958
  charlie              0.958
  kevin                0.958
  ex                   0.958

[sociology]
  seal                 0.988
  jeff                 0.988
  iv                   0.988
  eric                 0.987
  wright               0.987
  ed                   0.987
  motor                0.987
  ron                  0.987
  pleasure             0.987
  greg                 0.987

[duke]
  commando             0.949
  montana              0.949
  leonard              0.948
  rocky                0.947
  wellington           0.946
  detroit              0.946
  hampshire            0.945
  robinson             0.945
  lesser               0.944
  broadcasting         0.943

[prince]
  au                   0.959
  warren               0.934
  canyon               0.933
  robinson             0.931
  adjacent             0.931
  lesser               0.929
  green                0.929
  sweep                0.929
  palm                 0.928
  detroit              0.928

[duchess]
  pack                 0.971
  plus                 0.970
  max                  0.970
  karl                 0.970
  von                  0.970
  bound                0.970
  obtain               0.970
  mrs                  0.969
  jones                0.969
  der                  0.969

[empress]
  coffee               0.978
  matt                 0.978
  franklin             0.977
  alan                 0.977
  narrowly             0.977
  lee                  0.977
  carriage             0.977
  raising              0.977
  sufficient           0.977
  sam                  0.977

[princess]
  jim                  0.991
  del                  0.991
  kevin                0.991
  moore                0.991
  larry                0.991
  sam                  0.990
  williams             0.990
  ben                  0.990
  motor                0.990
  ed                   0.990

[queen]
  ward                 0.981
  norway               0.981
  garden               0.980
  jimmy                0.980
  lesser               0.979
  detroit              0.978
  rocky                0.978
  pete                 0.978
  jerry                0.978
  fighter              0.978

[yankees]
  england              0.987
  defeat               0.987
  defeated             0.986
  manager              0.985
  conference           0.985
  york                 0.984
  against              0.984
  draw                 0.984
  federer              0.984
  scotland             0.983

[saves]
  record               0.989
  era                  0.989
  consecutive          0.986
  wins                 0.986
  jordan               0.985
  final                0.985
  fourth               0.985
  sixth                0.985
  third                0.985
  points               0.984
[Neighbors] Category-aware purity@10 = 0.010
[WordSim] n=318 | Spearman=0.078 | Pearson=0.080
[WordSim] n=909 | Spearman=0.063 | Pearson=0.076
[Analogy] 3cosadd Acc@1 = 0.000 (n=4)
[Analogy] 3cosmul Acc@1 = 0.000 (n=4)
[20NG-CLS] Acc=0.477, Macro-F1=0.468
[20NG-CLU] ARI=0.034, NMI=0.043
```

![](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_001.png)

![plot_002](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_002.png)

![plot_003](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_003.png)

![plot_004](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_004.png)

![plot_005](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_005.png)

![plot_006](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_006.png)

#### 12.2 Codes

##### 12.2.1 Word2Vec(Skip-Gram)

```python
# Assignment1 for DSA4213
# A0329409A Zhang Jingxuan
# Word2Vec (Skip-gram)

# Imports & Global Settings
import json
import os
import random
import re
import time
import sys

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

plt.ioff()

# Redirect print output
class Tee(object):
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = Tee(sys.stdout, open("Outputs_SGNS.txt", "w", encoding="utf-8"))

# Save all plots
save_dir = "plots/Word2Vec"
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
plot_2d(XYp, wordsM, labelsM, "SGNS — PCA (Manual Buckets)", True)

# t-SNE
XYt = TSNE(n_components=2, perplexity=20, learning_rate=200, max_iter=2000, random_state=SEED, init="pca").fit_transform(XM)
plot_2d(XYt, wordsM, labelsM, "SGNS — t-SNE (Manual Buckets)", True)

# UMAP
reducer = umap.UMAP(
    n_components=2,
    random_state=SEED,
    n_neighbors=15,
    min_dist=0.1,
    n_jobs=1
)
XYu = reducer.fit_transform(XM)
plot_2d(XYu, wordsM, labelsM, "SGNS — UMAP (Manual Buckets)", True)

# Seed-Expanded Buckets
# PCA
auto_buckets = seed_expand_buckets(model, SEED_BUCKETS, per_seed=25)
wordsA, labelsA, XA = collect_bucket_vectors(model, auto_buckets)
XYa = PCA(n_components=2, random_state=SEED).fit_transform(XA)
plot_2d(XYa, wordsA, labelsA, "SGNS — PCA (Seed-Expanded Buckets)", False)

# t-SNE
XYtA = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1500,
            random_state=SEED, init="pca").fit_transform(XA)
plot_2d(XYtA, wordsA, labelsA, "SGNS — t-SNE (Seed-Expanded Buckets)", False)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=25, min_dist=0.1, n_jobs=1)
XYuA = reducer.fit_transform(XA)
plot_2d(XYuA, wordsA, labelsA, "SGNS — UMAP (Seed-Expanded Buckets)", False)

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
```

##### 12.2.3 SPPMI=SVD

```python
# Assignment1 for DSA4213
# A0329409A Zhang Jingxuan
# SPPMI-SVD

# Imports & Global Settings
import json
import os
import random
import re
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score
from collections import Counter, defaultdict
from math import log
from scipy import sparse
from sklearn.decomposition  import TruncatedSVD
from gensim.models.keyedvectors import KeyedVectors

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
sys.stdout = Tee(sys.stdout, open("Outputs_SPPMI-SVD.txt", "w", encoding="utf-8"))

# Save all plots
save_dir = "plots/SPPMI-SVD"
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
def build_vocab_from_sentences(sentences, min_count=2):
    freq = Counter(w for s in sentences for w in s)
    vocab = [w for w, c in freq.items() if c >= min_count]
    vocab.sort(key=lambda w: (-freq[w], w))
    w2i = {w:i for i, w in enumerate(vocab)}
    return w2i, vocab, freq

def build_cooc(sentences, w2i, window = 10, subsample_t = 1e-3, rng = None):
    if rng is None:
        rng = np.random.RandomState(4213)

    keep = None
    if subsample_t is not None:
        total = sum(len(s) for s in sentences)
        freqs = np.zeros(len(w2i), dtype=np.float64)
        for s in sentences:
            for w in s:
                if w in w2i:
                    freqs[w2i[w]] += 1
        freqs /= max(total, 1)
        p_keep = np.minimum(1.0, (np.sqrt(freqs / subsample_t) + 1.0) * (subsample_t / np.maximum(freqs, 1e-12)))
        keep = p_keep

    rows, cols, data = [], [], []
    for s in sentences:
        idx = [w2i[w] for w in s if w in w2i]
        if keep is None:
            idx = [i for i in idx if rng.random() < keep[i]]
        n = len(idx)
        for i in range(n):
            wi = idx[i]
            win = rng.integers(1, window + 1)
            lo, hi = max(0, i - win), min(i + win + 1, n)
            for j in range(lo, hi):
                if j == i:
                    continue
                wj = idx[j]
                d = abs(j - i)
                weight = 1.0 / d
                rows.append(wi)
                cols.append(wj)
                data.append(weight)

    X = sparse.coo_matrix((data, (rows, cols)), shape=(len(w2i), len(w2i)), dtype=np.float64).tocsr()
    X = X + X.T
    return X

def sppmi_matrix(X, shift_k=5.0):
    row_sum = np.asarray(X.sum(axis=1)).ravel()
    col_sum = np.asarray(X.sum(axis=0)).ravel()
    N = row_sum.sum()
    alpha = 0.75
    col_sum_smooth = col_sum ** alpha

    Xcoo = X.tocoo()
    pmi_vals = []
    log_k = log(shift_k)
    for i, j, x in zip(Xcoo.row, Xcoo.col, Xcoo.data):
        denom = row_sum[i] * col_sum_smooth[j]
        if denom <= 0 or x <= 0:
            pmi_vals.append(0.0)
        else:
            val = log((x * N) / denom) - log_k
            pmi_vals.append(val if val > 0 else 0.0)
    SPPMI = sparse.coo_matrix((pmi_vals, (Xcoo.row, Xcoo.col)), shape=X.shape).tocsr()
    return SPPMI

def train_sppmi_svd(sentences, vector_size=300, window=10, min_count=2, subsample_t=1e-3, shift_k=5.0, svd_power=0.5, seed=4213):
    w2i, vocab, freq = build_vocab_from_sentences(sentences, min_count=min_count)
    print(f"[SPPMI] |V| after min_count={min_count}: {len(vocab)}")

    t0 = time.time()
    X = build_cooc(sentences, w2i, window=window, subsample_t=subsample_t, rng=np.random.default_rng(seed))
    print(f"[SPPMI] Cooc built: nnz={X.nnz:,} in {time.time() - t0:.2f}s")

    t1 = time.time()
    M = sppmi_matrix(X, shift_k=shift_k)
    print(f"[SPPMI] SPPMI built: nnz={M.nnz:,} in {time.time() - t1:.2f}s")

    svd = TruncatedSVD(n_components=vector_size, n_iter=10, random_state=seed)
    t2 = time.time()
    U = svd.fit_transform(M)
    S = svd.singular_values_
    print(f"[SPPMI] SVD done in {time.time() - t2:.2f}s; explained_var={svd.explained_variance_ratio_.sum():.3f}")

    if svd_power != 0:
        U = U * (S ** svd_power)

    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)

    kv = KeyedVectors(vector_size)
    kv.add_vectors(vocab, U.astype(np.float32))

    class _Wrapper:
        def __init__(self, kv):
            self.wv = kv
            self.vector_size = kv.vector_size

    return _Wrapper(kv)

sppmi_cfg = dict(
    sentences=sentences,
    vector_size=300,
    window=10,
    min_count=2,
    subsample_t=1e-3,
    shift_k=15.0,
    svd_power=0.5,
    seed=SEED
)
print("[Train] SPPMI-SVD config:", json.dumps({k:v for k,v in sppmi_cfg.items() if k!='sentences'}, indent=2))
t1 = time.time()
model = train_sppmi_svd(**sppmi_cfg)
print(f"[Train] Done in {time.time()-t1:.2f}s (SPPMI-SVD)")

os.makedirs("artifacts", exist_ok=True)
model.wv.save_word2vec_format("artifacts/sppmi_svd_wikitext2.txt")
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
plot_2d(XYp, wordsM, labelsM, "SPPMI_SVD — PCA (Manual Buckets)", True)

# t-SNE
XYt = TSNE(n_components=2, perplexity=20, learning_rate=200, max_iter=2000, random_state=SEED, init="pca").fit_transform(XM)
plot_2d(XYt, wordsM, labelsM, "SPPMI_SVD — t-SNE (Manual Buckets)", True)

# UMAP
reducer = umap.UMAP(
    n_components=2,
    random_state=SEED,
    n_neighbors=15,
    min_dist=0.1,
    n_jobs=1
)
XYu = reducer.fit_transform(XM)
plot_2d(XYu, wordsM, labelsM, "SPPMI_SVD — UMAP (Manual Buckets)", True)

# Seed-Expanded Buckets
# PCA
auto_buckets = seed_expand_buckets(model, SEED_BUCKETS, per_seed=25)
wordsA, labelsA, XA = collect_bucket_vectors(model, auto_buckets)
XYa = PCA(n_components=2, random_state=SEED).fit_transform(XA)
plot_2d(XYa, wordsA, labelsA, "SPPMI_SVD — PCA (Seed-Expanded Buckets)", False)

# t-SNE
XYtA = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1500,
            random_state=SEED, init="pca").fit_transform(XA)
plot_2d(XYtA, wordsA, labelsA, "SPPMI_SVD — t-SNE (Seed-Expanded Buckets)", False)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=25, min_dist=0.1, n_jobs=1)
XYuA = reducer.fit_transform(XA)
plot_2d(XYuA, wordsA, labelsA, "SPPMI_SVD — UMAP (Seed-Expanded Buckets)", False)

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
```

##### 12.2.3 GloVe

```python
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
```

#### 12.3 Code repository address

https://github.com/Yo1ogreyZz/DSA4213_Natural-Language-Processing-for-Data-Science.git
