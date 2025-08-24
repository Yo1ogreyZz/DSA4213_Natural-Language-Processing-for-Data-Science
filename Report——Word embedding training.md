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

#### 12.1 Training result plot:

##### SGNS:

![](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_001.png)

![plot_002](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_002.png)

![plot_003](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_003.png)

![plot_004](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_004.png)

![plot_005](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_005.png)

![plot_006](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Word2Vec\plot_006.png)

##### SPPMI-SVD:

![](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_001.png)

![plot_002](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_002.png)

![plot_003](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_003.png)

![plot_004](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_004.png)

![plot_005](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_005.png)

![plot_006](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\SPPMI-SVD\plot_006.png)

##### GloVe:

![](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_001.png)

![plot_002](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_002.png)

![plot_003](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_003.png)

![plot_004](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_004.png)

![plot_005](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_005.png)

![plot_006](D:\Coding_Nus\DSA4213\DSA4213_Natural-Language-Processing-for-Data-Science\plots\Glove\plot_006.png)

#### 12.2 Code repository address

https://github.com/Yo1ogreyZz/DSA4213_Natural-Language-Processing-for-Data-Science.git