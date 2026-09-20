# Phase 6 — Unsupervised Learning & Anomaly Detection

> **Target Duration**: 2–3 Weeks  
> **Prerequisites**: Phase 1 (Linear Algebra & SVD), Phase 3 (scikit-learn Pipelines), and Phase 4 (Algorithms).  
> **Key Goal**: Uncover hidden structures in unlabeled data. Master clustering algorithms, geometric dimensionality reduction (PCA, t-SNE, UMAP), and unsupervised anomaly detection algorithms (Isolation Forest, LOF) critical for threat intelligence and cybersecurity.

---

## 1. Clustering Paradigms: Partitioning vs. Density vs. Hierarchy

```text
                               CLUSTERING METHODS
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
    PARTITIONING                    DENSITY                     HIERARCHICAL
      (K-Means)                    (DBSCAN)                   (Agglomerative)
  - Spherical clusters         - Arbitrary shapes           - Tree-like dendrogram
  - Pre-fixed K                - Finds noise/outliers       - No pre-fixed K
  - Fast O(N * K * iter)       - O(N * log N) to O(N^2)     - O(N^2) to O(N^3)
```

### 1.1 K-Means & Cluster Validation

- **Objective Function (Inertia / Within-Cluster Sum of Squares - WCSS)**:
  $$\text{WCSS} = \sum_{k=1}^K \sum_{\mathbf{x}_i \in S_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$
- **Finding Optimal $K$**:
  1. **Elbow Method**: Plot WCSS vs. $K$; look for the point of diminishing returns (elbow).
  2. **Silhouette Coefficient**:
     $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, +1]$$
     - $a(i)$: Mean intra-cluster distance.
     - $b(i)$: Mean nearest-cluster distance.
     - Value near $+1$ indicates well-separated, compact clusters.

---

### 1.2 DBSCAN for Network Threat Clustering

DBSCAN is exceptionally well-suited for network log and telemetry analysis because it does not force anomalous outliers into artificial clusters.

- **Parameters**:
  - `eps` ($\epsilon$): Radius of neighborhood.
  - `min_samples`: Minimum number of samples required to form a dense region.
- **Output Labels**: Normal clusters assigned non-negative integers (`0, 1, 2, ...`); anomalies and port-scan probes are assigned **`-1` (Noise)**.

---

## 2. Dimensionality Reduction: PCA, t-SNE, and UMAP

### 2.1 Principal Component Analysis (PCA)

- **Linear Projection**: Finds orthogonal axes that maximize data variance.
- **Singular Value Decomposition (SVD)**:
  $$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
  - Columns of $\mathbf{V}$ are eigenvectors (principal components).
  - Squared diagonal entries of $\mathbf{\Sigma}$ represent explained variance.
- **Production Use Cases**:
  - Compressing high-dimensional features (e.g., 500 network features $\to$ 20 components) before training models.
  - Decorrelating multicollinear features.
  - Fast 2D/3D visualization of system telemetry.

### 2.2 t-SNE vs. UMAP (Non-linear Manifold Learning)

| Feature | t-SNE | UMAP |
| :--- | :--- | :--- |
| **Mathematical Basis** | Minimizes KL divergence using Student-t distribution. | Fuzzy simplicial sets & Riemannian geometry. |
| **Global Structure** | Preserves local neighborhoods; distorts global distances. | Preserves both local and global cluster relationships. |
| **Speed & Scalability** | Slow ($O(N^2)$ or $O(N \log N)$ Barnes-Hut). | Very fast; scales to millions of points. |
| **Out-of-Sample Transform** | **No**: Cannot call `.transform()` on new incoming data! | **Yes**: Can embed new unseen points into existing manifold. |
| **Primary Use Case** | Visualizing latent embeddings in 2D plots. | Feature preprocessing for clustering & visualization. |

---

## 3. Unsupervised Anomaly Detection

### 3.1 Isolation Forest (The Gold Standard)

Instead of modeling normal instances (like density estimators), **Isolation Forest explicitly isolates anomalies**.

```text
               NORMAL INSTANCE                         ANOMALOUS INSTANCE
          (Deep in the dense cluster)               (Isolated in sparse region)
                 [ Root ]                                   [ Root ]
                /        \                                 /        \
              ...        ...                         [ Isolated! ]   ...
             /                                      (Path length h(x) = 1)
       [ Isolated! ]
  (Path length h(x) = 12)
```

- **Core Insight**: Anomalies are "few and different". In a random binary tree, anomalous points are isolated with very few random splits (short path length $h(\mathbf{x})$).
- **Anomaly Score**:
  $$s(\mathbf{x}, N) = 2^{-\frac{\mathbb{E}[h(\mathbf{x})]}{c(N)}}$$
  - $c(N)$: Average path length of unsuccessful searches in a Binary Search Tree.
  - $s \to 1$: Highly likely anomaly.
  - $s < 0.5$: Normal instance.

---

## 4. End-to-End Implementation: Telemetry Threat Detection

Below is an end-to-end Python pipeline using **Isolation Forest** and **PCA** on multidimensional network behavior telemetry:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Simulate 10,000 normal network connection events
np.random.seed(42)
N_normal = 10_000
normal_data = np.random.multivariate_normal(
    mean=[50, 1000, 20],
    cov=[[10, 5, 2], [5, 50, 10], [2, 10, 5]],
    size=N_normal
)

# Simulate 100 DDoS/Port-Scan attacks (Extreme Outliers)
N_attack = 100
attack_data = np.random.uniform(
    low=[150, 10000, 500],
    high=[300, 50000, 2000],
    size=(N_attack, 3)
)

X_raw = np.vstack([normal_data, attack_data])
feature_names = ["syn_rate", "bytes_transferred", "open_ports"]

# 2. Preprocess features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Fit Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.01,  # Expect ~1% anomalies
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)

# Predictions: 1 = normal, -1 = anomaly
preds = iso_forest.predict(X_scaled)
scores = iso_forest.decision_function(X_scaled)  # Lower score = more abnormal

# 4. Dimensionality Reduction with PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA Explained Variance: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
print(f"Total Anomalies Flagged: {(preds == -1).sum()} out of {len(X_raw)} events")

# Check top 5 most anomalous events
anomalous_indices = np.argsort(scores)[:5]
print("\nTop 5 Most Severe Anomalies:")
for idx in anomalous_indices:
    print(f"Sample {idx}: Features={X_raw[idx].round(1)}, Score={scores[idx]:.4f}")
```

---

## 5. Summary & Self-Check

1. **Why is scaling features strictly mandatory before running K-Means or PCA?**
   - *Answer*: Both algorithms depend on Euclidean distances. If one feature is measured in bytes ($10^6$) and another in seconds ($10^1$), the byte feature will completely dominate the distance computation by orders of magnitude.
2. **Why can't you use t-SNE in a live production API to project incoming user requests?**
   - *Answer*: t-SNE optimizes an objective function directly over the coordinates of the input batch using gradient descent; it does not learn a parametric mapping function $f(\mathbf{x})$. To embed a single new point, you would have to re-optimize all coordinates from scratch.
3. **What is the difference between One-Class SVM and Isolation Forest?**
   - *Answer*: One-Class SVM fits a boundary around the densest region in kernel space; it scales poorly ($O(N^2 \text{ to } N^3)$) and is sensitive to hyperparameter tuning. Isolation Forest builds tree partitions with $O(N \log N)$ complexity, making it vastly faster and more robust on large-scale production datasets.

➡️ **Next Phase**: [07_ml_projects.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/07_ml_projects.md)
