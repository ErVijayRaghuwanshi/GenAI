# Phase 4 — Core Machine Learning Algorithms (Mechanics, Assumptions & Code)

> **Target Duration**: 6–8 Weeks  
> **Prerequisites**: Phase 1 (Math), Phase 2 (Fundamentals), Phase 3 (scikit-learn Pipelines).  
> **Key Goal**: Dissect the 13 fundamental machine learning algorithms. For each algorithm, understand:
> 1. What problem does it solve?
> 2. How does it work mathematically?
> 3. What assumptions does it make?
> 4. What are its critical hyperparameters?
> 5. When does it fail?
> 6. How do you evaluate it?

---

## The 6-Question Algorithm Evaluation Framework

Before deploying any ML algorithm into a backend service, evaluate it against this checklist:

```text
 1. PROBLEM        Regression, Classification, Clustering, or Density Estimation?
 2. MECHANISM      Parametric closed-form vs Iterative gradient vs Partitioning tree?
 3. ASSUMPTIONS    Linearity, normality, independence, spherical clusters?
 4. PARAMETERS     Regularization (C, alpha), tree depth (max_depth), neighbors (k)?
 5. FAILURE MODES  Curse of dimensionality, outliers, multicollinearity, imbalance?
 6. COMPUTATION    Training complexity O(?) vs Inference latency O(?)?
```

---

## 🟢 LEVEL 1: Foundational Linear & Instance Models

### 1. Linear Regression (OLS, Ridge, Lasso, ElasticNet)

- **Problem**: Predicts a continuous outcome $y \in \mathbb{R}$ as a linear combination of features $\mathbf{x}$.
- **Math & Mechanics**:
  - Model: $\hat{y} = \mathbf{w}^T \mathbf{x} + b$
  - Ordinary Least Squares (OLS) Loss:
    $$J_{\text{OLS}}(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$
  - Closed-form Normal Equation: $\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
  - **Ridge ($L_2$ Regularization)**: Adds $\frac{\lambda}{2} \|\mathbf{w}\|_2^2$. Prevents exploding weights when features are collinear.
  - **Lasso ($L_1$ Regularization)**: Adds $\lambda \|\mathbf{w}\|_1$. Produces exact zeros, performing automatic feature selection.
  - **ElasticNet**: Blends $L_1$ and $L_2$: $\lambda_1 \|\mathbf{w}\|_1 + \frac{\lambda_2}{2} \|\mathbf{w}\|_2^2$.
- **Assumptions**: Linear relationship between features and target, independent errors (no autocorrelation), homoscedasticity (constant variance of residuals), normally distributed residuals.
- **Critical Hyperparameters**: `alpha` (regularization strength $\lambda$), `l1_ratio` (ElasticNet balance).
- **When It Fails**: Non-linear relationships (e.g., sine waves, decision boundaries), extreme outliers (OLS squares errors, heavily skewing line).

---

### 2. Logistic Regression

- **Problem**: Estimates class probabilities $P(y=1 \mid \mathbf{x})$ for binary and multi-class classification.
- **Math & Mechanics**:
  - Maps linear logits $z = \mathbf{w}^T \mathbf{x} + b$ into probabilities using the **Sigmoid** function:
    $$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$
  - Loss Function: **Binary Cross-Entropy (Log-Loss)**:
    $$J(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
  - Gradient vector for optimization:
    $$\nabla_{\mathbf{w}} J = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{p}} - \mathbf{y})$$

#### From-Scratch Pure NumPy Implementation

```python
import numpy as np

class ScratchLogisticRegression:
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_pred = X @ self.weights + self.bias
            y_pred = self._sigmoid(linear_pred)

            # Gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Parameter updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
```

- **Assumptions**: Log-odds of target is linear with features, observations are independent, minimal multicollinearity.
- **Hyperparameters**: `C` (inverse regularization strength $1/\lambda$), `penalty` ('l1', 'l2', 'elasticnet'), `class_weight` ('balanced').

---

### 3. $k$-Nearest Neighbors ($k$-NN)

- **Problem**: Non-parametric classification or regression based on spatial proximity.
- **Math & Mechanics**:
  - Given query point $\mathbf{x}_q$, compute distance metric (Euclidean $L_2$, Manhattan $L_1$, or Cosine) to all $N$ training samples.
  - Find the $k$ nearest neighbors: $\mathcal{N}_k(\mathbf{x}_q)$.
  - Predict by majority vote (classification) or mean value (regression):
    $$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x}_q)} y_i$$
- **Assumptions**: Points close to each other in feature space share similar target labels.
- **Critical Hyperparameters**: `n_neighbors` ($k$), `weights` ('uniform' vs 'distance'), `metric` ('minkowski', 'cosine').
- **Failure Modes & Production Warning**:
  - **Curse of Dimensionality**: As dimensions $D > 50$, all points become almost equidistant in Euclidean space.
  - **Inference Latency**: $O(N \cdot D)$ per prediction. Unusable for real-time low-latency serving on massive datasets without vector indexers (e.g., FAISS, HNSW).

---

### 4. Naive Bayes

- **Problem**: High-speed probabilistic classification (spam filtering, text classification, threat log analysis).
- **Math & Mechanics**:
  - Uses Bayes' Theorem under the **"Naive" conditional independence assumption**:
    $$P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^D P(x_j \mid y)$$
  - Variants:
    - **GaussianNB**: Continuous features modeled as Gaussian distributions $\mathcal{N}(\mu_{c, j}, \sigma_{c, j}^2)$.
    - **MultinomialNB**: Discrete word/event count frequencies.
    - **BernoulliNB**: Binary indicator features (e.g., word present or absent).
- **Assumptions**: All features are conditionally independent given the class label (rarely true in reality, but often works surprisingly well).
- **Critical Hyperparameters**: `var_smoothing` (GaussianNB), `alpha` (Laplace smoothing parameter for MultinomialNB).

---

## 🟡 LEVEL 2: Tree-Based Models & Ensembles

```text
               DECISION TREES & ENSEMBLE EVOLUTION
               
       Single Decision Tree (High Variance, Overfits)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
      BAGGING (Parallel)             BOOSTING (Sequential)
    Random Forest:                  Gradient Boosted Trees / XGBoost:
    - Independent trees             - Trees trained on residual errors
    - Feature subsampling           - Trees correct predecessor mistakes
    - Reduces Variance              - Reduces Bias and Variance
```

### 5. Decision Trees (CART)

- **Problem**: Non-linear classification and regression via recursive binary splitting.
- **Math & Mechanics**:
  - Iteratively finds feature $j$ and split threshold $t$ that maximizes impurity reduction.
  - **Gini Impurity** (Classification):
    $$I_G(S) = 1 - \sum_{k=1}^C p_k^2$$
  - **Entropy / Information Gain**:
    $$H(S) = - \sum_{k=1}^C p_k \log_2(p_k)$$
  - **Variance Reduction** (Regression): Minimizes $\sum_{i \in \text{Left}} (y_i - \bar{y}_L)^2 + \sum_{i \in \text{Right}} (y_i - \bar{y}_R)^2$.
- **Hyperparameters**: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- **When It Fails**: Unpruned trees easily memorize noise, creating deep trees with high variance that fail on unseen data.

---

### 6. Random Forest (Bagging)

- **Problem**: Ensembles many decorrelated decision trees to radically lower variance.
- **Math & Mechanics**:
  1. **Bootstrap Aggregating (Bagging)**: Draw $B$ random samples of size $N$ with replacement from the training set.
  2. **Feature Subsampling**: At each node split, randomly select a subset of features (typically $\sqrt{D}$ for classification, $D/3$ for regression).
  3. **Aggregation**: Average predictions across all trees (or majority vote).
- **Key Property: Out-of-Bag (OOB) Error**: ~36.8% of samples are omitted in each bootstrap fold; they can be used for zero-cost validation.
- **Hyperparameters**: `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`, `n_jobs=-1`.

---

### 7. Gradient Boosting Decision Trees (GBDT)

- **Problem**: Sequential ensemble that converts weak learners (shallow trees) into a strong learner.
- **Math & Mechanics**:
  - Instead of training trees independently, each new tree fits the **pseudo-residuals** (negative gradients of the loss function) of the existing ensemble:
    $$r_{i, m} = -\left[ \frac{\partial L(y_i, f(\mathbf{x}_i))}{\partial f(\mathbf{x}_i)} \right]_{f = f_{m-1}}$$
  - Model update with shrinkage (learning rate $\eta$):
    $$f_m(\mathbf{x}) = f_{m-1}(\mathbf{x}) + \eta \sum_j \gamma_{j, m} I(\mathbf{x} \in R_{j, m})$$

---

### 8. XGBoost (Extreme Gradient Boosting)

- **Why it dominates production tabular ML**:
  1. **Second-Order Taylor Approximation**: Uses both first gradients $g_i$ and second derivatives (Hessians) $h_i$ of the loss for precise split optimization.
     $$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
  2. **Tree Regularization**: Directly penalizes leaf count $T$ and leaf weights $w$:
     $$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
  3. **Hardware Optimizations**: Cache-aware block structures, out-of-core computing, automatic missing value handling.
- **Critical Hyperparameters**:
  - `learning_rate` ($\eta$): Step size (0.01 - 0.1).
  - `max_depth`: Tree depth (typically 3–8).
  - `subsample`: Row sampling fraction (e.g., 0.8).
  - `colsample_bytree`: Feature subsampling fraction.
  - `scale_pos_weight`: Ratio of negative/positive instances (essential for fraud/cyber intrusion datasets).

---

## 🔴 LEVEL 3: Kernel Methods, Manifolds & Unsupervised Geometries

### 9. Support Vector Machines (SVM)

- **Problem**: Finds the optimal hyperplane that maximizes the margin (distance to closest training points, called **Support Vectors**).
- **Math & Mechanics**:
  - Optimization problem:
    $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to } y_i(\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i$$
  - **The Kernel Trick**: Computes inner products in an infinite-dimensional feature space without ever explicitly computing coordinate mapping $\phi(\mathbf{x})$:
    $$K(\mathbf{x}, \mathbf{z}) = \exp\left( -\gamma \|\mathbf{x} - \mathbf{z}\|^2 \right) \quad (\text{Radial Basis Function / RBF})$$
- **Hyperparameters**: `C` (penalty for misclassifications), `kernel` ('linear', 'rbf', 'poly'), `gamma` (spread of RBF kernel).

---

### 10. Principal Component Analysis (PCA)

- **Problem**: Unsupervised linear dimensionality reduction; removes feature correlation and extracts dominant axes of variance.
- **Math & Mechanics**:
  - Centers data $\mathbf{X}$ ($\mu = 0$).
  - Computes Singular Value Decomposition (SVD): $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$.
  - Right singular vectors $\mathbf{V}$ are the principal component directions (eigenvectors of $\mathbf{X}^T \mathbf{X}$).
  - Projects data onto top $k$ components: $\mathbf{Z} = \mathbf{X} \mathbf{V}_k$.

---

### 11. $K$-Means Clustering

- **Problem**: Partitions $N$ observations into $K$ spherical clusters.
- **Algorithm (Lloyd's Algorithm)**:
  1. Initialize $K$ centroids randomly (or via **$K$-Means++** for spread).
  2. **Assignment step**: Assign each point $\mathbf{x}_i$ to nearest centroid:
     $$c_i = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$
  3. **Update step**: Recompute centroids as mean of assigned points:
     $$\boldsymbol{\mu}_k = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{x}_i$$
  4. Repeat until convergence.
- **Hyperparameters**: `n_clusters` ($K$), `init` ('k-means++'), `n_init` (restarts).
- **Failure Mode**: Assumes spherical, equal-variance clusters. Fails on non-convex or elongated manifolds.

---

### 12. DBSCAN (Density-Based Spatial Clustering)

- **Problem**: Discovers arbitrary-shaped clusters and identifies explicit noise/outlier points.
- **Math & Mechanics**:
  - Finds points with at least `min_samples` within distance `eps` ($\epsilon$).
  - Points classified as:
    - **Core Point**: Has $\ge \text{min\_samples}$ within $\epsilon$.
    - **Border Point**: Within $\epsilon$ of a core point, but fewer than `min_samples` neighbors.
    - **Noise Point**: Neither core nor border (direct anomaly flag!).
- **Hyperparameters**: `eps` ($\epsilon$ radius), `min_samples`.
- **Advantage over $K$-Means**: Does not require specifying $K$ upfront; robustly isolates anomaly noise.

---

### 13. Gaussian Mixture Models (GMM)

- **Problem**: Probabilistic soft clustering; represents data as a mixture of $K$ multivariate Gaussian distributions.
- **Math & Mechanics**:
  $$P(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
  - Optimized via **Expectation-Maximization (EM)** algorithm:
    - **E-step**: Calculate posterior probability that point $i$ belongs to cluster $k$.
    - **M-step**: Update mixture weights $\pi_k$, means $\boldsymbol{\mu}_k$, and covariance matrices $\boldsymbol{\Sigma}_k$.
- **Hyperparameters**: `n_components` ($K$), `covariance_type` ('full', 'tied', 'diag', 'spherical').

---

## 7. Comparative Algorithm Selection Matrix

| Algorithm | Scalability to Large $N$ | High Dimension $D$ | Non-Linear Data | Outlier Sensitivity | Primary Industry Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Excellent ($O(N)$) | Excellent with $L_1$ | Poor | Medium | Ultra-low latency click-through / fraud scoring |
| **Decision Trees** | High | Medium | High | Robust | Explainable clinical or financial rules |
| **Random Forest** | High | High | High | Robust | General-purpose tabular baseline |
| **XGBoost / LightGBM** | Excellent | High | Very High | Robust | Kaggle-winning & production tabular models |
| **SVM (RBF)** | Poor ($O(N^2 \text{ to } N^3)$) | High | High | High | Small-sample bioinformatics, text classification |
| **$k$-NN** | Poor ($O(N \cdot D)$ inference) | Poor | High | High | Recommendation baselines, vector similarity |
| **DBSCAN** | Medium | Poor | High | Identifies them | Spatial clustering, network anomaly detection |
| **PCA** | High | Excellent | Fails non-linear | High | Preprocessing, feature compression, visualization |

➡️ **Next Phase**: [05_ml_evaluation.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/05_ml_evaluation.md)
