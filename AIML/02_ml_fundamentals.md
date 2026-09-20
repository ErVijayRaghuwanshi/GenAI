# Phase 2 — Machine Learning Fundamentals

> **Target Duration**: 3–4 Weeks  
> **Prerequisites**: Phase 0 (Python/NumPy) and Phase 1 (Math & Statistics).  
> **Key Goal**: Master the theoretical underpinnings of ML: function approximation, empirical risk minimization, the bias-variance tradeoff, and problem formulation before diving into high-level libraries.

---

## 1. What Machine Learning Actually Is

Traditional software engineering writes explicit deterministic rules:
$$\text{Input Data} + \text{Explicit Logic/Rules} \longrightarrow \text{Output Answers}$$

In Machine Learning, we invert the paradigm to approximate unknown complex functions:
$$\text{Input Data} (\mathbf{X}) + \text{Observed Outputs} (\mathbf{y}) \longrightarrow \text{Learned Function } \hat{f}(\mathbf{x})$$

Given an unknown true data-generating distribution $P(\mathbf{x}, y)$, our goal is to select an optimal hypothesis function $\hat{f} \in \mathcal{H}$ that maps inputs to outputs such that expected loss on **unseen future data** is minimized.

---

## 2. Empirical Risk Minimization (ERM) & Generalization

We want to minimize the **True Risk** (expected loss over the true population):
$$R(f) = \mathbb{E}_{(\mathbf{x}, y) \sim P}[L(f(\mathbf{x}), y)]$$

Because the true joint distribution $P$ is unknown, we only have access to an observed training sample $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$. We therefore minimize the **Empirical Risk**:
$$R_{\text{emp}}(f) = \frac{1}{N} \sum_{i=1}^N L(f(\mathbf{x}_i), y_i)$$

```text
                           Data Space
  ┌────────────────────────────────────────────────────────┐
  │  True Data Distribution P(x, y)                        │
  │                                                        │
  │     [ Training Set D_train ]      [ Test Set D_test ]  │
  │        Minimize R_emp(f)       →     Estimate R(f)     │
  │        (Optimization)                (Generalization)  │
  └────────────────────────────────────────────────────────┘
```

### The Fundamental Generalization Gap

$$\text{Generalization Gap} = R_{\text{test}}(f) - R_{\text{train}}(f)$$

- If $R_{\text{train}}$ is low and $R_{\text{test}}$ is high $\implies$ **Overfitting** (the model memorized the training sample noise).
- If both $R_{\text{train}}$ and $R_{\text{test}}$ are high $\implies$ **Underfitting** (the model family lacks expressive capacity).

---

## 3. Parametric vs. Non-Parametric Models

| Dimension | Parametric Models | Non-Parametric Models |
| :--- | :--- | :--- |
| **Definition** | Fixed number of parameters $\boldsymbol{\theta}$, independent of dataset size $N$. | Parameter count grows dynamically with the size of dataset $N$. |
| **Examples** | Linear Regression, Logistic Regression, Neural Networks, Naive Bayes. | $k$-Nearest Neighbors ($k$-NN), Decision Trees, SVM with RBF kernel. |
| **Pros** | Fast inference, fixed memory footprint, simple to interpret. | Highly flexible, can capture arbitrary non-linear boundaries. |
| **Cons** | Strong assumptions about data distribution; prone to underfitting. | Slow inference at scale, high memory consumption, prone to overfitting. |
| **Backend / Scale Context** | Excellent for ultra-low latency real-time API scoring. | Often requires vector indexing or large model memory footprints. |

---

## 4. The Bias-Variance Tradeoff (Mathematical Decomposition)

For any regression model $\hat{f}(\mathbf{x})$ evaluated using Mean Squared Error (MSE), the expected test error at a query point $\mathbf{x}$ decomposes into three distinct components:

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \underbrace{\left(\mathbb{E}[\hat{f}(\mathbf{x})] - f(\mathbf{x})\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[\left(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})]\right)^2\right]}_{\text{Variance}} + \underbrace{\sigma_\epsilon^2}_{\text{Irreducible Error}}$$

```text
Error
 ▲
 │        Total Test Error
 │           \      /
 │            \    /        Variance (Overfitting)
 │             \  /        /
 │              \/________/
 │              /\
 │             /  \
 │            /    \
 │           /      Bias^2 (Underfitting)
 └──────────┴────────────────────────► Model Complexity
         Simple                    Complex
       (Linear)                 (Deep Trees / Large NNs)
```

1. **Bias**: Error introduced by approximating a real-world complex problem with an overly simplistic model (e.g., fitting a straight line to a quadratic curve).
2. **Variance**: Sensitivity of the model to fluctuations in the training data. A high-variance model changes dramatically if trained on a slightly different subset of data.
3. **Irreducible Error ($\sigma_\epsilon^2$)**: Inherent stochastic noise in the measurement, missing variables, or unobservable features. No model can beat this lower bound.

### Remedies Matrix

| Condition | Diagnostic Symptom | Engineering Solution |
| :--- | :--- | :--- |
| **High Bias** (Underfitting) | High training loss + High validation loss | Add polynomial/interaction features, choose more complex model (Tree Ensembles, Deep NN), reduce regularization ($\lambda$). |
| **High Variance** (Overfitting) | Low training loss + High validation loss | Collect more training data, feature selection (drop noise columns), increase regularization ($L_1/L_2$), dropout, ensemble bagging. |

---

## 5. Machine Learning Taxonomy

```text
                               MACHINE LEARNING
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
   SUPERVISED                   UNSUPERVISED                SELF-SUPERVISED
  (Labeled Data)              (Unlabeled Data)            (Pretext Task Labels)
        │                             │                             │
 ├── Regression (Continuous)   ├── Clustering (K-Means/DBSCAN)├── Masked LM (BERT)
 │    - Price estimation       ├── Dimensionality (PCA/UMAP) └── Next-Token (GPT)
 │    - Latency prediction     └── Anomaly (Isolation Forest)
 └── Classification (Discrete)
      - Cyber intrusion detection (0 or 1)
      - Telecom churn prediction (0 or 1)
      - Malicious domain categorization (Multi-class)
```

---

## 6. Problem Formulation: From Backend Request to ML Objective

Before writing a single line of training code, transform the business requirement into a formal ML objective:

### Case Study: Telecom Network Churn & Fraud

1. **Raw Business Problem**: "Stop high-value customers from abandoning our service after experiencing dropped calls."
2. **ML Formulation**:
   - **Target variable $y$**: Binary flag $y_i \in \{0, 1\}$ where $1 = \text{Subscriber cancelled within 30 days}$.
   - **Observation window**: Aggregate CDR logs from day $t-60$ to $t$.
   - **Prediction window**: Predict churn probability for day $t+1$ to $t+30$.
   - **Loss Function**: Binary Cross-Entropy (Log-Loss) with cost weighting:
     $$\mathcal{L} = - \sum_{i=1}^N \left[ c_{\text{fn}} y_i \log(\hat{p}_i) + c_{\text{fp}} (1 - y_i) \log(1 - \hat{p}_i) \right]$$
     *(Where $c_{\text{fn}}$ represents the cost of losing a customer, typically much higher than $c_{\text{fp}}$, the cost of a retention SMS).*

---

## 7. Concept Check & Key Questions

1. **Why is training error an overly optimistic estimate of test performance?**
   - *Answer*: Because the optimization algorithm directly tuned model parameters to fit the idiosyncrasies and noise of that exact sample.
2. **Can a model have zero training error and still be completely useless in production?**
   - *Answer*: Yes. A 1-Nearest Neighbor ($k=1$) classifier or a memorizing lookup table achieves 0% training error, but has high variance and fails to generalize to unseen inputs.
3. **If adding 10,000 more training samples does not improve validation accuracy, is your model suffering from high bias or high variance?**
   - *Answer*: High bias. Adding more data does not fix an underfitting model; you need to increase model capacity or engineer more informative features.

➡️ **Next Phase**: [03_scikit_learn.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/03_scikit_learn.md)
