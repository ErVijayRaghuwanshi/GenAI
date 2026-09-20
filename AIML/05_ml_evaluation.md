# Phase 5 — Machine Learning Evaluation & Production Metrics

> **Target Duration**: 3–4 Weeks  
> **Prerequisites**: Phase 2 (Fundamentals) and Phase 4 (ML Algorithms).  
> **Key Goal**: Master the mathematical definitions, trade-offs, and failure points of evaluation metrics. Learn how to optimize decision thresholds under extreme class imbalance (cybersecurity intrusions, telecom fraud, and anomaly detection) where accuracy is useless.

---

## 1. Regression Metrics: Magnitude, Penalty & Scale

| Metric | Mathematical Formula | Sensitivity to Outliers | Production Use Case |
| :--- | :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | $\frac{1}{N}\sum_{i=1}^N \|y_i - \hat{y}_i\|$ | Low (Linear penalty) | Business-facing error; easy to explain to stakeholders. |
| **MSE** (Mean Squared Error) | $\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$ | Very High (Quadratic penalty) | Differentiable loss for gradient-based training. |
| **RMSE** (Root Mean Squared Error) | $\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$ | High | Penalizes large errors heavily; same units as target $y$. |
| **$R^2$** (Coefficient of Determination) | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | High | Fraction of variance explained by model ($1.0 = \text{perfect}$, $0.0 = \text{baseline mean}$, $<0 = \text{worse than mean}$). |
| **Adjusted $R^2$** | $1 - \left[ \frac{(1 - R^2)(N - 1)}{N - p - 1} \right]$ | High | Penalizes addition of non-informative features $p$. |
| **MAPE** (Mean Absolute % Error) | $\frac{100\%}{N}\sum \left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | Extreme (Fails when $y_i \to 0$) | Percentage error for non-zero forecasting (e.g., cloud cost). |

---

## 2. Classification Metrics & The Confusion Matrix

In production systems (e.g., cyber threat detection, CDR fraud), positive samples ($y=1$) are rare ($<1\%$).

```text
                           PREDICTED CLASS
                      Positive (1)        Negative (0)
                   ┌───────────────────┬───────────────────┐
     Positive (1)  │ True Positive(TP) │False Negative(FN) │ ◄── SENSITIVITY / RECALL
ACTUAL             │ Attack detected   │ Attack MISSED!    │     TP / (TP + FN)
CLASS              ├───────────────────┼───────────────────┤
     Negative (0)  │False Positive(FP) │ True Negative(TN) │ ◄── SPECIFICITY
                   │ False alarm!      │ Benign ignored    │     TN / (TN + FP)
                   └───────────────────┴───────────────────┘
                             ▲
                         PRECISION
                      TP / (TP + FP)
```

### 2.1 The Accuracy Paradox

Suppose you have 1,000,000 network requests, of which 1,000 are DDoS attacks ($0.1\%$).
A dummy model that always predicts `Benign` achieves:
$$\text{Accuracy} = \frac{999,000}{1,000,000} = 99.90\%$$
Yet it detected **0 attacks** ($\text{Recall} = 0\%$). In imbalanced domains, **never report raw accuracy**.

### 2.2 Precision vs. Recall Trade-off

- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
  - *"Out of all requests flagged as malicious, how many were actually malicious?"*
  - **High Precision Priority**: Spam filtering, automated account blocking (false alarms anger real users).
- **Recall (Sensitivity)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
  - *"Out of all actual malicious attacks that occurred, how many did we catch?"*
  - **High Recall Priority**: Intrusion detection, telecom fraud, cancer screening (missing an attack is catastrophic).
- **$F_1$-Score**: Harmonic mean:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **$F_\beta$-Score**: Weighted harmonic mean:
  $$F_\beta = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$
  - $\beta = 2$: Weighs **recall** twice as heavily as precision ($F_2$ score for cyber/fraud).
  - $\beta = 0.5$: Weighs **precision** twice as heavily as recall.

---

## 3. ROC-AUC vs. PR-AUC

### 3.1 ROC-AUC (Receiver Operating Characteristic)

- Plots **True Positive Rate (TPR / Recall)** vs. **False Positive Rate (FPR)** across all possible probability thresholds $\tau \in [0, 1]$:
  $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
- **AUC (Area Under Curve)**: Probability that the model ranks a randomly chosen positive sample higher than a randomly chosen negative sample.
  - $1.0$ = Perfect ranking
  - $0.5$ = Random guessing
- **Weakness with Heavy Imbalance**: Because $\text{TN}$ is massive, $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$ remains microscopic even when $\text{FP}$ numbers in the thousands. ROC-AUC gives an overly optimistic picture!

### 3.2 PR-AUC (Precision-Recall Curve / Average Precision)

- Plots **Precision** vs. **Recall** across thresholds.
- **Rule of Thumb**: For severe class imbalance ($<5\%$ positive rate), **PR-AUC is the industry standard** metric because it completely ignores $\text{TN}$ and focuses strictly on true positives and false alarms.

---

## 4. Cost-Sensitive Threshold Optimization (Production Code)

In standard Scikit-learn, `.predict()` uses an arbitrary threshold of $\tau = 0.5$. In production, you must optimize $\tau$ based on the **Business Cost Matrix**:

$$C_{\text{total}}(\tau) = c_{\text{FP}} \cdot \text{FP}(\tau) + c_{\text{FN}} \cdot \text{FN}(\tau)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score

# Simulate model predicted probabilities and true binary labels
np.random.seed(42)
N = 100_000
fraud_rate = 0.01  # 1% true fraud rate

y_true = np.random.choice([0, 1], size=N, p=[1 - fraud_rate, fraud_rate])
# True fraudsters score higher on average
y_probs = np.where(y_true == 1, np.random.beta(5, 2, size=N), np.random.beta(1, 10, size=N))

# Financial cost parameters
COST_FP = 5.0    # $5 cost of manual security investigation for false alarm
COST_FN = 500.0  # $500 average loss of missed fraud attack

thresholds = np.linspace(0.01, 0.99, 100)
costs = []

for tau in thresholds:
    y_pred = (y_probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * COST_FP) + (fn * COST_FN)
    costs.append(total_cost)

optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
min_cost = costs[optimal_idx]
default_cost = costs[np.argmin(np.abs(thresholds - 0.5))]

print(f"ROC-AUC: {roc_auc_score(y_true, y_probs):.4f}")
print(f"PR-AUC (Average Precision): {average_precision_score(y_true, y_probs):.4f}")
print(f"Default (0.50) Threshold Cost: ${default_cost:,.2f}")
print(f"Optimal ({optimal_threshold:.2f}) Threshold Cost: ${min_cost:,.2f}")
print(f"Cost Savings: ${default_cost - min_cost:,.2f} ({(default_cost - min_cost)/default_cost*100:.1f}%)")
```

---

## 5. Interview Questions & Key Takeaways

1. **Why does AUC-ROC remain unchanged under monotonic transformations of predicted probabilities?**
   - *Answer*: Because ROC-AUC depends strictly on the **rank ordering** of predictions, not their absolute calibrated values. If you apply any strictly monotonically increasing function (e.g., $f(p) = p^3$ or $f(p) = \log(p)$), the pairwise ranks between samples do not change.
2. **When would you deliberately choose a model with lower Accuracy but higher Recall?**
   - *Answer*: In malware detection, intrusion prevention, or fraud detection, where the business cost of a False Negative (missed zero-day breach or financial theft) is orders of magnitude higher than a False Positive (a temporary MFA verification prompt).
3. **What is Log-Loss / Cross-Entropy and why is it preferred for training over Accuracy or $F_1$?**
   - *Answer*: Accuracy and $F_1$ are step-function metrics with zero gradients almost everywhere, making them unsuitable for gradient descent optimization. Cross-Entropy is continuous, smooth, convex, and directly penalizes confident incorrect predictions exponentially.

➡️ **Next Phase**: [06_unsupervised_learning.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/06_unsupervised_learning.md)
