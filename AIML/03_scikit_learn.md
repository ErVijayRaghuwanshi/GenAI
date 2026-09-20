# Phase 3 — scikit-learn Mastery & Production Pipelines

> **Target Duration**: 6–8 Weeks  
> **Prerequisites**: Phase 0 (Python/Pandas), Phase 1 (Math), and Phase 2 (ML Fundamentals).  
> **Key Goal**: Master scikit-learn's API design, data leakage prevention, cross-validation strategies, and build production-grade `ColumnTransformer` + `Pipeline` architectures that can be serialized directly into backend microservices.

---

## 1. Scikit-learn Design Architecture

Scikit-learn adheres to three core object-oriented interfaces:

```text
1. ESTIMATOR    .fit(X, y)            Learns parameters from data (e.g., mean/std in scaler, weights in model).
2. TRANSFORMER  .transform(X)         Applies learned parameters to transform data (e.g., standardizes features).
3. PREDICTOR    .predict(X)           Generates predictions on new data.
                .predict_proba(X)     Outputs class probability distributions.
```

---

## 2. Dataset Splitting & Preventing Data Leakage

### 2.1 The Silent Killer: Data Leakage

**Data Leakage** occurs when information from outside the training dataset (such as the validation set or future timestamps) contaminates model training, producing artificially inflated validation scores that collapse in production.

```text
CRITICAL BUG (Data Leakage):
  Raw Data ──► [ StandardScaler().fit_transform(X) ] ──► [ train_test_split ]
               ^ Scaler learned mean/std of test set!

CORRECT PRODUCTION WORKFLOW:
  Raw Data ──► [ train_test_split ]
                     ├── X_train ──► [ StandardScaler().fit(X_train) ] ──► X_train_scaled
                     └── X_test  ──► [ StandardScaler().transform(X_test) ] ──► X_test_scaled
```

### 2.2 Cross-Validation Strategies

| Splitter | When to Use | Danger if Used Incorrectly |
| :--- | :--- | :--- |
| `KFold` | Balanced, independent and identically distributed (i.i.d.) tabular data. | In severe class imbalance (e.g., 99.5% benign, 0.5% fraud), some folds may contain zero fraud samples. |
| `StratifiedKFold` | Classification with imbalanced classes (Fraud, Intrusion detection). | Ensures each fold preserves the exact class ratio of the population. |
| `TimeSeriesSplit` | Time-series, telemetry, CDR logs, server metrics. | Standard shuffle leaks future events into past training data (lookahead bias). TimeSeriesSplit uses expanding rolling windows. |
| `GroupKFold` | Data grouped by entities (e.g., multiple sessions per `user_id` or `device_id`). | Standard split splits a user's logs across train and test, causing identity memorization rather than learning general patterns. |

#### Time-Series Cross-Validation Diagram

```text
Fold 1: [ Train: Month 1 ] ──► [ Test: Month 2 ]
Fold 2: [ Train: Month 1-2 ] ──► [ Test: Month 3 ]
Fold 3: [ Train: Month 1-3 ] ──► [ Test: Month 4 ]
(Never train on future data to predict the past!)
```

---

## 3. Feature Transformation Toolkit

| Transformer | Formula / Mechanism | Best Suited For |
| :--- | :--- | :--- |
| `StandardScaler` | $z = \frac{x - \mu}{\sigma}$ | Features with Gaussian distribution; sensitive to extreme outliers. |
| `MinMaxScaler` | $z = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$ | Bounded features (e.g., image pixels, probabilities, fixed ranges). |
| `RobustScaler` | $z = \frac{x - Q_2}{Q_3 - Q_1}$ (Median & IQR) | Data with extreme outliers (e.g., network packet spikes, transaction values). |
| `SimpleImputer` | Replaces `NaN` with mean, median, mode, or constant. | Handling missing data without dropping rows. Always compute on train fold! |
| `OneHotEncoder` | Creates binary dummy indicators. | Low-cardinality nominal categories (e.g., Protocol: TCP, UDP, ICMP). Use `handle_unknown='ignore'`. |
| `TargetEncoder` | Replaces category with target expected value. | High-cardinality categories (e.g., ZIP codes, Cell Tower IDs). Built-in smoothing avoids leakage. |

---

## 4. Building Production Pipelines with ColumnTransformer

In real backend systems, incoming payloads contain mixed types: continuous numbers, noisy outliers, missing values, and categorical strings. The production approach is a unified, serializable **Pipeline**.

### Complete End-to-End Pipeline Implementation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1. Create realistic Telecom CDR / Network dataset
np.random.seed(42)
N = 5000
df = pd.DataFrame({
    "call_duration": np.random.exponential(scale=15.0, size=N),
    "bytes_transferred": np.random.exponential(scale=5000.0, size=N),
    "failed_attempts": np.random.poisson(lam=0.5, size=N),
    "device_brand": np.random.choice(["Apple", "Samsung", "Xiaomi", "Other", None], size=N),
    "connection_type": np.random.choice(["5G", "4G", "3G", "WiFi"], size=N),
    "is_fraud": np.random.choice([0, 1], size=N, p=[0.96, 0.04])
})

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Define Feature Subsets
robust_numeric_features = ["bytes_transferred", "call_duration"] # Outlier-heavy
standard_numeric_features = ["failed_attempts"]
categorical_features = ["device_brand", "connection_type"]

# 3. Build Sub-Pipelines
robust_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

standard_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 4. Assemble Full ColumnTransformer Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num_robust", robust_pipeline, robust_numeric_features),
        ("num_standard", standard_pipeline, standard_numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)

# 5. Connect Preprocessor directly to Estimator in an End-to-End Pipeline
full_model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])

# 6. Evaluate with Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(full_model_pipeline, X_train, y_train, cv=cv, scoring="f1")
print(f"Stratified 5-Fold F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 7. Fit on entire training set & evaluate on held-out test set
full_model_pipeline.fit(X_train, y_train)
y_pred = full_model_pipeline.predict(X_test)
print("\nFinal Test Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# 8. Export Model Artifact for FastAPI Production Serving
joblib.dump(full_model_pipeline, "telecom_fraud_pipeline.joblib")
print("Saved pipeline artifact to telecom_fraud_pipeline.joblib")
```

---

## 5. Hyperparameter Tuning with Pipeline Integration

Never tune hyperparameters outside a pipeline; doing so leaks validation fold information into feature selection or scaling.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
    "classifier__penalty": ["l2"],
    "classifier__solver": ["lbfgs"]
}

grid_search = GridSearchCV(
    estimator=full_model_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
print("Best Parameters:", grid_search.best_params_)
```

---

## 6. Common Pitfalls & Interview Questions

1. **Why is `fit_transform()` only called on `X_train` and never on `X_test`?**
   - *Answer*: `fit()` learns statistics (mean, variance, category dictionaries) from the data. Calling `fit()` or `fit_transform()` on test data corrupts the evaluation by letting the model peek into test distributions, violating the fundamental assumption of generalization.
2. **What does `handle_unknown="ignore"` in `OneHotEncoder` do in production?**
   - *Answer*: If a new category appears during live inference that was never seen in training (e.g., a new device brand), the encoder outputs an all-zero vector for that feature instead of throwing a 500 runtime exception.
3. **When should you prefer `RobustScaler` over `StandardScaler`?**
   - *Answer*: When the feature contains large or frequent outliers (e.g., DDoS traffic spikes or massive financial transactions). `StandardScaler` squares errors in variance calculation, heavily distorting the mean and scale when outliers exist.

➡️ **Next Phase**: [04_ml_algorithms.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/04_ml_algorithms.md)
