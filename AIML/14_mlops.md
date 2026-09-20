# Phase 14 — Production MLOps (Experiment Tracking, Serving & Drift)

> **Target Duration**: 4–6 Weeks  
> **Prerequisites**: Phases 0 through 13.  
> **Key Goal**: Bridge the final gap between a model working in a Jupyter notebook and a resilient, monitored, reproducible machine learning system operating continuously in production.

---

## 🔁 The Complete MLOps Lifecycle

```text
 ┌────────────────┐      ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
 │ 1. EXPERIMENT  │─────►│ 2. EVALUATE    │─────►│ 3. MODEL       │─────►│ 4. CI/CD       │
 │ TRACKING       │      │ & VALIDATE     │      │ REGISTRY       │      │ CONTAINERIZE   │
 │ (MLflow/W&B)   │      │ (Gate Checks)  │      │ (Staging/Prod) │      │ (Docker/Tests) │
 └────────────────┘      └────────────────┘      └────────────────┘      └───────┬────────┘
                                                                                 │
 ┌────────────────┐      ┌────────────────┐      ┌────────────────┐              │
 │ 7. AUTOMATED   │◄─────│ 6. DRIFT       │◄─────│ 5. SERVING     │◄─────────────┘
 │ RETRAINING     │      │ MONITORING     │      │ & OBSERVABILITY│
 │ (Airflow/Spark)│      │ (Evidently/PSI)│      │ (FastAPI/Prom) │
 └────────────────┘      └────────────────┘      └────────────────┘
```

---

## 1. Experiment Tracking & Model Registry with MLflow

Never lose track of hyperparameter configurations, metrics, and serialized artifacts.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

# Configure tracking backend (PostgreSQL + S3/MinIO in production)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("telecom-fraud-detection")

with mlflow.start_run(run_name="rf_tuned_estimators_300"):
    # 1. Log Hyperparameters
    params = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "class_weight": "balanced"
    }
    mlflow.log_params(params)

    # 2. Train Model
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Evaluate and Log Metrics
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.70).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("val_roc_auc", auc)
    mlflow.log_metric("val_f1", f1)

    # 4. Log and Register Model Artifact
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        registered_model_name="TelecomFraudClassifier"
    )
    print(f"Logged run to MLflow with AUC: {auc:.4f}")
```

### 1.1 Managing Model Stages in the Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model version to 'Production' alias
client.set_registered_model_alias(
    name="TelecomFraudClassifier",
    alias="production",
    version=1
)

# Load production model dynamically in FastAPI backend
prod_model = mlflow.sklearn.load_model("models:/TelecomFraudClassifier@production")
```

---

## 2. Serving Patterns: Synchronous vs. Asynchronous vs. Streaming

| Pattern | Latency | Protocol | Tech Stack | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Real-time Sync** | $< 20\text{ms}$ | HTTP/REST or gRPC | FastAPI, Triton, TorchServe | Online fraud checks, user click prediction, search ranking. |
| **Streaming Async** | $< 100\text{ms}$ | Kafka / RabbitMQ | Kafka Consumer + Python Worker | Telemetry log ingestion, CDR burst processing. |
| **Offline Batch** | Minutes to Hours | Parquet / Object Store | Apache Spark, Ray, Airflow | Daily churn risk scoring, monthly billing predictions. |

---

## 3. Production Monitoring: Data Drift vs. Concept Drift

```text
  TRAINING TIME                          PRODUCTION RUNTIME
  ┌─────────────────────────┐            ┌─────────────────────────┐
  │ P_train(X)              │            │ P_prod(X)               │
  │ Features Distribution   │            │ Shifted Distribution    │
  └───────────┬─────────────┘            └───────────┬─────────────┘
              │                                      │
              ▼                                      ▼
  ┌─────────────────────────┐   DATA DRIFT!   ┌─────────────────────────┐
  │ P(Y | X)                ├────────────────►│ P(Y | X)                │
  │ Constant relationship   │  CONCEPT DRIFT! │ Relationship changed!   │
  └─────────────────────────┘                 └─────────────────────────┘
```

1. **Data Drift (Covariate Shift)**: The input feature distribution $P(X)$ changes while $P(Y|X)$ stays the same.
   - *Example*: An update to a mobile app OS increases average byte payload sizes, shifting the feature distribution.
   - *Statistical Test*: **Two-Sample Kolmogorov-Smirnov (KS) Test** (continuous) or **Population Stability Index (PSI)**.
2. **Concept Drift**: The fundamental relationship between inputs and targets $P(Y \mid X)$ changes.
   - *Example*: Fraudsters discover a new evasion technique; transactions that looked legitimate now conceal malicious intent.
3. **Performance Drift**: Degradation of ground-truth business KPIs over time.

### 3.1 Detecting Feature Drift with Kolmogorov-Smirnov (KS) Test

```python
from scipy.stats import ks_2samp
import numpy as np

def detect_numerical_drift(reference_data: np.ndarray, production_data: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Performs two-sample Kolmogorov-Smirnov test to detect data drift.
    Returns True if statistically significant drift is detected.
    """
    statistic, p_value = ks_2samp(reference_data, production_data)
    has_drifted = p_value < alpha
    print(f"KS Statistic: {statistic:.4f} | p-value: {p_value:.4e} | Drift Detected: {has_drifted}")
    return has_drifted

# Simulate reference training distribution vs drifted production data
ref_packets = np.random.normal(loc=100, scale=15, size=5000)
prod_packets = np.random.normal(loc=125, scale=20, size=5000)  # Drifted!

detect_numerical_drift(ref_packets, prod_packets)
```

---

## 4. Production CI/CD & Automated Retraining Triggers

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                    RETRAINING TRIGGERS                      │
 ├─────────────────────────────────────────────────────────────┤
 │ 1. SCHEDULE-BASED     Retrain every Sunday at 02:00 UTC     │
 │ 2. DRIFT-BASED        Retrain when PSI > 0.2 on key columns │
 │ 3. PERFORMANCE-BASED  Retrain when F1 drops below 0.85      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                       CI/CD PIPELINE                        │
 ├─────────────────────────────────────────────────────────────┤
 │ 1. Fetch fresh snapshot from Data Lake (Parquet/Spark)      │
 │ 2. Execute deterministic unit tests on preprocessing logic  │
 │ 3. Train candidate model on GPU instance                    │
 │ 4. Run Champion vs. Challenger evaluation on held-out test  │
 │ 5. IF Challenger F1 > Champion F1:                          │
 │      - Register artifact in MLflow                          │
 │      - Build new Docker image                               │
 │      - Canary deploy (10% traffic -> 100% traffic)          │
 └─────────────────────────────────────────────────────────────┘
```

---

## 5. Complete AI/ML Engineer Checklist

Congratulations! You now possess a comprehensive, production-oriented curriculum bridging:
- **NumPy & Pandas Data Vectorization**
- **Linear Algebra, Calculus & Statistics**
- **Scikit-learn Pipelines & Leakage Prevention**
- **13 Fundamental ML Algorithms (From Scratch to Production)**
- **Cost-Sensitive Metrics for Imbalanced Threat Datasets**
- **Unsupervised Anomaly Detection (Isolation Forest)**
- **Deep Learning Fundamentals & Backprop Math**
- **PyTorch Idiomatic Loops & CUDA/MPS Acceleration**
- **CNNs, LSTMs & Transformer Self-Attention**
- **TensorFlow/Keras Functional Ecosystem**
- **Transformers, Hugging Face & PEFT (LoRA/QLoRA)**
- **Modern AI Engineering (Hybrid RAG, LangGraph Multi-Agents)**
- **Production MLOps (MLflow, Docker, Drift Detection & Retraining)**

⬅️ **Return to Master Roadmap**: [README.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/README.md)
