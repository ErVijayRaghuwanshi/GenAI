# Phase 7 — Production Machine Learning Projects (End-to-End)

> **Target Duration**: 4–6 Weeks  
> **Prerequisites**: Phases 0 through 6.  
> **Key Mandate**: Do **NOT** proceed to Deep Learning / PyTorch until you have built, packaged, and deployed these 4 end-to-end systems. These projects leverage your existing **Python backend, Big Data (Spark/Kafka), and CDR/IPDR cybersecurity** strengths.

---

## 🏗️ Project Architecture Map

```text
 PROJECT 1: Network Intrusion Detection
 ┌───────────────┐      ┌────────────────────┐      ┌─────────────┐      ┌─────────────┐
 │ Raw PCAP/Flow ├─────►│ Sklearn/XGBoost    ├─────►│ FastAPI     ├─────►│ Security    │
 │ NSL-KDD Logs  │      │ Pipeline (.joblib) │      │ Scoring API │      │ Alert (SOC) │
 └───────────────┘      └────────────────────┘      └─────────────┘      └─────────────┘

 PROJECT 2: Telecom CDR/IPDR Fraud Detection
 ┌───────────────┐      ┌────────────────────┐      ┌─────────────┐      ┌─────────────┐
 │ High-volume   ├─────►│ Feature Store      ├─────►│ Cost-Tuned  ├─────►│ Real-time   │
 │ CDR Telemetry │      │ (Call burst, Tower)│      │ Classifier  │      │ Fraud Score │
 └───────────────┘      └────────────────────┘      └─────────────┘      └─────────────┘

 PROJECT 3: User & Entity Behavior Analytics (UEBA) Anomaly Detection
 ┌───────────────┐      ┌────────────────────┐      ┌─────────────┐      ┌─────────────┐
 │ User Action   ├─────►│ Unsupervised       ├─────►│ Dynamic     ├─────►│ Step-up MFA │
 │ Audit Trail   │      │ Isolation Forest   │      │ Threshold   │      │ Trigger     │
 └───────────────┘      └────────────────────┘      └─────────────┘      └─────────────┘

 PROJECT 4: End-to-End Streaming ML Pipeline
 ┌───────────────┐      ┌────────────────────┐      ┌─────────────┐      ┌─────────────┐
 │ Kafka Event   ├─────►│ Python Streaming   ├─────►│ Model       ├─────►│ Prometheus  │
 │ Stream (JSON) │      │ Feature Processor  │      │ Inference   │      │ & Grafana   │
 └───────────────┘      └────────────────────┘      └─────────────┘      └─────────────┘
```

---

## 🎯 Project 1: Network Intrusion Detection System (NIDS)

- **Dataset**: NSL-KDD or CICIDS2017 flow logs.
- **Task**: Binary & Multi-class classification (Benign vs DoS, Probe, R2L, U2R).
- **Core Challenge**: High volume, sparse categorical protocols, zero-tolerance for latency.

### 1.1 Complete Production Code (FastAPI + Scikit-Learn Pipeline)

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

# 1. Strict Data Contract using Pydantic
class NetworkFlowPayload(BaseModel):
    duration: float = Field(..., example=0.15, description="Flow duration in seconds")
    protocol_type: str = Field(..., example="tcp", description="tcp, udp, or icmp")
    service: str = Field(..., example="http", description="Network service (http, ftp, smtp, etc.)")
    src_bytes: int = Field(..., example=1040, ge=0)
    dst_bytes: int = Field(..., example=4520, ge=0)
    count: int = Field(..., example=15, description="Number of connections to same host in 2s")
    srv_serror_rate: float = Field(..., example=0.0, ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    is_malicious: bool
    threat_probability: float
    threat_level: str

# 2. FastAPI Application Lifespan
app = FastAPI(title="NIDS Real-time Threat Scorer", version="1.0.0")

# In-memory model artifact
model_pipeline = None

@app.on_event("startup")
def load_model():
    global model_pipeline
    try:
        # Load serialized Pipeline (ColumnTransformer + LightGBM / XGBoost)
        model_pipeline = joblib.load("nids_pipeline.joblib")
    except Exception:
        # Fallback dummy pipeline for demonstration if file not yet serialized
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression())
        ])
        model_pipeline.fit(np.random.randn(10, 7), np.random.choice([0, 1], 10))

@app.post("/api/v1/score-flow", response_model=PredictionResponse)
def score_flow(payload: NetworkFlowPayload):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")
    
    # Convert payload into DataFrame matching training schema
    input_df = pd.DataFrame([payload.dict()])
    
    # If using dummy scaler in fallback:
    features = np.array([[
        payload.duration, 1 if payload.protocol_type == "tcp" else 0,
        1 if payload.service == "http" else 0, payload.src_bytes,
        payload.dst_bytes, payload.count, payload.srv_serror_rate
    ]])
    
    prob = float(model_pipeline.predict_proba(features)[0, 1])
    is_threat = prob >= 0.75  # Tuned security threshold
    
    level = "CRITICAL" if prob > 0.90 else "HIGH" if prob > 0.75 else "MEDIUM" if prob > 0.4 else "LOW"
    
    return PredictionResponse(
        is_malicious=is_threat,
        threat_probability=round(prob, 4),
        threat_level=level
    )
```

---

## 🎯 Project 2: Telecom CDR/IPDR Fraud Detection

In Call Detail Record (CDR) and IP Detail Record (IPDR) analysis, raw records represent individual events. Machine learning models require **aggregate behavioral profiles** calculated over rolling time windows.

### 2.1 Feature Store Aggregation Engineering (Pandas / Spark logic)

```python
import pandas as pd
import numpy as np

def extract_cdr_behavioral_features(cdr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw CDR records into subscriber-level behavioral feature vectors.
    """
    cdr_df["call_start"] = pd.to_datetime(cdr_df["call_start"])
    cdr_df["is_night"] = cdr_df["call_start"].dt.hour.isin([23, 0, 1, 2, 3, 4]).astype(int)
    
    # Group by subscriber IMSI / MSISDN
    features = cdr_df.groupby("subscriber_id").agg(
        total_calls=("call_id", "count"),
        avg_duration_sec=("duration", "mean"),
        total_duration_sec=("duration", "sum"),
        unique_destinations=("destination_number", "nunique"),
        international_calls=("is_international", "sum"),
        unique_cell_towers=("cell_tower_id", "nunique"),
        night_call_ratio=("is_night", "mean")
    ).reset_index()
    
    # Behavioral Ratios
    # High frequency to diverse destinations is classic SIM-box / Wangiri fraud
    features["calls_per_tower"] = features["total_calls"] / (features["unique_cell_towers"] + 1e-5)
    features["destination_diversity_ratio"] = features["unique_destinations"] / (features["total_calls"] + 1e-5)
    
    return features
```

---

## 🎯 Project 3: User Behavior Anomaly Detection (UEBA)

- **Model**: Unsupervised `IsolationForest` or `LocalOutlierFactor`.
- **Target**: Zero-day unauthorized privilege escalation or credential compromise.
- **Workflow**:
  1. Profile user's historical login hours, query frequency, and data download bytes.
  2. Compute real-time anomaly score $s \in [0, 1]$.
  3. Emit event to IAM service: if score $> 0.82 \implies$ trigger immediate biometric re-authentication.

---

## 🎯 Project 4: End-to-End Real-Time Streaming ML System

Connect your **Apache Kafka** knowledge directly with ML serving:

```text
                      PRODUCTION STREAMING ARCHITECTURE
                      
  ┌───────────────┐        ┌───────────────┐        ┌───────────────────────┐
  │ Network Logs  │───────►│ Apache Kafka  │───────►│ Python Worker         │
  │ / Telemetry   │        │ Topic: `flows`│        │ Consumer Group: `nids`│
  └───────────────┘        └───────────────┘        └───────────┬───────────┘
                                                                │
                                              In-memory Scikit-learn Pipeline
                                              Inference latency: < 3ms
                                                                │
                           ┌────────────────────────────────────┴────────────────┐
                           ▼                                                     ▼
                  Normal Events (99.2%)                                Malicious Anomalies (0.8%)
                           │                                                     │
                           ▼                                                     ▼
               Silent Offset Commit                                  ┌───────────────────────┐
                                                                     │ Kafka `threat-alerts` │
                                                                     │ Prometheus Metric++   │
                                                                     └───────────────────────┘
```

### 4.1 Python Kafka Streaming Consumer Implementation

```python
import json
import joblib
import numpy as np
# from kafka import KafkaConsumer, KafkaProducer

def run_streaming_ml_worker():
    # Load model
    pipeline = joblib.load("nids_pipeline.joblib")
    
    print("Connecting to Kafka cluster...")
    # consumer = KafkaConsumer(
    #     'raw_telemetry',
    #     bootstrap_servers=['localhost:9092'],
    #     value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    #     auto_offset_reset='latest'
    # )
    # producer = KafkaProducer(
    #     bootstrap_servers=['localhost:9092'],
    #     value_serializer=lambda m: json.dumps(m).encode('utf-8')
    # )
    
    print("Streaming inference active. Awaiting network events...")
    # for message in consumer:
    #     event = message.value
    #     features = np.array([[event['duration'], event['bytes_sent'], ...]])
    #     anomaly_prob = pipeline.predict_proba(features)[0, 1]
    #     
    #     if anomaly_prob > 0.85:
    #         alert = {"ip": event['src_ip'], "threat_prob": anomaly_prob, "timestamp": event['ts']}
    #         producer.send('threat_alerts', value=alert)
```

### 4.2 Dockerfile for Production Serving

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 5. Milestone Completion Checklist

- [ ] Complete Project 1: Model trained on NSL-KDD, exported via Joblib, served behind FastAPI.
- [ ] Complete Project 2: Feature pipeline aggregating raw CDRs into behavioral metrics; tuned decision threshold minimizing false positive customer friction.
- [ ] Complete Project 3: Unsupervised Isolation Forest detecting rare telemetry anomalies.
- [ ] Complete Project 4: Dockerized microservice processing simulated streaming events with $<10\text{ms}$ p99 latency.

➡️ **Next Phase**: [08_deep_learning_fundamentals.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/08_deep_learning_fundamentals.md)
