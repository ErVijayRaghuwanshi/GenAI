# Phase 11 — TensorFlow & Keras (Enterprise & Production Ecosystem)

> **Target Duration**: 2–3 Weeks  
> **Prerequisites**: Phase 9 (PyTorch) and Phase 10 (Architectures).  
> **Strategic Role**: PyTorch is your primary deep learning tool for model development and modern AI. TensorFlow / Keras is learned secondarily for **enterprise production compatibility**, high-throughput **TensorFlow Serving (TFS)**, and Google Cloud (Vertex AI / TPU) deployment ecosystems.

---

## 1. PyTorch vs. TensorFlow / Keras Mental Model

```text
CONCEPT                    PYTORCH IDIOM                     KERAS / TENSORFLOW IDIOM
──────────────────────────────────────────────────────────────────────────────────────────
Model Definition           class Model(nn.Module)            Functional API / keras.Sequential
Forward Step               def forward(self, x)              def call(self, x)
Data Pipeline              Dataset + DataLoader              tf.data.Dataset (Graph-optimized)
Training Execution         Explicit Python `for` loops       model.compile() + model.fit()
Production Serving         TorchScript / TensorRT / ONNX     TF Serving (C++ gRPC/REST daemon)
```

---

## 2. The Functional API: Directed Acyclic Graphs & Multi-Branch Networks

While `Sequential` is fine for simple toy models, enterprise production architectures often feature **multi-input, multi-output, or residual skip branches**. The **Functional API** is ideal for this:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_multimodal_threat_model(tabular_dim: int, text_vocab_size: int) -> keras.Model:
    """
    Builds a dual-input model processing both numerical network metrics
    and categorical/log string sequences simultaneously.
    """
    # Branch 1: Numerical telemetry features
    numeric_input = layers.Input(shape=(tabular_dim,), name="numeric_telemetry")
    x1 = layers.Dense(64, activation="relu")(numeric_input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.2)(x1)

    # Branch 2: Log sequence / text tokens
    text_input = layers.Input(shape=(50,), name="log_sequence")
    x2 = layers.Embedding(input_dim=text_vocab_size, output_dim=32)(text_input)
    x2 = layers.GlobalAveragePooling1D()(x2)
    x2 = layers.Dense(32, activation="relu")(x2)

    # Merge branches via Concatenation
    merged = layers.concatenate([x1, x2])
    
    # Shared classification head
    dense = layers.Dense(64, activation="relu")(merged)
    output = layers.Dense(1, activation="sigmoid", name="threat_score")(dense)

    # Instantiate Model with explicit inputs and outputs
    model = keras.Model(inputs=[numeric_input, text_input], outputs=output, name="multimodal_nids")
    return model

model = build_multimodal_threat_model(tabular_dim=25, text_vocab_size=1000)
model.summary()
```

---

## 3. High-Throughput I/O with `tf.data.Dataset`

`tf.data` constructs an optimized C++ input pipeline that overlaps GPU computation with CPU prefetching to prevent GPU starvation:

```python
import tensorflow as tf
import numpy as np

def create_tf_data_pipeline(features: np.ndarray, labels: np.ndarray, batch_size: int = 128):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    # Chain transformations
    dataset = (
        dataset
        .shuffle(buffer_size=10000, seed=42)
        .batch(batch_size, drop_remainder=True)
        .map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)  # Decouples producer from consumer!
    )
    return dataset
```

---

## 4. Production Callbacks: Automating Training Health

Never run model training without callbacks to guard against overfitting and capture metrics:

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import datetime

# 1. Early Stopping: Halt training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# 2. Dynamic Learning Rate Reduction on Plateau
lr_reducer = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# 3. Model Checkpoint: Persist model in native .keras format
checkpoint = ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_auc",
    mode="max",
    save_best_only=True
)

# 4. TensorBoard Logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Compile & Fit
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)
```

---

## 5. Model Export & High-Performance TensorFlow Serving (TFS)

In high-throughput enterprise backends, models are exported to `SavedModel` format and loaded into **TF Serving**, a compiled C++ server with dynamic batching.

### 5.1 Exporting SavedModel

```python
# Export model artifact for TF Serving
model.export("saved_models/threat_scorer/1")
```

### 5.2 Launching TF Serving via Docker

```bash
docker run -d -p 8501:8501 -p 8500:8500 \
  --name tf_serving_nids \
  --mount type=bind,source=$(pwd)/saved_models/threat_scorer,target=/models/threat_scorer \
  -e MODEL_NAME=threat_scorer \
  tensorflow/serving:latest
```

### 5.3 Querying the Serving Daemon from Python Backend

```python
import requests
import json

payload = {
    "signature_name": "serving_default",
    "instances": [
        {
            "numeric_telemetry": [0.5] * 25,
            "log_sequence": [12] * 50
        }
    ]
}

response = requests.post(
    "http://localhost:8501/v1/models/threat_scorer:predict",
    data=json.dumps(payload)
)
print("TF Serving Prediction:", response.json())
```

---

## 6. Key Takeaways for AI Engineers

1. **Strategic Allocation**: Spend 70% of your deep learning effort on **PyTorch** and 30% on **TensorFlow/Keras**.
2. **Keras 3 Advantage**: Modern Keras (`keras.io`) allows you to write one set of code that compiles seamlessly across **PyTorch, JAX, and TensorFlow** backends.
3. **Serving Tradeoff**: For lightweight deployments, FastAPI + ONNX Runtime or PyTorch JIT is simpler. For massive concurrent enterprise workloads with automatic tensor batching, TF Serving remains a high-performance choice.

➡️ **Next Phase**: [12_transformers_huggingface.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/12_transformers_huggingface.md)
