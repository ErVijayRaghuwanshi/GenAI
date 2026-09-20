# Phase 0 — Python for Machine Learning (Engineering Perspective)

> **Target Duration**: 1–2 Weeks  
> **Prerequisites**: Solid Python backend experience (OOP, functions, virtual environments).  
> **Key Goal**: Shift mindset from scalar row-by-row backend iteration to **vectorized array-oriented computation**, memory-efficient feature transforms, and production typing.

---

## 1. Vectorization vs. Python Loops: The Memory & Hardware Reality

In standard Python backend engineering, list comprehensions and iterative loops are standard practice. In Machine Learning, iterative Python loops destroy performance because of the **Global Interpreter Lock (GIL)**, dynamic type dispatch overhead, and poor CPU cache locality.

### 1.1 Memory Layout: Python Lists vs. NumPy Ndarrays

```text
Python List of Floats:
[ Ptr 0 ] ──────> [ PyObject Header (16B) | Type Ptr (8B) | Float Value (8B) ] (32 bytes per number!)
[ Ptr 1 ] ──────> [ PyObject Header (16B) | Type Ptr (8B) | Float Value (8B) ]
(Scattered across heap memory -> Cache miss on every iteration)

NumPy Contiguous Ndarray:
┌──────────┬──────────┬──────────┬──────────┐
│ float64  │ float64  │ float64  │ float64  │ (8 contiguous bytes each, aligned in L1/L2 cache)
└──────────┴──────────┴──────────┴──────────┘
(Loaded into CPU SIMD registers -> AVX2/AVX-512 executes 4 to 8 operations per clock cycle)
```

### 1.2 Performance Benchmark Demonstration

```python
import time
import numpy as np

# 10 Million elements
N = 10_000_000
python_list = list(range(N))
numpy_array = np.arange(N, dtype=np.float64)

# Native Python Loop
start = time.perf_counter()
res_list = [x * 2.5 + 1.0 for x in python_list]
py_time = time.perf_counter() - start

# Vectorized NumPy
start = time.perf_counter()
res_np = numpy_array * 2.5 + 1.0
np_time = time.perf_counter() - start

print(f"Python loop: {py_time:.4f}s")
print(f"NumPy vectorized: {np_time:.4f}s")
print(f"Speedup: {py_time / np_time:.1f}x")
# Typical output: ~40x to 80x faster
```

---

## 2. NumPy Mastery for ML Engineers

### 2.1 Broadcasting Rules

NumPy compares shapes element-wise from **trailing (right-most) dimensions** backward. Two dimensions are compatible when:
1. They are equal, OR
2. One of them is $1$.

```text
Shape A:  (3, 1, 5)
Shape B:     (4, 5)  -> Virtual alignment: (1, 4, 5)
                      Dimension 2: 5 == 5 (OK)
                      Dimension 1: 1 and 4 -> 1 stretches to 4 (OK)
                      Dimension 0: 3 and 1 -> 1 stretches to 3 (OK)
Result:   (3, 4, 5)
```

#### Code Example: Normalizing Feature Matrix (Centering & Scaling)

```python
import numpy as np

# Matrix of 100 samples and 4 features (e.g., CDR durations, packet counts)
X = np.random.randn(100, 4) * 20.0 + 50.0

# Calculate mean and standard deviation along features (axis=0)
mean = np.mean(X, axis=0)  # Shape: (4,)
std = np.std(X, axis=0)    # Shape: (4,)

# Broadcasting: (100, 4) - (4,) -> (4,) broadcasted across all 100 rows
X_standardized = (X - mean) / (std + 1e-8)

print("Standardized Mean:", np.round(X_standardized.mean(axis=0), 2))
print("Standardized Std:", np.round(X_standardized.std(axis=0), 2))
```

### 2.2 Memory Views vs. Copies

Creating unintended copies in memory can exhaust RAM when working with gigabyte-scale datasets.

```python
import numpy as np

arr = np.ones((5000, 5000), dtype=np.float64)  # ~200MB

# SLICE: Creates a VIEW (zero memory copy)
view_slice = arr[:1000, :1000]
print("Is view sharing memory?", np.shares_memory(arr, view_slice))  # True

# FANCY INDEXING / BOOLEAN MASK: Creates a COPY
copy_mask = arr[arr > 0.5]
print("Is boolean mask sharing memory?", np.shares_memory(arr, copy_mask))  # False

# Ensure C-contiguous memory layout before passing to C/Cython/C++ extensions
contiguous_arr = np.ascontiguousarray(view_slice)
```

---

## 3. Idiomatic Pandas for Production ML Wrangling

As a backend and big data engineer, avoid using `.iterrows()` or `.apply()` on rows. They are unvectorized Python loops wrapped in a DataFrame.

### 3.1 Efficient Data Wrangling Pipeline

```python
import pandas as pd
import numpy as np

# Simulated Telecom CDR / Network Log Data
data = {
    "source_ip": np.random.choice(["192.168.1.1", "10.0.0.5", "172.16.0.2"], size=100_000),
    "protocol": np.random.choice(["TCP", "UDP", "ICMP"], size=100_000),
    "bytes_sent": np.random.exponential(scale=1000, size=100_000),
    "duration_sec": np.random.uniform(0.1, 60.0, size=100_000),
    "is_malicious": np.random.choice([0, 1], size=100_000, p=[0.98, 0.02])
}
df = pd.DataFrame(data)

# 1. Optimize Memory using Categories & Downcasting
df["protocol"] = df["protocol"].astype("category")
df["is_malicious"] = df["is_malicious"].astype(np.int8)
df["bytes_sent"] = pd.to_numeric(df["bytes_sent"], downcast="float")

# 2. Vectorized Feature Engineering
df["byte_rate"] = df["bytes_sent"] / df["duration_sec"]

# 3. Fast Groupby Aggregation without Row Loops
ip_summary = df.groupby("source_ip", observed=True).agg(
    total_bytes=("bytes_sent", "sum"),
    avg_duration=("duration_sec", "mean"),
    attack_count=("is_malicious", "sum"),
    packet_count=("bytes_sent", "count")
).reset_index()

# 4. Filter high-risk traffic using vectorized query
flagged_traffic = df.query("byte_rate > 5000 and protocol == 'TCP'")
```

### 3.2 Parquet & Arrow: High-Speed ML I/O

Always prefer **Parquet** with Snappy or ZSTD compression over CSV.

```python
# Save DataFrame to columnar Parquet
df.to_parquet("network_logs.parquet", engine="pyarrow", compression="snappy", index=False)

# Read only the required subset of columns (Column Pruning)
features_df = pd.read_parquet(
    "network_logs.parquet",
    columns=["bytes_sent", "duration_sec", "byte_rate"]
)
```

---

## 4. Visualizing Distributions & Statistical Diagnostics

Before feeding features into a model, you must diagnose:
1. **Skewness & Kurtosis**
2. **Outliers**
3. **Multicollinearity (Feature Correlation)**

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

np.random.seed(42)
df_sample = pd.DataFrame({
    "call_duration": np.random.exponential(scale=5.0, size=1000),
    "bytes_transferred": np.random.normal(loc=500, scale=100, size=1000),
    "failed_logins": np.random.poisson(lam=1.5, size=1000)
})

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Distribution Plot (Histogram + KDE)
sns.histplot(df_sample["call_duration"], kde=True, ax=axes[0], color="royalblue")
axes[0].set_title("Right-Skewed Distribution (Log Transform Candidate)")

# 2. Boxplot (Outlier Detection)
sns.boxplot(y=df_sample["bytes_transferred"], ax=axes[1], color="lightgreen")
axes[1].set_title("Feature Spread & Outliers")

# 3. Correlation Heatmap
corr = df_sample.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2])
axes[2].set_title("Feature Multicollinearity Matrix")

plt.tight_layout()
plt.savefig("eda_diagnostic.png")
plt.close()
```

---

## 5. Python Architecture for ML: Custom Estimators

Scikit-learn follows an elegant Object-Oriented interface. Write custom, reusable transformers that conform to `BaseEstimator` and `TransformerMixin`:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class RobustLog1pTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log(1 + x) transformation to right-skewed positive features
    while handling clipping for zero or negative values.
    """
    def __init__(self, clip_min: float = 0.0):
        self.clip_min = clip_min

    def fit(self, X, y=None):
        # Stateless transformer: fit returns self
        return self

    def transform(self, X):
        X_copy = np.array(X, copy=True)
        X_clipped = np.clip(X_copy, a_min=self.clip_min, a_max=None)
        return np.log1p(X_clipped)

# Usage
transformer = RobustLog1pTransformer()
raw_features = np.array([[0.0], [10.0], [100.0], [1000.0]])
transformed = transformer.transform(raw_features)
print(transformed)
```

---

## 6. Type Annotations for High-Reliability ML Code

Use modern Python typing for tensor shapes and matrix dimensions:

```python
from typing import Annotated, Literal
import numpy as np
import numpy.typing as npt

# Define custom matrix type aliases
FloatMatrix = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]

def compute_euclidean_distance(
    matrix_a: FloatMatrix, 
    matrix_b: FloatMatrix
) -> FloatMatrix:
    """
    Computes pairwise Euclidean distances between two matrices.
    matrix_a: Shape (N, D)
    matrix_b: Shape (M, D)
    Returns: Distance matrix of shape (N, M)
    """
    assert matrix_a.ndim == 2 and matrix_b.ndim == 2
    assert matrix_a.shape[1] == matrix_b.shape[1], "Feature dimension D must match"
    
    # Vectorized formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b^T)
    a_squared = np.sum(matrix_a ** 2, axis=1, keepdims=True)  # (N, 1)
    b_squared = np.sum(matrix_b ** 2, axis=1, keepdims=True)  # (M, 1)
    dot_prod = np.dot(matrix_a, matrix_b.T)                  # (N, M)
    
    dist_sq = a_squared + b_squared.T - 2 * dot_prod
    return np.sqrt(np.maximum(dist_sq, 0.0))
```

---

## 7. Practical Exercises & Checklist

- [ ] **Exercise 1**: Implement pairwise cosine similarity between two 2D NumPy matrices using pure vectorization (no loops).
- [ ] **Exercise 2**: Load a 1GB dataset into Pandas, inspect `df.info(memory_usage='deep')`, downcast numerical types, convert string labels to categories, and reduce memory footprint by $>60\%$.
- [ ] **Exercise 3**: Write a custom Scikit-learn transformer that detects outliers using IQR (Interquartile Range) and caps them at the 1st and 99th percentiles.

➡️ **Next Phase**: [01_math_and_statistics.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/01_math_and_statistics.md)
