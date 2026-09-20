# Phase 9 — PyTorch Mastery (From Tensors to Production Loops)

> **Target Duration**: 6–8 Weeks  
> **Prerequisites**: Phase 0 (Python/Vectorization), Phase 1 (Math), and Phase 8 (Deep Learning Fundamentals).  
> **Key Goal**: Master PyTorch as your primary deep learning framework. Learn tensor internals, autograd computational graphs, custom `Dataset`/`DataLoader` pipelines, and master writing a production-grade training loop **from scratch without copying from a tutorial**.

---

## 1. PyTorch Tensors & Hardware Acceleration

A PyTorch `torch.Tensor` is an n-dimensional array backed by a contiguous block of C++ memory, equipped with **Automatic Differentiation (`autograd`)** and GPU acceleration.

### 1.1 Device Management & Hardware Agnostic Code

```python
import torch

# Automatic device selection (NVIDIA CUDA -> Apple Silicon MPS -> CPU)
def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Mac Apple Silicon M1/M2/M3/M4
    return torch.device("cpu")

device = get_torch_device()
print(f"Active Accelerator Device: {device}")

# Move tensor to accelerator memory
x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
```

### 1.2 Memory Layout, Strides, and Reshaping

```python
# Create a 2D tensor
t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("Shape:", t.shape)      # torch.Size([2, 3])
print("Strides:", t.stride())  # (3, 1) -> Jump 3 elements for next row, 1 for next col

# Transposing changes strides without moving data in memory
t_transposed = t.t()
print("Is transposed contiguous?", t_transposed.is_contiguous())  # False!

# t_transposed.view(6) will crash with RuntimeError!
# Always use .contiguous() before .view(), or use .reshape() directly:
t_flat = t_transposed.contiguous().view(-1)
```

---

## 2. Autograd & Computational Graph Dynamics

PyTorch uses **Dynamic Computational Graphs** (Define-by-Run). The graph is constructed on-the-fly during the forward pass and freed immediately after `.backward()`.

```text
       x (requires_grad=True)
         │
         ▼
    [ a = x * 2 ]  ──► (GradFn: MulBackward0)
         │
         ▼
    [ y = a^2 + 3 ] ──► (GradFn: AddBackward0)
         │
         ▼
    [ Loss L ]
```

```python
import torch

# 1. Leaf node tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 2. Forward operations
y = x ** 2 + 5 * x
loss = y.sum()

# 3. Backward Pass
loss.backward()

# 4. Inspect analytic gradients: d(loss)/dx = 2x + 5
print("Analytic Gradients:", x.grad)  # [2(2)+5, 2(3)+5] -> [9.0, 11.0]

# 5. INFERENCE EFFICIENCY: Disable graph construction
with torch.inference_mode():  # Faster and less overhead than torch.no_grad()
    y_test = x ** 2
    # No computational graph is tracked here
```

---

## 3. Data Pipelines: Custom `Dataset` and `DataLoader`

Never load entire gigabyte-scale datasets directly into VRAM. Use `torch.utils.data.Dataset` and `DataLoader` for multi-threaded streaming.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NetworkLogDataset(Dataset):
    """
    Custom Dataset for high-throughput network event classification.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Store as standard float32 and int64 tensors
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# Production DataLoader Configuration
dataset = NetworkLogDataset(
    features=np.random.randn(50000, 30),
    labels=np.random.choice([0, 1], size=50000)
)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,          # Parallel worker processes for data loading
    pin_memory=True,        # Fast page-locked host memory transfer to GPU
    drop_last=False
)
```

---

## 4. Building Custom Models with `nn.Module`

```python
import torch
import torch.nn as nn

class NetworkThreatClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),     # Normalizes activations across batch
            nn.GELU(),
            nn.Dropout(p=dropout_p), # Regularization: randomly zeros 30% of neurons
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(64, num_classes) # Raw unnormalized logits (no Softmax here!)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

---

## 5. The Idiomatic Production Training Loop

A complete, clean, modular training and evaluation loop with gradient clipping, learning rate scheduling, and model checkpointing:

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, criterion, optimizer, device, clip_norm=1.0):
    model.train()  # Activates Dropout & BatchNorm training behavior
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

        # 1. Zero gradients (set_to_none=True is faster than zero_grad())
        optimizer.zero_grad(set_to_none=True)

        # 2. Forward pass
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        # 3. Backward pass
        loss.backward()

        # 4. Gradient clipping (prevents exploding gradients)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        # 5. Optimizer step
        optimizer.step()

        # Metrics tracking
        running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return running_loss / total, correct / total

@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()  # Disables Dropout & locks BatchNorm running stats
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return running_loss / total, correct / total
```

### 5.1 Training Orchestration & Checkpointing

```python
# Execution Setup
device = get_torch_device()
model = NetworkThreatClassifier(input_dim=30, num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=20)

best_val_loss = float("inf")
epochs = 5

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(model, loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Checkpointing best model weights
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_threat_model.pt")
        print("  --> Saved checkpoint: best_threat_model.pt")
```

---

## 6. Self-Check & Critical Traps

1. **Why must you call `model.train()` and `model.eval()`?**
   - *Answer*: Layers like `Dropout` and `BatchNorm1d` behave fundamentally differently during training vs. inference. `Dropout` drops connections during training, but must pass all activations during inference. `BatchNorm` updates running statistics during training, but freezes them during evaluation. Failing to switch modes causes inaccurate predictions.
2. **Why shouldn't you include `nn.Softmax()` at the end of your model when using `nn.CrossEntropyLoss()`?**
   - *Answer*: `nn.CrossEntropyLoss` in PyTorch expects raw unnormalized logits. Internally, it applies `log_softmax` followed by `nll_loss` using the numerically stable Log-Sum-Exp trick. Passing probabilities through Softmax twice will distort the gradients.
3. **What does `optimizer.zero_grad(set_to_none=True)` do?**
   - *Answer*: By default, PyTorch **accumulates** gradients (`param.grad += ...`). You must reset them to zero at every step. Setting them to `None` instead of zero frees memory allocations and provides a modest execution speedup.

➡️ **Next Phase**: [10_neural_network_architectures.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/10_neural_network_architectures.md)
