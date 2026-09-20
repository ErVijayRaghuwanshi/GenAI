# Phase 10 — Deep Learning Architectures (MLP, CNN, RNN to Transformers)

> **Target Duration**: 4–6 Weeks  
> **Prerequisites**: Phase 8 (DL Fundamentals) and Phase 9 (PyTorch).  
> **Key Goal**: Understand the architectural evolution of deep learning: from spatial filters (CNNs) and temporal recurrence (RNN/LSTM) to the modern **Transformer** foundation powering contemporary LLMs and Foundation Models.

---

## 🏛️ Neural Architecture Taxonomy

```text
 ARCHITECTURE       INPUT DATA TYPE         CORE MECHANISM                     PARALLELIZABLE?
 ──────────────────────────────────────────────────────────────────────────────────────────────
 MLP                Tabular / Vectors       Dense Matrix Multiply (W x + b)    Yes (Batch-wide)
 CNN                Spatial / Grids / 2D    Local Receptive Field (Kernel)     Yes (Spatial & Batch)
 RNN / LSTM / GRU   Temporal Sequences      Hidden Recurrent State (h_t)       NO (Sequential in T)
 TRANSFORMER ⭐     Arbitrary Sequences     Multi-Head Self-Attention (Q,K,V)  YES (Full Sequence O(T^2))
```

---

## 1. Convolutional Neural Networks (CNNs)

CNNs exploit two fundamental spatial priors in physical data (images, spectrograms, packet payloads):
1. **Local Connectivity**: Pixels close together are correlated.
2. **Translation Invariance**: An edge or pattern has the same meaning anywhere in the image.

### 1.1 Convolution Mathematics

For input feature map $\mathbf{X}$ and kernel $\mathbf{K} \in \mathbb{R}^{k_h \times k_w}$:
$$(\mathbf{X} * \mathbf{K})_{i, j} = \sum_{m} \sum_{n} \mathbf{X}_{i+m, j+n} \mathbf{K}_{m, n}$$

- **Stride**: Step size of kernel slide across the tensor.
- **Padding**: Adding zero borders to preserve spatial dimensions:
  $$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K + 2P}{S} \right\rfloor + 1$$

### 1.2 Residual Connections (ResNet): The Deep Gradient Highway

Prior to ResNet, stacking $>20$ layers degraded performance due to vanishing gradients during backpropagation.  
**He et al. (2015)** introduced the **Skip (Residual) Connection**:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{x}$$

```text
       x ───────────────┬───────────────┐ (Identity shortcut)
                        ▼               │
                  [ Conv3x3 ]           │
                        ▼               │
                  [ Conv3x3 ]           │
                        ▼               │
                [ Addition: F(x) + x ] ◄┘
                        ▼
                     [ ReLU ]
```

When taking gradients:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right)$$
The identity matrix term $\mathbf{I}$ ensures that **gradients flow backward unhindered through hundreds of layers**.

---

## 2. Recurrent Architectures (RNN, LSTM, GRU)

Recurrent models process sequential inputs $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T$ step-by-step:
$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

```text
  x_1               x_2               x_T
   │                 │                 │
   ▼                 ▼                 ▼
 [h_0] ──► [RNN] ──► [h_1] ──► [RNN] ──► [h_2] ... ──► [h_T]
```

### The LSTM Gating Mechanism

Standard RNNs fail on sequences longer than ~50 steps due to exponential gradient decay ($(\mathbf{W}_{hh})^T$).  
**LSTM (Long Short-Term Memory)** introduces an explicit linear memory highway called the **Cell State ($\mathbf{C}_t$)** controlled by three gates:

1. **Forget Gate**: Decides what to discard from previous memory:
   $$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
2. **Input Gate & Candidate State**: Decides what new information to store:
   $$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i), \quad \tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
3. **Cell State Update**: Linear combination without non-linear squash:
   $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
4. **Output Gate**: Produces the emitted hidden state:
   $$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o), \quad \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

> **Why LSTMs were replaced by Transformers**: LSTMs are strictly sequential ($O(T)$ steps where step $t$ must wait for $t-1$). They cannot saturate massive GPU parallelism during training!

---

## 3. The Transformer Architecture ⭐⭐⭐

Introduced in *"Attention Is All You Need"* (Vaswani et al., 2017), the Transformer eliminates recurrence entirely, replacing it with **Self-Attention**.

```text
                           TRANSFORMER ENCODER LAYER
                           
                            Input Tokens / Vectors
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Positional Encoding +   │
                        └─────────────┬─────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │   Residual Connection   │
                         │            │            │
                         │     ┌──────▼──────┐     │
                         │     │ Multi-Head  │     │
                         │     │  Attention  │     │
                         │     └──────┬──────┘     │
                         │            ▼            │
                         │       [ Add & Norm ] ◄──┘
                         │            │
                         │   Residual Connection   │
                         │            │            │
                         │     ┌──────▼──────┐     │
                         │     │ Feed-Forward│     │
                         │     │ (MLP Block) │     │
                         │     └──────┬──────┘     │
                         │            ▼            │
                         │       [ Add & Norm ] ◄──┘
                         └────────────┬────────────┘
                                      ▼
                          Output Encoded Vectors
```

### 3.1 Scaled Dot-Product Attention

Given an input sequence mapped into three projection spaces:
- **Queries ($\mathbf{Q}$)**: What each token is looking for.
- **Keys ($\mathbf{K}$)**: What each token contains (labels/descriptors).
- **Values ($\mathbf{V}$)**: The actual informational content to aggregate.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

- **Why scale by $\frac{1}{\sqrt{d_k}}$?** As projection dimension $d_k$ grows large, the dot products grow large in magnitude, pushing the Softmax function into regions with near-zero gradients. Scaling ensures unit variance.

---

## 4. From-Scratch PyTorch Implementation: Multi-Head Attention

Below is an idiomatic, vectorized implementation of Multi-Head Self-Attention in pure PyTorch:

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # 1. Project and reshape to (Batch, Heads, Seq_len, d_k)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product: (Batch, Heads, Seq_len, Seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Optional Causal / Padding Masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax over key sequence dimension
        attn_weights = torch.softmax(scores, dim=-1)

        # 5. Multiply by Values: (Batch, Heads, Seq_len, d_k)
        context = torch.matmul(attn_weights, V)

        # 6. Concatenate heads back to original shape: (Batch, Seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 7. Final linear projection
        return self.out_proj(context)

# Verify with simulated batch of tokens
B, T, D = 4, 16, 64  # 4 samples, 16 tokens each, 64 embedding dimensions
mha = MultiHeadAttention(d_model=D, num_heads=8)
tokens = torch.randn(B, T, D)
out = mha(tokens)
print("Output Tensor Shape:", out.shape)  # torch.Size([4, 16, 64])
```

---

## 5. Architectural Comparison Summary

| Model | Complexity per Layer | Sequential Operations | Maximum Path Length |
| :--- | :--- | :--- | :--- |
| **Recurrent (LSTM)** | $O(T \cdot d^2)$ | $O(T)$ | $O(T)$ (Gradients must step through all $T$ steps) |
| **Convolutional (CNN)**| $O(k \cdot T \cdot d^2)$ | $O(1)$ | $O(\log_k(T))$ (Requires deep dilated trees) |
| **Self-Attention** | $O(T^2 \cdot d)$ | $O(1)$ | $O(1)$ (Every token connects directly to every token) |

➡️ **Next Phase**: [11_tensorflow_keras.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/11_tensorflow_keras.md)
