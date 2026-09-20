# Phase 8 — Deep Learning Fundamentals (From Perceptrons to Backprop)

> **Target Duration**: 3–4 Weeks  
> **Prerequisites**: Phase 1 (Multivariable Calculus & Linear Algebra) and Phase 2 (ML Fundamentals).  
> **Key Goal**: Understand why neural networks work, how gradients flow backward through computational graphs, the physics of activation and loss surfaces, and implement a full multilayer neural network from scratch in pure NumPy.

---

## 1. Anatomy of a Neuron & The Non-Linearity Requirement

An artificial neuron computes an affine linear transformation followed by an element-wise non-linear activation:
$$z = \sum_{j=1}^D w_j x_j + b = \mathbf{w}^T \mathbf{x} + b$$
$$a = \sigma(z)$$

```text
  x_1 ──(w_1)──┐
  x_2 ──(w_2)──┼──► [ Sum: z = w^T x + b ] ──► [ Activation: a = σ(z) ] ──► Output a
  x_D ──(w_D)──┘
        Bias b ─┘
```

### Why Non-Linear Activation Functions Are Mandatory

Suppose we build a 3-layer network without non-linear activations:
$$\hat{\mathbf{y}} = \mathbf{W}_3 (\mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3$$
Expanding this algebraically:
$$\hat{\mathbf{y}} = (\mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1) \mathbf{x} + (\mathbf{W}_3 \mathbf{W}_2 \mathbf{b}_1 + \mathbf{W}_3 \mathbf{b}_2 + \mathbf{b}_3) = \mathbf{W}_{\text{eff}} \mathbf{x} + \mathbf{b}_{\text{eff}}$$
**A 1,000-layer linear neural network collapses into a single linear regression model.** Non-linearities are what give neural networks the ability to act as **Universal Function Approximators**.

---

## 2. Activation Function Zoo

```text
       SIGMOID                  TANH                    RELU                   GELU
       σ(z) = 1/(1+e^-z)        tanh(z)                 max(0, z)              x · Φ(x)
          ▲                        ▲                       ▲                      ▲
        1 ┼───-.._               1 ┼───-.._              4 ┼     /              4 ┼     /
          │       `\               │       `\              │    /                 │    /
      0.5 ┼   *    │             0 ┼───────*──────       0 ┼───*───────         0 ┼──~*───────
          │ _..-───┘            -1 ┼ _..-───┘              │                      │
        0 ┴────────►             0 ┴────────►            0 ┴────────►           0 ┴────────►
       -4    0    4             -4    0    4            -4    0    4             -4    0    4
```

| Activation | Formula | Gradient $\sigma'(z)$ | Properties & Traps |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | Output in $(0, 1)$. Saturated gradients for $\|z\| > 4$ cause **vanishing gradients**. Not zero-centered. |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | Output in $(-1, 1)$. Zero-centered, but still suffers from vanishing gradients at extremes. |
| **ReLU** | $\max(0, z)$ | $1 \text{ if } z > 0 \text{ else } 0$ | Fast computation, constant gradient for $z > 0$. Suffers from **"Dying ReLU"** if neurons get stuck with $z < 0$. |
| **Leaky ReLU** | $\max(\alpha z, z)$ | $1 \text{ if } z > 0 \text{ else } \alpha$ | Small slope $\alpha \approx 0.01$ ensures gradients flow even when $z < 0$. |
| **GELU** | $z \cdot \Phi(z)$ | Smooth non-monotonic | Smooth approximation to ReLU; default in modern **Transformers (BERT, GPT-4, Llama)**. |
| **Softmax** | $\frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}$ | $\text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$ | Converts logits into multi-class probability distribution summing to $1.0$. |

---

## 3. Loss Functions: Maximum Likelihood Estimation

1. **Mean Squared Error (MSE)** (Regression):
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{2N} \sum_{i=1}^N \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|^2$$
2. **Binary Cross-Entropy (BCE)** (Binary Classification):
   $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
3. **Categorical Cross-Entropy (CCE)** (Multi-Class Classification):
   $$\mathcal{L}_{\text{CCE}} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i, c} \log(\hat{y}_{i, c})$$
   - *Numerical Stability Tip*: In code, compute cross-entropy directly from raw unnormalized logits using the **Log-Sum-Exp trick** to prevent float underflow/overflow.

---

## 4. Backpropagation & The Computational Graph

Backpropagation is simply the recursive application of the **Multivariable Chain Rule** computed in reverse topological order.

```text
FORWARD PASS (Compute Activations):
  x ──► [ Linear: z = W x + b ] ──► [ Activation: a = ReLU(z) ] ──► [ Loss: L(a, y) ]

BACKWARD PASS (Propagate Gradients):
  dL/dx ◄── [ dL/dz · W^T ] ◄── [ dL/da · ReLU'(z) ] ◄── [ dL/da ] ◄── Initial dL/dL = 1
```

For a linear layer $\mathbf{z} = \mathbf{X} \mathbf{W} + \mathbf{b}$ with incoming upstream gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{z}}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^T \left( \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \right), \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{\text{rows}} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \right), \quad \frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \right) \mathbf{W}^T$$

---

## 5. Optimizers: SGD to AdamW

```text
SGD                    SGD + Momentum              RMSprop                     Adam / AdamW
w -= lr * grad         v = beta*v + grad           s = beta*s + (1-beta)*grad^2 Combines Momentum
                       w -= lr * v                 w -= lr * grad / sqrt(s+eps) and RMSprop
(Oscillates wildly)    (Builds velocity in         (Scales individual coords    (Industry standard
                        consistent direction)       inversely by gradient size)  for DL & LLMs)
```

### AdamW: Why Decoupled Weight Decay Matters

In standard Adam, adding $L_2$ regularization to the loss results in the weight penalty being scaled inversely by $\sqrt{s_t + \epsilon}$, causing weights with large historical gradients to be regularized *less* than weights with small gradients.  
**AdamW** decouples weight decay directly into the parameter update step:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \lambda \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon} \hat{\mathbf{v}}_t$$

---

## 6. From-Scratch Neural Network Implementation (Pure NumPy)

A complete 2-layer Neural Network solving the non-linear XOR/Circle problem:

```python
import numpy as np

class ScratchMLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 0.1):
        self.lr = lr
        # He initialization for ReLU
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        # Xavier initialization for Sigmoid output
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_grad(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._sigmoid(self.z2)
        return self.a2

    def backward(self, y):
        N = y.shape[0]
        # Loss derivative w.r.t z2 (for Sigmoid + BCE: dz2 = a2 - y)
        dz2 = (self.a2 - y) / N
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Backprop into hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_grad(self.z1)
        dW1 = self.X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient Descent Updates
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

# Verify on non-linear XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = ScratchMLP(input_dim=2, hidden_dim=8, output_dim=1, lr=0.5)
for epoch in range(1000):
    preds = mlp.forward(X)
    loss = -np.mean(y * np.log(preds + 1e-8) + (1 - y) * np.log(1 - preds + 1e-8))
    mlp.backward(y)
    if epoch % 200 == 0:
        print(f"Epoch {epoch:04d} | BCE Loss: {loss:.4f}")

print("\nFinal XOR Predictions:")
print(np.round(mlp.forward(X), 3))
```

---

## 7. Deep Learning Diagnostics

1. **Vanishing Gradient**: Gradients shrink exponentially toward 0 in early layers $\implies$ Use ReLU/GELU, skip connections (ResNet), and LayerNorm.
2. **Exploding Gradient**: Gradients grow exponentially $\to \infty$ or NaN $\implies$ Use **Gradient Clipping** (`torch.nn.utils.clip_grad_norm_`).
3. **Weight Initialization**: Never initialize weights to all zeros! All neurons will compute identical features and identical gradients, breaking symmetry. Use He (Kaiming) or Xavier (Glorot) initialization.

➡️ **Next Phase**: [09_pytorch.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/09_pytorch.md)
