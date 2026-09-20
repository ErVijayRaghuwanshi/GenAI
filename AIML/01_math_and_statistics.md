# Phase 1 — Mathematics & Statistics for Machine Learning

> **Target Duration**: 3–4 Weeks  
> **Prerequisites**: Basic high school algebra and calculus.  
> **Key Goal**: Build an intuitive, geometric, and computational understanding of vectors, matrices, probability, and gradients without getting lost in abstract pure-math proofs.

---

## 1. Linear Algebra: The Language of Tensors and Features

In Machine Learning, every data point is a vector, every dataset is a matrix, and every neural network transformation is a series of matrix multiplications.

### 1.1 Vectors, Dot Products, and Projections

A vector $\mathbf{x} \in \mathbb{R}^d$ represents a sample with $d$ features.

- **Dot Product**:
  $$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^d u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta)$$
  - Geometric meaning: Measures directional alignment. If $\theta = 0^\circ$, dot product is maximized. If orthogonal ($\theta = 90^\circ$), dot product is $0$.
  - Used in: Cosine similarity, attention mechanisms ($Q K^T$), linear decision boundaries.

- **Vector Norms**:
  - **$L_1$ Norm (Manhattan)**: $\|\mathbf{x}\|_1 = \sum_{i=1}^d |x_i|$ (promotes sparsity in Lasso regression).
  - **$L_2$ Norm (Euclidean)**: $\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^d x_i^2}$ (penalizes large values in Ridge regression).
  - **$L_\infty$ Norm (Max norm)**: $\|\mathbf{x}\|_\infty = \max_i |x_i|$.

### 1.2 Matrix Operations & Neural Layer Formulation

Consider a mini-batch of $N$ input samples, each with $D_{\text{in}}$ features.  
- Input matrix $\mathbf{X} \in \mathbb{R}^{N \times D_{\text{in}}}$
- Weight matrix $\mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$
- Bias vector $\mathbf{b} \in \mathbb{R}^{D_{\text{out}}}$

The linear affine transformation is:
$$\mathbf{Z} = \mathbf{X} \mathbf{W} + \mathbf{b} \quad (\text{Shape: } N \times D_{\text{out}})$$

```text
       X (N x Din)              W (Din x Dout)             b (1 x Dout)
┌───────────────────────┐   ┌───────────────────────┐
│  x_0,0   ...  x_0,Din │   │  w_0,0   ...  w_0,Dout│
│    :            :     │ × │    :            :     │  +  [ b_0 ... b_Dout ]
│  x_N,0   ...  x_N,Din │   │  w_Din,0 ... w_Din,Dout│
└───────────────────────┘   └───────────────────────┘
```

### 1.3 Eigenvalues and Eigenvectors

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$:
$$\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$$
- $\mathbf{v}$ is an **eigenvector** (direction remains unchanged during linear transformation).
- $\lambda$ is an **eigenvalue** (scalar scale factor).
- **Core ML Application**: In **Principal Component Analysis (PCA)**, the eigenvectors of the feature covariance matrix $\mathbf{X}^T \mathbf{X}$ define the orthogonal axes of maximum variance, and eigenvalues represent the magnitude of variance explained.

---

## 2. Probability: Reasoning Under Uncertainty

### 2.1 Conditional Probability & Bayes' Theorem

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

Where:
- $P(A)$: **Prior** probability (e.g., baseline fraud rate across all telecom calls).
- $P(B \mid A)$: **Likelihood** (e.g., probability of observing 50 international calls at 3 AM given that the user is fraudulent).
- $P(B)$: **Evidence** (total probability of observing the behavior across all users).
- $P(A \mid B)$: **Posterior** probability (updated belief that the user is fraudulent given observed behavior).

#### Python Implementation: Bayesian Fraud Probability Calculator

```python
def compute_posterior_fraud(prior_fraud: float, p_anomaly_given_fraud: float, p_anomaly_given_legit: float) -> float:
    """
    Computes P(Fraud | Anomaly Detected) using Bayes' Rule.
    """
    prior_legit = 1.0 - prior_fraud
    # Total probability of anomaly (evidence)
    p_anomaly = (p_anomaly_given_fraud * prior_fraud) + (p_anomaly_given_legit * prior_legit)
    
    # Posterior
    posterior = (p_anomaly_given_fraud * prior_fraud) / p_anomaly
    return posterior

# Base fraud rate in network = 0.5% (0.005)
# Anomaly detector triggers on 95% of fraudsters
# Anomaly detector has a 2% false alarm rate on legit users
p_fraud = compute_posterior_fraud(
    prior_fraud=0.005,
    p_anomaly_given_fraud=0.95,
    p_anomaly_given_legit=0.02
)
print(f"P(Fraud | Alert Triggered) = {p_fraud * 100:.2f}%")
# Despite 95% sensitivity, the posterior is only ~19.3% due to low base rate (Base Rate Fallacy)!
```

### 2.2 Key Probability Distributions in ML

1. **Bernoulli**: Binary classification targets $y \in \{0, 1\}$. $P(y=1) = p$.
2. **Binomial**: Number of successes in $n$ independent Bernoulli trials.
3. **Gaussian (Normal)**: $\mathcal{N}(\mu, \sigma^2)$. Fundamental to error models, weight initialization, and VAE latent spaces.
   $$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$
4. **Poisson**: Count of independent events occurring in fixed time/space (e.g., requests per minute arriving at an API endpoint).

---

## 3. Statistics: Inference, Estimation & Hypothesis Testing

### 3.1 Central Limit Theorem (CLT)

Regardless of the underlying distribution of a population (uniform, exponential, bimodal), the distribution of the sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ approaches a Normal distribution as sample size $n \to \infty$.

> **Why this matters for ML**: This is why Linear Regression residuals tend to be normally distributed when noise is an aggregation of many independent micro-factors, and why standard error calculations for model evaluation hold.

### 3.2 Hypothesis Testing & $p$-values

When comparing two models (e.g., Model A vs Model B) in production A/B testing:
- **Null Hypothesis ($H_0$)**: There is no significant difference in metric (e.g., CTR or accuracy) between Model A and Model B.
- **Alternative Hypothesis ($H_1$)**: Model B is significantly better than Model A.
- **$p$-value**: The probability of observing results at least as extreme as the measured data, assuming $H_0$ is true.
  - If $p < 0.05$ (standard significance level $\alpha$), reject $H_0$.
- **Type I Error ($\alpha$)**: False Positive (concluding Model B is better when it is not).
- **Type II Error ($\beta$)**: False Negative (failing to detect that Model B is genuinely superior). Statistical power is $1 - \beta$.

### 3.3 Correlation vs. Covariance

- **Covariance**: Measures joint variability:
  $$\text{Cov}(X, Y) = \frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})$$
- **Pearson Correlation ($r$)**: Normalized covariance bounded in $[-1, 1]$:
  $$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$
  - Captures only **linear** relationships. Non-linear relationships (e.g., $y = x^2$) can have $r \approx 0$.
- **Spearman Rank Correlation**: Computes Pearson correlation on ranks, capturing non-linear monotonic relationships.

---

## 4. Calculus & Optimization: The Engine of Model Training

Every ML training process minimizes an objective (loss) function $J(\boldsymbol{\theta})$ with respect to model parameters $\boldsymbol{\theta}$.

### 4.1 Gradients & Partial Derivatives

For a multivariable function $J(w_1, w_2, \dots, w_d)$, the gradient $\nabla J$ is the vector of all first-order partial derivatives:
$$\nabla_{\mathbf{w}} J = \begin{bmatrix} \frac{\partial J}{\partial w_1} \\ \frac{\partial J}{\partial w_2} \\ \vdots \\ \frac{\partial J}{\partial w_d} \end{bmatrix}$$
- The gradient points in the direction of **steepest ascent**.
- To minimize the loss, we step in the **negative gradient** direction:
  $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla_{\mathbf{w}} J(\mathbf{w}_t)$$
  where $\alpha > 0$ is the **learning rate**.

### 4.2 The Multivariable Chain Rule

If $y = f(u)$ and $u = g(x)$, then:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

In deep neural networks, a loss $L$ depends on layer activations $\mathbf{a}_l$, which depend on linear logits $\mathbf{z}_l$, which depend on weights $\mathbf{W}_l$:
$$\frac{\partial L}{\partial \mathbf{W}_l} = \frac{\partial L}{\partial \mathbf{a}_l} \cdot \frac{\partial \mathbf{a}_l}{\partial \mathbf{z}_l} \cdot \frac{\partial \mathbf{z}_l}{\partial \mathbf{W}_l}$$
This recursive application from output back to input is **Backpropagation**.

---

## 5. From-Scratch Implementation: Vectorized Gradient Descent

Below is a complete, runnable implementation of Gradient Descent optimizing an Ordinary Least Squares (OLS) loss function in pure NumPy:

```python
import numpy as np

# Synthetic linear data: y = 3.5 * x_1 - 2.0 * x_2 + 1.2 + Gaussian Noise
np.random.seed(42)
N = 1000
X = np.random.randn(N, 2)
true_weights = np.array([3.5, -2.0])
true_bias = 1.2
y = X @ true_weights + true_bias + np.random.normal(0, 0.1, size=N)

# Add bias column (intercept term) to feature matrix X -> shape (N, 3)
X_b = np.c_[np.ones((N, 1)), X]  # Column of 1s + features

# Initialize random weights: shape (3,)
w = np.random.randn(3)
learning_rate = 0.05
n_epochs = 100

for epoch in range(n_epochs):
    # 1. Forward Pass: Predictions y_hat = X_b @ w
    y_hat = X_b @ w
    
    # 2. Compute Mean Squared Error Loss: J = (1 / (2 * N)) * sum((y_hat - y)^2)
    loss = (1 / (2 * N)) * np.sum((y_hat - y) ** 2)
    
    # 3. Compute Analytic Gradient: dJ/dw = (1 / N) * X_b^T @ (y_hat - y)
    gradients = (1 / N) * (X_b.T @ (y_hat - y))
    
    # 4. Parameter Update
    w -= learning_rate * gradients
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.5f} | Weights [bias, w1, w2]: {w.round(3)}")

print("\nFinal Learned Parameters:", w.round(3))
print("Ground Truth Parameters: ", [true_bias, true_weights[0], true_weights[1]])
```

---

## 6. Self-Assessment & Interview Questions

1. **Why does $L_1$ regularization produce sparse weights while $L_2$ produces small weights?**
   - *Answer*: $L_1$ has constant gradients $(\pm 1)$ regardless of parameter magnitude, driving coefficients directly to exact zero at the non-differentiable sharp corners of the diamond constraint contour. $L_2$ has gradients proportional to the parameter magnitude ($2w$), shrinking weights toward zero without forcing them to exactly zero.
2. **What happens if the learning rate $\alpha$ is too high vs. too low?**
   - *Answer*: Too high $\to$ loss oscillates and diverges to $\infty$ or NaN. Too low $\to$ training crawls slowly and gets trapped in flat plateaus or poor saddle points.
3. **What is the difference between covariance and correlation?**
   - *Answer*: Covariance is dependent on the scale/units of measurement (making it impossible to compare across different feature pairs). Correlation normalizes covariance by the product of individual standard deviations, yielding a scale-free metric in $[-1, 1]$.

➡️ **Next Phase**: [02_ml_fundamentals.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/02_ml_fundamentals.md)
