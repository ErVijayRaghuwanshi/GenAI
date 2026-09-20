# Phase 12 — Transformers & Hugging Face (Fine-Tuning & PEFT)

> **Target Duration**: 4–6 Weeks  
> **Prerequisites**: Phase 9 (PyTorch) and Phase 10 (Transformer Architecture).  
> **Key Goal**: Bridge deep learning foundations with modern Generative AI. Master the Hugging Face ecosystem, modern subword tokenizers, encoder vs. decoder model families, sequence classification fine-tuning, and Parameter-Efficient Fine-Tuning (**LoRA / QLoRA**).

---

## 1. The Hugging Face Library Ecosystem

```text
  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
  │   tokenizers    │      │  transformers   │      │      peft       │
  │ Fast Rust BPE & │─────►│ Model Archs &   │─────►│ LoRA, QLoRA,    │
  │ WordPiece enc   │      │ Pretrained Wts  │      │ Prefix Tuning   │
  └─────────────────┘      └─────────────────┘      └─────────────────┘
           │                        │                        │
           ▼                        ▼                        ▼
  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
  │    datasets     │      │   accelerate    │      │    evaluate     │
  │ Memory-mapped   │─────►│ Distributed &   │─────►│ ROC-AUC, F1,    │
  │ Arrow streaming │      │ Mixed Precision │      │ ROUGE, BLEU     │
  └─────────────────┘      └─────────────────┘      └─────────────────┘
```

---

## 2. Tokenization Deep Dive: From Characters to BPE

Neural networks cannot process raw strings. Tokenizers decompose text into discrete integers:

```text
Raw String:    "Unauthorized access attempt detected: port 443"
                      │
Subword Split: ['Un', 'authorized', 'access', 'attempt', 'detected', ':', 'port', '443']
                      │
Token IDs:     [ 101,  4892,        2156,    4103,     5120,       1024, 3421,  8931, 102 ]
                      │
Attention Mask:[   1,     1,           1,       1,        1,          1,    1,     1,   1 ]
```

### 2.1 Major Subword Algorithms

1. **Byte-Pair Encoding (BPE)** (GPT-2/3/4, Llama, RoBERTa): Starts with byte characters, iteratively merges the most frequent adjacent pairs.
2. **WordPiece** (BERT): Similar to BPE, but scores merges based on maximizing the likelihood of the training data.
3. **SentencePiece** (T5, Llama): Language-independent tokenizer treating whitespace as an ordinary character (`_`).

---

## 3. The Three Transformer Families

| Family | Archetype Models | Training Objective | Best Suited For |
| :--- | :--- | :--- | :--- |
| **Encoder-Only** | BERT, RoBERTa, DeBERTa | Masked Language Modeling (MLM): bidirectional context | Classification, NER, threat log classification, dense semantic embeddings. |
| **Decoder-Only** | GPT-4, Llama 3, Mistral, Gemma | Causal Language Modeling (CLM): autoregressive next-token | Open-ended text generation, code synthesis, conversational AI, reasoning agents. |
| **Encoder-Decoder** | T5, BART | Span corruption / Sequence-to-Sequence | Document summarization, translation, structured output extraction. |

---

## 4. End-to-End Fine-Tuning for Sequence Classification

Fine-tuning a pretrained transformer (e.g., `distilbert-base-uncased`) to classify security/fraud logs:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Prepare sample dataset
logs_data = {
    "text": [
        "Failed root password from 192.168.1.10 port 22 ssh2",
        "Accepted publickey for ubuntu from 10.0.0.1 port 54122",
        "Connection reset by peer during TLS handshake",
        "SQL injection attempt in query param 'id=1 OR 1=1'",
        "HTTP GET /index.html 200 OK"
    ] * 200,
    "label": [1, 0, 0, 1, 0] * 200
}
dataset = Dataset.from_dict(logs_data).train_test_split(test_size=0.2, seed=42)

# 2. Tokenizer initialization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_batch, batched=True)

# 3. Model initialization
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Custom Metrics Computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 5. Training Configuration
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# trainer.train()  # Launches fine-tuning
```

---

## 5. Parameter-Efficient Fine-Tuning: LoRA & QLoRA ⭐⭐⭐

Full fine-tuning of 7B+ parameter LLMs requires hundreds of gigabytes of VRAM to store optimizer states (Adam stores 8 bytes per parameter). **LoRA** drastically slashes resource demands.

### 5.1 LoRA (Low-Rank Adaptation) Mechanics

Freeze the pretrained weights $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$. Inject two small trainable matrices $\mathbf{A} \in \mathbb{R}^{r \times k}$ and $\mathbf{B} \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$:

$$\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} (\mathbf{B} \cdot \mathbf{A})$$

```text
               Input Vector x
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   Pretrained W_0           Matrix A (k x r)
     (FROZEN)                    ▼
         │                  Matrix B (r x d)
         │                  (TRAINABLE, r=8)
         │                       │
         ▼                       ▼
      W_0 · x     +     (α/r) · (B · A) · x
         │                       │
         └───────────┬───────────┘
                     ▼
               Output Vector h
```

- **Parameter Savings**: Fine-tunes $<1\%$ of model weights (e.g., 8 million parameters instead of 7 billion).
- **Zero Latency Penalty at Inference**: At deployment, simply fold $\Delta \mathbf{W}$ into $\mathbf{W}_0$: $\mathbf{W}_{\text{deployed}} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$.

### 5.2 QLoRA (Quantized LoRA)

Takes LoRA further by:
1. **4-bit NormalFloat (NF4)**: Quantizes the frozen weights $\mathbf{W}_0$ to 4-bits without loss of precision.
2. **Double Quantization (DQ)**: Quantizes the quantization constants to save additional memory.
3. **Paged Optimizers**: Prevents CUDA Out-of-Memory crashes during gradient checkpoints.

### 5.3 Hands-on PEFT / LoRA Code Example

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                          # Rank dimension
    lora_alpha=16,                # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"] # Target query and value projections
)

# Wrap model with LoRA adapters
peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()
# Typical output: trainable params: 147,458 || all params: 66,511,106 || trainable%: 0.22%
```

---

## 6. Interview Questions & Key Concepts

1. **Why does LoRA adapt attention projection matrices ($W_q, W_v$) rather than MLP layers?**
   - *Answer*: Empirically, adapting the query and value projections delivers the highest downstream task performance per parameter. However, adapting all linear layers (including MLP projections) is increasingly standard in modern QLoRA recipes.
2. **What is catastrophic forgetting, and how does PEFT mitigate it?**
   - *Answer*: When a neural network completely overrides its general knowledge while learning a specific new task. Because LoRA freezes the original pretrained weights $\mathbf{W}_0$ and only trains low-rank adapter matrices $\Delta \mathbf{W}$, the model preserves its broad foundational reasoning abilities.

➡️ **Next Phase**: [13_modern_ai_engineering.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/13_modern_ai_engineering.md)
