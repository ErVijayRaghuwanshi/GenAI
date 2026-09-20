# Phase 13 — Modern AI Engineering (RAG, Agents & Production GenAI)

> **Target Duration**: 4–6 Weeks  
> **Prerequisites**: Phases 0 through 12 + your background in **Python backend, Big Data (Spark/Kafka), and LangChain/LangGraph**.  
> **Key Goal**: Integrate classical ML discipline (metrics, latency, data pipelines) with state-of-the-art Generative AI. Move beyond simple API wrappers to architect resilient, low-latency, production-grade **Hybrid RAG systems, Autonomous Multi-Agent Workflows, and Guardrail Defenses**.

---

## 🏛️ Enterprise LLM Application Architecture

```text
                                  CLIENT / USER
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   FastAPI Gateway / Auth  │
                          │   Rate Limiting & PII Mask│
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   Input Guardrails Filter   │
                         │ (Prompt Injection / Safety) │
                         └──────────────┬──────────────┘
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
      ┌─────────────────────────┐               ┌─────────────────────────┐
      │  Advanced Hybrid RAG    │               │ LangGraph Agent State   │
      │ ├── Dense (Vector DB)   │               │ ├── Tool Router         │
      │ ├── Sparse (BM25)       │               │ ├── SQL/Data Lake API   │
      │ ├── Reciprocal Rank Fus │               │ ├── Classical ML Models │
      │ └── Cross-Encoder Rerank│               │ └── Human-in-the-Loop   │
      └────────────┬────────────┘               └────────────┬────────────┘
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   LLM Inference Orchestrator│
                         │ (vLLM / Ollama / OpenAI API)│
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   Output Guardrail / Schema │
                         │ (JSON Pydantic Validation)  │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                                  FINAL ANSWER
```

---

## 1. Advanced Production RAG Architecture

A basic toy RAG pipeline (`retrieve top-k -> stuff prompt`) fails in production due to hallucinations, lost-in-the-middle context decay, and irrelevant retrieval.

### 1.1 The 4-Stage Production Retrieval Pipeline

```text
 1. HYBRID RETRIEVAL       Combine Dense Semantic Vector search with Sparse BM25 Keyword Search.
           │
 2. RECIPROCAL RANK FUSION Blend both ranking lists into a single candidate pool (Top 50).
           │
 3. CROSS-ENCODER RERANK   Compute joint attention between query and candidate chunks (Top 5).
           │
 4. CONTEXT PRUNING        Format selected chunks with metadata citations into LLM context window.
```

### 1.2 Mathematical Reciprocal Rank Fusion (RRF)

Given rankings from Dense search $R_{\text{dense}}$ and Sparse search $R_{\text{sparse}}$:
$$\text{RRF\_Score}(d \in D) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + r_m(d)}$$
where $k \approx 60$ is a smoothing constant preventing high ranks from dominating.

---

## 2. Production Hybrid Search & Reranking Implementation

```python
from typing import List, Dict, Any
import numpy as np

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60, 
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Combines dense semantic vector hits with sparse BM25 hits using RRF.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Rank sparse documents
    for rank, doc in enumerate(sparse_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    # Rank dense documents
    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    # Sort descending by fused RRF score
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]
    
    return [doc_map[doc_id] for doc_id in sorted_doc_ids]
```

---

## 3. Stateful Multi-Agent Orchestration with LangGraph

Unlike fragile linear chains, **LangGraph** models agents as state machines with cyclical edges, persistence, and error recovery:

```python
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END

# 1. Define Agent State Schema
class SecurityAgentState(TypedDict):
    query: str
    threat_level: str
    ip_address: str
    investigation_notes: Annotated[Sequence[str], operator.add]
    next_step: str

# 2. Define Worker Nodes
def triage_node(state: SecurityAgentState):
    notes = [f"Triaged query for IP: {state['ip_address']}"]
    # Inspect telemetry / invoke classical ML model
    threat_level = "CRITICAL" if "192.168.1.100" in state["ip_address"] else "LOW"
    return {"threat_level": threat_level, "investigation_notes": notes}

def deep_investigation_node(state: SecurityAgentState):
    notes = ["Executed reverse DNS and queried CDR database. High anomaly score detected."]
    return {"investigation_notes": notes}

def automated_mitigation_node(state: SecurityAgentState):
    notes = ["Issued firewall block rule on edge gateway."]
    return {"investigation_notes": notes}

# 3. Define Conditional Routing
def route_threat(state: SecurityAgentState) -> str:
    if state["threat_level"] == "CRITICAL":
        return "deep_investigate"
    return END

# 4. Construct Graph
workflow = StateGraph(SecurityAgentState)
workflow.add_node("triage", triage_node)
workflow.add_node("deep_investigate", deep_investigation_node)
workflow.add_node("mitigate", automated_mitigation_node)

workflow.set_entry_point("triage")
workflow.add_conditional_edges("triage", route_threat, {
    "deep_investigate": "deep_investigate",
    END: END
})
workflow.add_edge("deep_investigate", "mitigate")
workflow.add_edge("mitigate", END)

# Compile executable app
agent_app = workflow.compile()
```

---

## 4. Evaluation & Observability (The RAG Triad)

Never evaluate GenAI by subjective visual checks. Quantify quality using the **RAG Triad**:

```text
                             USER QUERY
                            /          \
                           /            \
                Context Relevance      Groundedness / Faithfulness
                         /                \
                        ▼                  ▼
                 RETRIEVED CONTEXT ──► GENERATED ANSWER
```

1. **Context Relevance**: Did the retriever fetch chunks relevant to the user query?
2. **Faithfulness (Groundedness)**: Is the generated answer mathematically grounded in the retrieved chunks, or did the model hallucinate outside facts?
3. **Answer Relevance**: Does the generated answer directly address the original user query?

---

## 5. The Production Enterprise Tech Stack

As an AI Engineer bridging Big Data with AI, your production toolchain is:

- **Serving Framework**: FastAPI (Async, Pydantic v2, OpenAPI)
- **Agent Framework**: LangGraph / LangChain
- **Orchestration & ETL**: Apache Spark + Apache Kafka
- **Vector Storage**: PostgreSQL (`pgvector`), Qdrant, or Milvus
- **Caching & Rate Limiting**: Redis
- **Embedding & Reranking**: HuggingFace `sentence-transformers`, BGE-Reranker
- **Deployment**: Docker, Helm, Kubernetes, AWS EKS / GCP GKE

➡️ **Next Phase**: [14_mlops.md](file:///Users/ervijay/Documents/Programs/Repo/GenAI/AIML/14_mlops.md)
