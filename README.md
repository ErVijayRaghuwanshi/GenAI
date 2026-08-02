# 🤖 GenAI: Learn & Build Applications with Local Ollama Models

Welcome to the **GenAI** repository! This project is a hands-on playground and tutorial workspace designed for learning, experimenting, and building state-of-the-art **Generative AI applications** and **Agentic Systems** leveraging **local Ollama LLM models**.

By running LLMs locally via [Ollama](https://ollama.com), you benefit from **complete data privacy**, **zero API latency/cost**, and **offline development capabilities**, integrated with modern Python AI frameworks like **LangChain**, **LangGraph**, **AutoGen / AG2**, **Pydantic AI**, and **FastRTC**.

---

## 🚀 Environment Setup & Dependency Management (`uv`)

This repository uses [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager written in Rust.

### Prerequisites
- **Python**: `>=3.12`
- **uv**: Installed on your system (`curl -sSf https://astral.sh/uv/install.sh | sh` or `brew install uv`)
- **Ollama**: Installed and running locally (`brew install ollama` or download from [ollama.com](https://ollama.com))

### Quick Start Guide

1. **Clone the Repository & Navigate to Directory**:
   ```bash
   git clone <repository-url>
   cd GenAI
   ```

2. **Synchronize Virtual Environment**:
   Create the `.venv` environment and install all locked dependencies automatically:
   ```bash
   uv sync
   ```

3. **Run Commands or Notebooks**:
   Run scripts directly inside the managed environment:
   ```bash
   uv run python hello.py
   ```
   Or launch Jupyter Notebook/Lab:
   ```bash
   uv run jupyter lab
   ```

4. **Upgrading Libraries**:
   To update all dependencies to their latest compatible versions and sync the virtual environment:
   ```bash
   uv lock --upgrade
   uv sync
   ```

---

## 🦙 Available Local Ollama LLM Models

The local Ollama service powers all LLM inferences across the notebooks and agent applications in this repository. 

### Currently Installed Local Models

| Model Name | Model Tag | Size | Category & Use Case |
| :--- | :--- | :--- | :--- |
| **Gemma 4** | `gemma4:e2b` | `7.2 GB` | High-capability general reasoning, instruction following, and agent tool-use |
| **Nemotron-3 Nano** | `nemotron-3-nano:4b` | `2.8 GB` | Lightweight, fast reasoning model ideal for quick local prototyping & agent tasks |
| **Nomic Embed Text** | `nomic-embed-text:latest` | `274 MB` | High-performance text embeddings for RAG (Retrieval-Augmented Generation) & vector search |

### Managing Ollama Models

- **Check Active Models**:
  ```bash
  ollama list
  ```
- **Run Model Interactively in Terminal**:
  ```bash
  ollama run gemma4:e2b
  ```
- **Pull Additional Models**:
  ```bash
  ollama pull llama3.2
  ollama pull mistral
  ```

---

## 📁 Repository Structure & Modules

```
GenAI/
├── AutoGen_Tutorial/       # Multi-agent orchestrations using AutoGen / AG2 & local Ollama
├── LangGraph_Tutorial/     # Cyclical, state-graph workflow agents using LangGraph
├── Langchain_Tutorial/     # Core LangChain tutorials: LCEL, Memory, Chains, & RAG
├── local-voice-ai-agent/   # Local real-time Voice AI agent (STT + Ollama LLM + Kokoro TTS)
├── ai-ui-agent-mvp/        # Full-stack AI Agent MVP with Web UI backend & frontend
├── main.ipynb              # Interactive experimentation playground notebook
├── hello.py                # Quick verification script
├── pyproject.toml          # Project configuration and dependency specifications
└── uv.lock                 # Deterministic dependency lockfile
```

---

## 🛠 Key Installed Frameworks & Libraries

- **LLM Integrations & Orchestration**:
  - `langchain`, `langchain-core`, `langchain-community`, `langchain-ollama`
  - `langgraph`, `langgraph-checkpoint`, `langgraph-prebuilt`
  - `ag2` (AutoGen), `autogen-agentchat`, `autogen-ext`
  - `pydantic-ai`, `ollama`
- **Real-Time Voice & Audio**:
  - `fastrtc`, `kokoro-onnx`, `soundfile`
- **Web UI & APIs**:
  - `fastapi`, `uvicorn`, `sse-starlette`
- **Tooling & Data Processing**:
  - `pymupdf`, `arxiv`, `wikipedia`, `duckduckgo-search`, `ddgs`
