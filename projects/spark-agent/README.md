# FastAPI + LangChain SparkLens AI Agent

This project provides a **FastAPI backend** and a **responsive Web UI** tailored for debugging and tuning Apache Spark applications. It dynamically integrates tools exposed by the **SparkLens MCP Server** to provide diagnostic summaries, failure analyses, resource skew detection, and configuration optimization tips.

## 📁 Directory Structure

```text
projects/spark-agent/
├── main.py              # FastAPI server with dynamic MCP tool registration
├── static/
│   └── index.html       # Diagnostic Dashboard UI
└── README.md            # Project documentation
```

## 🚀 Quick Start

### 1. Pre-requisites & Environment
Make sure your `.env` file in the project root (`GenAI/`) contains your Gemini API key and points to the running SparkLens MCP server:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
SPARK_MCP_URL=http://localhost:8030/sse
```

### 2. Verify SparkLens MCP Server is running
Ensure the SparkLens FastMCP server is active:
```bash
# Inside spark-lens repo
uv run --env-file .env python -m sparklens.server
```
The server will default to SSE transport on `http://localhost:8030/sse`.

### 3. Run the FastAPI Application
From the root repository directory (`GenAI/`), start the FastAPI app using `uvicorn`:

```bash
uv run uvicorn projects.spark-agent.main:app --reload --port 8000
```

### 4. Open the Web UI
Navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🩺 Diagnostic Capabilities

Using the conversational UI, you can query:
- `"List the completed spark applications"`
- `"Diagnose application health for app-20260722180849-0000"`
- `"Did any stages fail in app-20260722180849-0000?"`
- `"Analyze memory metrics or task-skew details for stages in local-1783802678869"`
