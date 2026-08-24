# FastAPI + LangChain Agent Web Application

This project provides a **FastAPI backend** and a **modern interactive Web UI** for a LangChain Agent with thread state memory checkpointer (`InMemorySaver`).

## 📁 Directory Structure

```text
projects/
├── main.py              # FastAPI server with agent endpoints (/api/chat, /api/new_thread)
├── static/
│   └── index.html       # Web UI for chatting with the agent
└── README.md            # Project documentation
```

## 🚀 Quick Start

### 1. Environment Setup
Make sure your `.env` file in the project root contains your Google API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Run the FastAPI Application

From the root repository directory (`GenAI/`), start the FastAPI app using `uvicorn`:

```bash
uv run uvicorn projects.main:app --reload --port 8000
```

or directly inside the `projects/` directory:

```bash
cd projects
uv run uvicorn main:app --reload --port 8000
```

### 3. Open the Web UI

Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧠 Conversation Memory & Architecture

The application uses **LangChain's `create_agent`** and **`InMemorySaver`**:

```python
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[],
    checkpointer=InMemorySaver(),
)
```

### Key Concepts:
1. **Thread Memory (`uuid7`)**: Every chat session is assigned a unique `thread_id`.
2. **Multi-Turn Context**: When you ask follow-up questions (e.g. *"What's the weather in San Francisco?"* followed by *"What about tomorrow?"*), reusing the same `thread_id` allows the agent to recall prior conversation turns seamlessly.
3. **API Endpoints**:
   - `POST /api/chat`: Send user prompt + thread ID.
   - `GET /api/new_thread`: Generate a fresh `thread_id` to start a new chat.
   - `GET /api/history/{thread_id}`: Retrieve stored state messages for debugging or session restoration.
