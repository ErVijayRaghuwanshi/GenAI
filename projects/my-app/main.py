import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

import dotenv
dotenv.load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

# Disable LangSmith tracing warnings if key is missing/invalid
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

app = FastAPI(
    title="LangChain Agent API & UI",
    description="FastAPI backend serving a LangChain agent with InMemorySaver memory checkpointer.",
    version="1.0.0"
)

# Initialize the agent and checkpointer as specified in prompt
checkpointer = InMemorySaver()

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "google_genai:gemini-3.6-flash")

try:
    agent = create_agent(
        model="google_genai:gemini-3.1-flash-lite",
        tools=[get_weather],
        checkpointer=checkpointer,
    )
except Exception as err:
    print(f"Warning: Initializing agent with {DEFAULT_MODEL} failed: {err}")
    # Fallback to gemini-2.0-flash if specified model fails
    agent = create_agent(
        model="google_genai:gemini-2.0-flash",
        tools=[get_weather],
        checkpointer=checkpointer,
    )

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message prompt")
    thread_id: Optional[str] = Field(None, description="Thread UUID to maintain conversation history")

class ChatResponse(BaseModel):
    response: str
    thread_id: str

class ThreadResponse(BaseModel):
    thread_id: str

@app.get("/api/new_thread", response_model=ThreadResponse)
def get_new_thread():
    """Generates a new unique thread ID using uuid7."""
    new_id = str(uuid7())
    return ThreadResponse(thread_id=new_id)

def extract_text_from_content(content: Any) -> str:
    """Extract clean string content from string, list of blocks, or dict."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    text_parts.append(part["text"])
                elif "text" in part:
                    text_parts.append(str(part["text"]))
            elif hasattr(part, "text"):
                text_parts.append(str(part.text))
        if text_parts:
            return "\n".join(text_parts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
    return str(content)

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Sends a message to the agent.
    If no thread_id is provided, a new thread_id (uuid7) is generated.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    thread_id = request.thread_id.strip() if request.thread_id else str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.message.strip()}]},
            config=config,
        )

        # Extract agent response content
        messages = result.get("messages", [])
        if not messages:
            response_text = "No response generated."
        else:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                response_text = extract_text_from_content(last_message.content)
            elif isinstance(last_message, dict) and "content" in last_message:
                response_text = extract_text_from_content(last_message["content"])
            else:
                response_text = extract_text_from_content(last_message)

        return ChatResponse(response=response_text, thread_id=thread_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

@app.get("/api/history/{thread_id}")
def get_thread_history(thread_id: str):
    """
    Retrieves the conversation state history for a specific thread_id from the checkpointer.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
        messages_data = []
        if state and state.values and "messages" in state.values:
            for msg in state.values["messages"]:
                role = "assistant"
                content = ""
                if hasattr(msg, "type"):
                    role = "user" if msg.type in ("human", "user") else "assistant"
                elif hasattr(msg, "role"):
                    role = msg.role
                
                if hasattr(msg, "content"):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get("content", "")

                messages_data.append({"role": role, "content": str(content)})

        return {"thread_id": thread_id, "messages": messages_data}
    except Exception as e:
        return {"thread_id": thread_id, "messages": [], "error": str(e)}

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the web chat application UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>LangChain Agent API Server</h1><p>UI file index.html not found.</p>")
