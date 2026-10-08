"""Modular RAG Ingestion, Query, and Evaluation Pipeline for Practice Knowledge Base.

Supports multi-format documents (.md, .txt, .json) with metadata enrichment,
vector storage via PostgreSQL/pgvector, and local LLMs via Ollama.
"""

import argparse
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional LangChain & Vector DB imports with graceful fallbacks
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_postgres import PGVector
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/rag_db"
COLLECTION_NAME = "practice_rag_kb"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "nemotron-3-nano:4b"
OLLAMA_BASE_URL = "http://localhost:11434"


def load_knowledge_base(kb_dir: Path) -> List[Document]:
    """Loads markdown, text, and JSON files from the knowledge base directory,

    enriching each chunk with metadata.
    """
    documents: List[Document] = []

    # 1. Load Markdown documents
    for md_file in kb_dir.glob("*.md"):
        if md_file.name == "README.md":
            continue
        content = md_file.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": md_file.name,
                    "file_type": "markdown",
                    "category": "documentation" if "api" in md_file.name or "auth" in md_file.name else "customer_support",
                },
            )
        )

    # 2. Load Plain Text documents
    for txt_file in kb_dir.glob("*.txt"):
        content = txt_file.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": txt_file.name,
                    "file_type": "plain_text",
                    "category": "troubleshooting" if "troubleshoot" in txt_file.name else "operations",
                },
            )
        )

    # 3. Load Structured JSON documents (excluding evaluation benchmark)
    for json_file in kb_dir.glob("*.json"):
        if "benchmark" in json_file.name or "eval" in json_file.name:
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if json_file.name == "product_pricing.json":
                # Convert structured pricing into fact-dense natural language text
                plans = data.get("plans", [])
                for plan in plans:
                    text_repr = (
                        f"Subscription Plan: {plan.get('tier')}\n"
                        f"Monthly Price: ${plan.get('monthly_price')}/month\n"
                        f"Annual Price: ${plan.get('annual_price_per_month')}/month (20% discount applied)\n"
                        f"User Seats: {plan.get('user_seats')}\n"
                        f"Cloud Storage: {plan.get('storage_gb')} GB\n"
                        f"Support Tier: {plan.get('support_tier')}\n"
                        f"API Access: {'Yes' if plan.get('api_access') else 'No'}\n"
                        f"API Rate Limit: {plan.get('api_rate_limit_per_minute', 'N/A')} req/min\n"
                        f"Key Features: {', '.join(plan.get('features', []))}"
                    )
                    documents.append(
                        Document(
                            page_content=text_repr,
                            metadata={
                                "source": json_file.name,
                                "file_type": "json",
                                "tier": plan.get("tier"),
                                "category": "pricing",
                            },
                        )
                    )
                add_ons = data.get("add_ons", [])
                add_ons_text = "Add-On Pricing Options:\n" + "\n".join(
                    [f"- {item.get('name')}: ${item.get('price_monthly')}/month" for item in add_ons]
                )
                documents.append(
                    Document(
                        page_content=add_ons_text,
                        metadata={
                            "source": json_file.name,
                            "file_type": "json",
                            "category": "pricing_addons",
                        },
                    )
                )

            elif json_file.name == "system_incidents.json":
                # Convert incident logs into individual post-mortem documents
                for inc in data:
                    inc_text = (
                        f"Incident ID: {inc.get('incident_id')}\n"
                        f"Date: {inc.get('date')}\n"
                        f"Affected Service: {inc.get('affected_service')}\n"
                        f"Severity: {inc.get('severity')}\n"
                        f"Downtime: {inc.get('downtime_minutes')} minutes\n"
                        f"Root Cause: {inc.get('root_cause')}\n"
                        f"Resolution: {inc.get('resolution')}"
                    )
                    documents.append(
                        Document(
                            page_content=inc_text,
                            metadata={
                                "source": json_file.name,
                                "file_type": "json",
                                "incident_id": inc.get("incident_id"),
                                "category": "system_incidents",
                            },
                        )
                    )
            else:
                documents.append(
                    Document(
                        page_content=json.dumps(data, indent=2),
                        metadata={"source": json_file.name, "file_type": "json", "category": "data"},
                    )
                )
        except Exception as e:
            print(f"⚠️ Error parsing {json_file.name}: {e}")

    return documents


def get_vector_store():
    """Initializes and returns the pgvector store."""
    if not HAS_RAG_DEPS:
        raise RuntimeError("LangChain dependencies are missing. Run with 'uv run python ...'")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )


def ingest_kb():
    """Loads all knowledge base files, splits into chunks, and saves to pgvector."""
    print(f"📂 Loading documents from: {KB_DIR}")
    raw_docs = load_knowledge_base(KB_DIR)
    print(f"📄 Loaded {len(raw_docs)} base documents/sections.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
    print(f"✂️  Generated {len(chunks)} text chunks for embedding.")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Preview Chunk #{i+1} [Source: {chunk.metadata.get('source')}] ---")
        print(chunk.page_content.strip()[:180] + ("..." if len(chunk.page_content) > 180 else ""))

    try:
        vector_store = get_vector_store()
        print(f"\n📦 Ingesting {len(chunks)} chunks into pgvector (collection: '{COLLECTION_NAME}')...")
        vector_store.add_documents(chunks)
        print(f"✅ Ingestion complete! Chunks successfully stored in pgvector.")
    except Exception as e:
        print(f"\n⚠️  Could not connect to pgvector database: {e}")
        print("💡 Note: Ensure PostgreSQL is running on port 5432 with pgvector installed.")
        return chunks

    return chunks


def build_rag_chain():
    """Constructs LCEL RAG chain."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    prompt_template = """You are a precise, helpful knowledge base assistant. Answer the user's question using ONLY the facts present in the provided context below.
If the answer cannot be determined from the context, respond strictly with: "I don't know based on the provided context." Do not make up answers.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def run_eval():
    """Loads rag_eval_benchmark.json and tests retrieval and generation."""
    bench_file = KB_DIR / "rag_eval_benchmark.json"
    if not bench_file.exists():
        print("❌ Benchmark file not found.")
        return

    with open(bench_file, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print(f"\n🧪 Running RAG Evaluation Suite ({len(benchmarks)} Test Cases)")
    print("=" * 70)

    try:
        rag_chain = build_rag_chain()
    except Exception as e:
        print(f"⚠️  Database or LLM connection unavailable for live inference: {e}")
        print("Displaying benchmark specifications:")
        for case in benchmarks:
            print(f"\n[{case['id']}] Category: {case['category']}")
            print(f"  Q: {case['question']}")
            print(f"  Expected: {case['ground_truth']}")
            print(f"  Source: {case['source_file']}")
        return

    for case in benchmarks:
        q = case["question"]
        print(f"\n🔹 Test ID: {case['id']} ({case['category']})")
        print(f"   Query: {q}")
        print(f"   Target Source: {case['source_file']}")
        print(f"   Ground Truth: {case['ground_truth']}")
        try:
            res = rag_chain.invoke(q)
            print(f"   🤖 Model Output:\n{res.strip()}")
        except Exception as err:
            print(f"   ❌ Query execution failed: {err}")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Practice RAG Knowledge Base Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Ingest all knowledge base documents into pgvector")
    parser.add_argument("--eval", action="store_true", help="Run the benchmark evaluation suite")
    parser.add_argument("--query", type=str, help="Run an ad-hoc query through the RAG chain")
    parser.add_argument("--dry-run", action="store_true", help="Parse and preview chunking without vector store insertion")
    args = parser.parse_args()

    if args.dry_run:
        print("🧪 Performing Dry-Run Document Loading & Chunking:")
        raw_docs = load_knowledge_base(KB_DIR)
        print(f"Total base sections: {len(raw_docs)}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(raw_docs)
        print(f"Total chunks produced: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"  Chunk {i+1:02d}: [{c.metadata.get('source')}] ({len(c.page_content)} chars)")
        return

    if args.ingest:
        ingest_kb()
    elif args.eval:
        run_eval()
    elif args.query:
        rag_chain = build_rag_chain()
        print(f"\n🔍 Query: '{args.query}'\n")
        ans = rag_chain.invoke(args.query)
        print(f"🤖 Answer:\n{ans}")
    else:
        # Default: Dry-run preview if no flags passed
        parser.print_help()


if __name__ == "__main__":
    main()
