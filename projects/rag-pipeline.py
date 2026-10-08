"""Quickstart RAG Pipeline for Practice Knowledge Base.

Matches the Jupyter notebook workflow in `projects/rag-pipeline.ipynb`.
"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/rag_db"
COLLECTION_NAME = "practice_rag_kb"
KB_DIR = Path(__file__).resolve().parent / "knowledge_base"


def get_vector_store():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )


def ingest_document(file_path: Path):
    print(f"🎒 Ingesting {file_path.name}...")
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    store = get_vector_store()
    store.add_documents(chunks)
    print(f"✅ Ingested {len(chunks)} chunks into '{COLLECTION_NAME}'.")


def query_rag(user_question: str) -> str:
    store = get_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="nemotron-3-nano:4b", base_url="http://localhost:11434")

    prompt_template = """You are a helpful assistant. Answer the question using ONLY the provided context.
If you don't know the answer based on context, reply: "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n🔍 Querying: '{user_question}'")
    return rag_chain.invoke(user_question)


if __name__ == "__main__":
    sample_file = KB_DIR / "refund_policy.md"
    if sample_file.exists():
        try:
            ingest_document(sample_file)
            ans = query_rag("How many days do refunds take to process?")
            print(f"\n🤖 Answer:\n{ans}")
        except Exception as e:
            print(f"Note: Vector store or Ollama unavailable: {e}")
