"""
app.py
──────
FastAPI application serving the RAG assistant.

Endpoints
─────────
POST /query       – Submit a question, get an answer with sources
POST /ingest      – Ingest new documents (file upload or URLs)
GET  /documents   – List all indexed documents
POST /feedback    – Submit thumbs-up/down feedback on an answer
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from graph import run_query
from ingest import ingest

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="Self-corrective RAG system built with LangGraph + FastAPI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory feedback store (good enough for assignment scope)
feedback_store: list[dict] = []


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

    model_config = {"json_schema_extra": {"example": {"question": "How do I define path parameters in FastAPI?"}}}


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    query_type: Optional[str] = None
    rewritten_query: Optional[str] = None
    retries_used: int = 0


class IngestURLRequest(BaseModel):
    urls: list[str]
    model_config = {"json_schema_extra": {"example": {"urls": ["https://fastapi.tiangolo.com/tutorial/body/"]}}}


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str          # "thumbs_up" | "thumbs_down"
    comment: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "How do path parameters work?",
                "answer": "Path parameters are ...",
                "rating": "thumbs_up",
                "comment": "Very clear explanation!",
            }
        }
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "RAG Assistant is running. Visit /docs for API docs."}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """
    Submit a natural language question.
    The LangGraph workflow retrieves relevant chunks, grades them,
    and generates a cited answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = run_query(request.question.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        query_type=result.get("query_type"),
        rewritten_query=result.get("rewritten"),
        retries_used=result.get("retries_used", 0),
    )


@app.post("/ingest", tags=["Ingestion"])
async def ingest_documents(
    files: list[UploadFile] = File(default=[]),
    urls: Optional[str] = Form(default=None),   # comma-separated URLs
):
    """
    Ingest new documents into the vector store.
    Accepts file uploads (.txt / .md) and/or comma-separated URLs.
    """
    extra_docs: list[Document] = []

    # Handle uploaded files
    for upload in files:
        if not upload.filename.endswith((".txt", ".md")):
            raise HTTPException(status_code=400, detail=f"Only .txt and .md files are supported. Got: {upload.filename}")
        content = (await upload.read()).decode("utf-8")
        extra_docs.append(Document(
            page_content=content,
            metadata={"source": upload.filename, "title": Path(upload.filename).stem},
        ))

    # Handle URLs
    if urls:
        from ingest import fetch_url
        for url in [u.strip() for u in urls.split(",") if u.strip()]:
            try:
                doc = fetch_url(title=url, url=url)
                extra_docs.append(doc)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Failed to fetch {url}: {e}")

    if not extra_docs:
        raise HTTPException(status_code=400, detail="No valid documents or URLs provided.")

    try:
        total_chunks = ingest(extra_docs=extra_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    return {
        "status": "success",
        "documents_processed": len(extra_docs),
        "chunks_stored": total_chunks,
    }


@app.get("/documents", tags=["Ingestion"])
def list_documents():
    """
    List all unique source documents currently indexed in the vector store.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        seen = {}
        for meta in metadatas:
            src = meta.get("source", "Unknown")
            if src not in seen:
                seen[src] = meta.get("title", src)

        docs_list = [{"source": src, "title": title} for src, title in seen.items()]
        return {"total_documents": len(docs_list), "documents": docs_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read vector store: {str(e)}")


@app.post("/feedback", tags=["Feedback"])
def submit_feedback(request: FeedbackRequest):
    """
    Submit thumbs-up / thumbs-down feedback on a generated answer.
    """
    if request.rating not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(status_code=400, detail="rating must be 'thumbs_up' or 'thumbs_down'.")

    entry = {
        "id":        str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "question":  request.question,
        "answer":    request.answer,
        "rating":    request.rating,
        "comment":   request.comment,
    }
    feedback_store.append(entry)
    return {"status": "received", "feedback_id": entry["id"]}


# ── Dev server entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
