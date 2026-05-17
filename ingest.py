"""
ingest.py
─────────
Loads technical documents, splits them into chunks, embeds them,
and stores them in a local ChromaDB vector store.

Run once before starting the API:
    python ingest.py
"""

import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP
)

# ── Corpus: URLs to fetch ─────────────────────────────────────────────────────
CORPUS_URLS = [
    ("FastAPI - First Steps",        "https://fastapi.tiangolo.com/tutorial/first-steps/"),
    ("FastAPI - Path Parameters",    "https://fastapi.tiangolo.com/tutorial/path-params/"),
    ("FastAPI - Request Body",       "https://fastapi.tiangolo.com/tutorial/body/"),
    ("FastAPI - Query Parameters",   "https://fastapi.tiangolo.com/tutorial/query-params/"),
    ("FastAPI - Error Handling",     "https://fastapi.tiangolo.com/tutorial/handling-errors/"),
]

# ── Local docs folder (optional) ──────────────────────────────────────────────
LOCAL_DOCS_DIR = Path("./docs")


def fetch_url(title: str, url: str) -> Document:
    """Fetch a web page and return a LangChain Document."""
    print(f"  Fetching: {url}")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove nav / footer noise
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return Document(page_content=text, metadata={"source": url, "title": title})


def load_local_docs() -> list[Document]:
    """Load .txt / .md files from the ./docs directory."""
    docs = []
    if not LOCAL_DOCS_DIR.exists():
        return docs
    for f in LOCAL_DOCS_DIR.iterdir():
        if f.suffix in {".txt", ".md"}:
            print(f"  Loading local file: {f.name}")
            text = f.read_text(encoding="utf-8")
            docs.append(Document(
                page_content=text,
                metadata={"source": f.name, "title": f.stem}
            ))
    return docs


def ingest(extra_docs: list[Document] | None = None) -> int:
    """
    Full ingestion pipeline.
    Returns the total number of chunks stored.
    """
    print("\n[Ingest] Loading documents …")
    raw_docs: list[Document] = []

    # 1. Fetch from URLs
    for title, url in CORPUS_URLS:
        try:
            raw_docs.append(fetch_url(title, url))
        except Exception as e:
            print(f"  ⚠ Could not fetch {url}: {e}")

    # 2. Local files
    raw_docs.extend(load_local_docs())

    # 3. Any docs passed in at runtime (e.g., from the /ingest API endpoint)
    if extra_docs:
        raw_docs.extend(extra_docs)

    if not raw_docs:
        print("[Ingest] ⚠ No documents loaded – aborting.")
        return 0

    print(f"[Ingest] Loaded {len(raw_docs)} document(s). Splitting …")

    # ── Chunking strategy ──────────────────────────────────────────────────
    # RecursiveCharacterTextSplitter respects natural boundaries
    # (paragraphs → sentences → words) which is good for technical prose.
    # chunk_size=800 keeps enough context per chunk without noise;
    # overlap=150 avoids cutting mid-concept.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[Ingest] Created {len(chunks)} chunk(s). Embedding …")

    # ── Embeddings (local, free) ───────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ── Store in ChromaDB ──────────────────────────────────────────────────
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist()
    print(f"[Ingest] ✅ Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DIR}'.")
    return len(chunks)


if __name__ == "__main__":
    ingest()
