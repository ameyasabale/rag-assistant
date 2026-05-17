import os
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from langchain_core.documents import Document

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = "llama-3.1-8b-instant"

# ── Vector store ──────────────────────────────────────────────────────────────
CHROMA_DIR        = "./chroma_db"
COLLECTION_NAME   = "tech_docs"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # free, local
TOP_K             = 5
MAX_RETRIES       = 2

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150


# ── LangGraph State ───────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question:               str                   # original user question
    rewritten_query:        Optional[str]         # rewritten query (if any)
    query_type:             Optional[str]         # conceptual / how-to / troubleshooting / api-reference
    documents:              List[Document]        # retrieved chunks
    relevant_docs:          List[Document]        # after grading
    answer:                 Optional[str]         # final generated answer
    sources:                List[str]             # source filenames/URLs cited
    retry_count:            int                   # how many re-retrieval attempts so far
    grading_outcome:        Optional[str]         # "relevant" | "irrelevant"
    hallucination_outcome:  Optional[str]         # "grounded" | "hallucinated"
    hallucination_count:    int                   # how many hallucination retries so far
