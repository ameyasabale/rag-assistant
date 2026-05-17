# RAG-Based Technical Documentation Assistant

A **self-corrective Retrieval-Augmented Generation (RAG)** system built with **LangGraph**, **FastAPI**, and **ChromaDB**. It answers questions about technical documentation using an agentic workflow that grades retrieved documents and retries with a rewritten query when results are irrelevant.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│  Query Analysis │  ← rewrites query, classifies type
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Retrieval    │  ← ChromaDB similarity search (top-k chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Grader │  ← LLM grades each chunk: relevant / irrelevant
└────────┬────────┘
         │
    ┌────┴──────────────────┐
    │ Relevant?             │
   YES                      NO
    │                       │
    ▼              retries < MAX_RETRIES?
┌──────────┐           │          │
│ Generate │          YES          NO
└──────────┘           │          │
    │                  ▼          ▼
    ▼          Query Analysis  Fallback
 Answer            (loop)      Response
```

### State Schema (`GraphState`)
| Field | Type | Purpose |
|---|---|---|
| `question` | str | Original user question |
| `rewritten_query` | str | LLM-rewritten query for better retrieval |
| `query_type` | str | conceptual / how-to / troubleshooting / api-reference |
| `documents` | List[Document] | Raw retrieved chunks |
| `relevant_docs` | List[Document] | Chunks that passed grading |
| `answer` | str | Final generated answer |
| `sources` | List[str] | Source URLs/filenames cited |
| `retry_count` | int | Tracks re-retrieval attempts |
| `grading_outcome` | str | "relevant" or "irrelevant" |

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangGraph (`StateGraph`) |
| LLM | Groq (llama-3.1-8b-instant) — free tier |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 — local, free |
| Vector Store | ChromaDB — local |
| API Framework | FastAPI |
| Document Corpus | FastAPI official documentation (5 pages) |

---

## Document Corpus

The system ingests 5 pages from the FastAPI official documentation:
- First Steps
- Path Parameters
- Query Parameters
- Request Body
- Error Handling

You can add your own `.txt` or `.md` files to the `./docs/` folder or POST to `/ingest`.

---

## Setup Instructions (Windows)

### 1. Prerequisites
- Python 3.11+ — download from [python.org](https://python.org)
- Git — download from [git-scm.com](https://git-scm.com)

### 2. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant
```

### 3. Create a virtual environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies
```cmd
pip install -r requirements.txt
```

### 5. Set up environment variables
```cmd
copy .env.example .env
```
Open `.env` in Notepad and replace `your_groq_api_key_here` with your actual Groq API key from [console.groq.com](https://console.groq.com).

### 6. Ingest the document corpus
```cmd
python ingest.py
```
This fetches the FastAPI docs, chunks them, embeds them locally, and stores them in ChromaDB. Run once.

### 7. Start the API server
```cmd
python app.py
```
Server runs at: **http://localhost:8000**  
Interactive API docs: **http://localhost:8000/docs**

---

## API Endpoints

### POST `/query` — Ask a question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define path parameters in FastAPI?"}'
```
**Response:**
```json
{
  "answer": "In FastAPI, path parameters are defined by placing a variable name inside curly braces in the path...\n\nSources:\n- https://fastapi.tiangolo.com/tutorial/path-params/",
  "sources": ["https://fastapi.tiangolo.com/tutorial/path-params/"],
  "query_type": "how-to",
  "rewritten_query": "How to declare and use path parameters in FastAPI routes?",
  "retries_used": 0
}
```

### POST `/ingest` — Add new documents
```bash
# Via URL
curl -X POST http://localhost:8000/ingest \
  -F "urls=https://fastapi.tiangolo.com/tutorial/response-model/"

# Via file upload
curl -X POST http://localhost:8000/ingest \
  -F "files=@my_docs.md"
```

### GET `/documents` — List indexed documents
```bash
curl http://localhost:8000/documents
```
**Response:**
```json
{
  "total_documents": 5,
  "documents": [
    {"source": "https://fastapi.tiangolo.com/tutorial/first-steps/", "title": "FastAPI - First Steps"},
    ...
  ]
}
```

### POST `/feedback` — Submit feedback
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do path parameters work?",
    "answer": "Path parameters are defined using curly braces...",
    "rating": "thumbs_up",
    "comment": "Very clear!"
  }'
```

---

## Design Decisions & Tradeoffs

### Chunking Strategy
- **Size: 800 tokens, Overlap: 150 tokens**
- `RecursiveCharacterTextSplitter` respects natural text boundaries (paragraphs → sentences → words)
- 800 chars balances context richness vs. retrieval noise; smaller chunks (300–400) lose context for technical explanations
- 150-char overlap avoids cutting concepts mid-sentence across chunk boundaries

### Embedding Model
- `all-MiniLM-L6-v2` — chosen because it's free, runs locally (no API cost), fast, and performs well on English technical text
- Tradeoff: OpenAI `text-embedding-3-small` would give higher quality but costs money

### LLM (Groq + Llama 3.1 8B)
- Free tier, very fast inference (~300 tokens/sec)
- Tradeoff: GPT-4o would give higher reasoning quality but costs money

### Self-Correction Logic
- MAX_RETRIES = 2 — prevents infinite loops while allowing meaningful recovery
- On retry: the query analysis node rewrites the query differently, then retrieves again
- Fallback response after retries exhausted is honest ("I couldn't find information")

### What I Would Improve With More Time
1. **Hallucination check node** — verify the generated answer is grounded in the retrieved context
2. **Web search fallback** — use Tavily API when vector store has no relevant results
3. **Conversation memory** — maintain chat history for follow-up questions
4. **Persistent feedback storage** — save to SQLite instead of in-memory list
5. **Better chunking** — semantic chunking based on meaning rather than character count
6. **Streamlit UI** — simple frontend for non-technical users

---

## Project Structure

```
rag-assistant/
├── app.py              # FastAPI application (4 endpoints)
├── graph.py            # LangGraph StateGraph (4 nodes + routing)
├── ingest.py           # Document ingestion pipeline
├── config.py           # Config, constants, GraphState schema
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .env                # Your actual env vars (not committed)
├── docs/               # Optional: place your own .txt/.md files here
├── chroma_db/          # ChromaDB persisted storage (auto-created)
└── README.md
```

---

## License
MIT
