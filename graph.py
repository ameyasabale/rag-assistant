"""
graph.py
────────
LangGraph StateGraph implementing the self-corrective RAG workflow.

Nodes
─────
1. query_analysis      – rewrites / classifies the user question
2. retrieve            – similarity search in ChromaDB
3. grade_documents     – LLM grades each chunk as relevant / irrelevant
4. generate            – produces the final cited answer
5. hallucination_check – verifies the answer is grounded in context (Self-RAG inspired)

Conditional edges
─────────────────
After grading:
  • relevant docs found  → generate
  • no relevant docs     → rewrite & re-retrieve (up to MAX_RETRIES times)
  • retries exhausted    → fallback response

After hallucination check:
  • grounded     → return answer to user
  • hallucinated → regenerate (up to 2 attempts) → fallback if still failing
"""

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from config import (
    GROQ_API_KEY, LLM_MODEL, CHROMA_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, TOP_K, MAX_RETRIES, GraphState
)

MAX_HALLUCINATION_RETRIES = 2


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0)


def _get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 – Query Analysis
# ─────────────────────────────────────────────────────────────────────────────

QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query analysis assistant for a technical documentation Q&A system.

Given a user question, do two things:
1. Rewrite the question to be clearer and more retrieval-friendly (add synonyms, expand abbreviations, clarify intent). Keep it concise — one sentence.
2. Classify the question type as ONE of: conceptual | how-to | troubleshooting | api-reference

Respond ONLY in this exact format (no extra text):
REWRITTEN: <rewritten question>
TYPE: <type>"""),
    ("human", "{question}"),
])


def query_analysis_node(state: GraphState) -> GraphState:
    print("[Node] query_analysis")
    llm = _get_llm()
    chain = QUERY_ANALYSIS_PROMPT | llm | StrOutputParser()

    question = state["rewritten_query"] or state["question"]
    result = chain.invoke({"question": question})

    rewritten = question
    qtype = "conceptual"
    for line in result.strip().splitlines():
        if line.startswith("REWRITTEN:"):
            rewritten = line.replace("REWRITTEN:", "").strip()
        elif line.startswith("TYPE:"):
            qtype = line.replace("TYPE:", "").strip()

    return {**state, "rewritten_query": rewritten, "query_type": qtype}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 – Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_node(state: GraphState) -> GraphState:
    print("[Node] retrieve")
    retriever = _get_retriever()
    query = state.get("rewritten_query") or state["question"]
    docs = retriever.invoke(query)
    return {**state, "documents": docs}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 – Document Grading
# ─────────────────────────────────────────────────────────────────────────────

GRADING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grader. Given a user question and a document chunk,
decide if the chunk contains information useful to answer the question.
Answer with ONLY one word: relevant  OR  irrelevant"""),
    ("human", "Question: {question}\n\nDocument chunk:\n{chunk}"),
])


def grade_documents_node(state: GraphState) -> GraphState:
    print("[Node] grade_documents")
    llm = _get_llm()
    chain = GRADING_PROMPT | llm | StrOutputParser()

    question = state["question"]
    relevant_docs = []

    for doc in state["documents"]:
        verdict = chain.invoke({
            "question": question,
            "chunk": doc.page_content[:1200],
        }).strip().lower()
        print(f"  grade → {verdict} | src: {doc.metadata.get('source','?')[:60]}")
        if "relevant" in verdict:
            relevant_docs.append(doc)

    outcome = "relevant" if relevant_docs else "irrelevant"
    return {**state, "relevant_docs": relevant_docs, "grading_outcome": outcome}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 – Generation
# ─────────────────────────────────────────────────────────────────────────────

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful technical documentation assistant.
Answer the user's question using ONLY the provided context.
Be concise, accurate, and include a "Sources" section at the end listing the document titles/URLs you used.
If the context does not contain enough information, say so honestly."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nProvide a clear answer followed by a Sources section."),
])


def generate_node(state: GraphState) -> GraphState:
    print("[Node] generate")
    llm = _get_llm()
    chain = GENERATION_PROMPT | llm | StrOutputParser()

    docs = state["relevant_docs"]
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content}"
        for doc in docs
    )
    sources = list({
        doc.metadata.get("source") or doc.metadata.get("title") or "Unknown"
        for doc in docs
    })
    answer = chain.invoke({"context": context, "question": state["question"]})

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "hallucination_count": state.get("hallucination_count", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 – Hallucination Check  (Self-RAG inspired bonus ✨)
# ─────────────────────────────────────────────────────────────────────────────

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a factual grounding checker.

Your job: decide whether an AI-generated answer is fully supported by the provided source documents.

Rules:
- If EVERY factual claim in the answer can be traced back to the context → respond: grounded
- If the answer contains ANY information not present in the context, or invents facts → respond: hallucinated
- Respond with ONLY one word: grounded  OR  hallucinated"""),
    ("human", "Source documents:\n{context}\n\nGenerated answer:\n{answer}"),
])


def hallucination_check_node(state: GraphState) -> GraphState:
    print("[Node] hallucination_check")
    llm = _get_llm()
    chain = HALLUCINATION_PROMPT | llm | StrOutputParser()

    docs = state["relevant_docs"]
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content}"
        for doc in docs
    )
    verdict = chain.invoke({"context": context, "answer": state["answer"]}).strip().lower()
    print(f"  hallucination verdict → {verdict}")

    hallucination_count = state.get("hallucination_count", 0)
    if "hallucinated" in verdict:
        hallucination_count += 1

    return {**state, "hallucination_outcome": verdict, "hallucination_count": hallucination_count}


# ─────────────────────────────────────────────────────────────────────────────
# NODE – Fallback
# ─────────────────────────────────────────────────────────────────────────────

def fallback_node(state: GraphState) -> GraphState:
    print("[Node] fallback")
    return {
        **state,
        "answer": (
            "I'm sorry, I couldn't find reliable information in the documentation "
            "to answer your question accurately. Please try rephrasing, or check the official docs directly."
        ),
        "sources": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE 1 – after document grading
# ─────────────────────────────────────────────────────────────────────────────

def route_after_grading(state: GraphState) -> str:
    if state["grading_outcome"] == "relevant":
        print("[Route] grading → generate")
        return "generate"

    retries = state.get("retry_count", 0)
    if retries < MAX_RETRIES:
        print(f"[Route] grading → query_analysis (retry {retries + 1}/{MAX_RETRIES})")
        state["retry_count"] = retries + 1
        return "query_analysis"

    print("[Route] grading → fallback (max retries reached)")
    return "fallback"


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE 2 – after hallucination check
# ─────────────────────────────────────────────────────────────────────────────

def route_after_hallucination(state: GraphState) -> str:
    outcome = state.get("hallucination_outcome", "grounded")

    if "grounded" in outcome:
        print("[Route] hallucination → END ✅")
        return "end"

    count = state.get("hallucination_count", 0)
    if count < MAX_HALLUCINATION_RETRIES:
        print(f"[Route] hallucination → regenerate (attempt {count}/{MAX_HALLUCINATION_RETRIES})")
        return "generate"

    print("[Route] hallucination → fallback (kept hallucinating)")
    return "fallback"


# ─────────────────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("query_analysis",      query_analysis_node)
    workflow.add_node("retrieve",            retrieve_node)
    workflow.add_node("grade_documents",     grade_documents_node)
    workflow.add_node("generate",            generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("fallback",            fallback_node)

    workflow.set_entry_point("query_analysis")

    workflow.add_edge("query_analysis", "retrieve")
    workflow.add_edge("retrieve",       "grade_documents")

    # Conditional edge 1: after grading
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate":       "generate",
            "query_analysis": "query_analysis",
            "fallback":       "fallback",
        },
    )

    # generate always goes to hallucination check
    workflow.add_edge("generate", "hallucination_check")

    # Conditional edge 2: after hallucination check
    workflow.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {
            "end":      END,
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("fallback", END)

    return workflow.compile()


# Singleton compiled graph
rag_graph = build_graph()


def run_query(question: str) -> dict:
    """Public helper used by the FastAPI and Streamlit layers."""
    initial_state: GraphState = {
        "question":             question,
        "rewritten_query":      None,
        "query_type":           None,
        "documents":            [],
        "relevant_docs":        [],
        "answer":               None,
        "sources":              [],
        "retry_count":          0,
        "grading_outcome":      None,
        "hallucination_outcome": None,
        "hallucination_count":  0,
    }
    final_state = rag_graph.invoke(initial_state)
    return {
        "answer":              final_state["answer"],
        "sources":             final_state["sources"],
        "query_type":          final_state.get("query_type"),
        "rewritten":           final_state.get("rewritten_query"),
        "retries_used":        final_state.get("retry_count", 0),
        "hallucination_check": final_state.get("hallucination_outcome", "grounded"),
    }
