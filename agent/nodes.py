"""
nodes.py — LangGraph node functions for the Legal AI Agent.

Flow:
  classify_query → retrieve → assess_sufficiency
    → (rewrite → retrieve)* [max 2 retries]
    → generate_answer
"""

from __future__ import annotations
import json
from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .state import AgentState

MAX_LOOPS = 2


# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """You are a legal AI routing assistant.
Classify the user's legal question into ONE of these categories:
  - case_law    : about court decisions, precedents, rulings, verdicts
  - contracts   : about contract clauses, agreements, obligations, rights
  - both        : question touches both case law and contract documents
  - general     : general legal concept requiring no document retrieval

Also generate 1-3 optimised search queries (short, keyword-dense phrases).

Respond ONLY with valid JSON:
{{
  "query_type": "<category>",
  "search_queries": ["<query1>", "<query2>"]
}}

User question: {question}
"""

SUFFICIENCY_PROMPT = """You are a legal research quality controller.
Review the retrieved context and the user's question.
Decide: does the context contain ENOUGH specific facts to answer the question completely?

Question: {question}

Retrieved context (first 3000 chars):
{context}

Respond ONLY with valid JSON:
{{
  "is_sufficient": true/false,
  "reason": "<one sentence>"
}}
"""

REWRITE_PROMPT = """You are a legal search query optimizer.
The previous retrieval returned insufficient context for this question.
Generate 2 improved, more specific search queries.

Original question: {question}
Previous queries: {previous_queries}

Respond ONLY with valid JSON:
{{
  "search_queries": ["<improved_query1>", "<improved_query2>"]
}}
"""

ANSWER_PROMPT = """You are a highly skilled legal research assistant. Answer based STRICTLY on the provided context.

RULES:
1. Cite sources after each factual claim using: [Source: filename, p.X]
2. Structure your answer with clear headings for complex questions
3. If context is insufficient, say so explicitly
4. Always end with: ⚠️ *This is research assistance only. Consult a licensed attorney.*

Question: {question}

Context:
{context}

Chat History:
{chat_history}

Provide a thorough, well-cited legal analysis:"""


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs):
        fname = d.metadata.get("filename", "unknown")
        page  = d.metadata.get("page", "?")
        parts.append(f"[Source: {fname}, p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _parse_json(text: str) -> dict:
    """Extract first JSON object from LLM output."""
    start = text.find("{")
    end   = text.rfind("}") + 1
    return json.loads(text[start:end])


# ── Node factory ──────────────────────────────────────────────────────────────

def create_nodes(
    case_law_retriever,
    contracts_retriever,
    general_retriever,
    llm: ChatOpenAI,
):
    """Return a dict of node callables closed over retrievers and llm."""

    # ── Node 1: classify ──────────────────────────────────────────────────────
    def classify_node(state: AgentState) -> dict:
        prompt = CLASSIFY_PROMPT.format(question=state["question"])
        response = llm.invoke(prompt)
        try:
            data = _parse_json(response.content)
            query_type    = data.get("query_type", "both")
            search_queries = data.get("search_queries", [state["question"]])
        except Exception:
            query_type    = "both"
            search_queries = [state["question"]]

        return {
            "query_type": query_type,
            "search_queries": search_queries,
            "loop_count": 0,
            "steps": [f"🔍 Classified as **{query_type}** | Queries: {search_queries}"],
        }

    # ── Node 2: retrieve ──────────────────────────────────────────────────────
    def retrieve_node(state: AgentState) -> dict:
        qt      = state.get("query_type", "both")
        queries = state.get("search_queries", [state["question"]])
        docs: List[Document] = []

        for q in queries:
            if qt in ("case_law", "both"):
                docs.extend(case_law_retriever.invoke(q))
            if qt in ("contracts", "both"):
                docs.extend(contracts_retriever.invoke(q))
            if qt == "general":
                docs.extend(general_retriever.invoke(q))

        # Deduplicate by page_content
        seen, unique = set(), []
        for d in docs:
            key = d.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(d)

        return {
            "retrieved_docs": unique,
            "loop_count": state.get("loop_count", 0) + 1,
            "steps": [f"📚 Retrieved **{len(unique)}** unique chunks (loop {state.get('loop_count', 0) + 1})"],
        }

    # ── Node 3: assess sufficiency ────────────────────────────────────────────
    def assess_node(state: AgentState) -> dict:
        docs    = state.get("retrieved_docs", [])
        context = _format_docs(docs)[:3000]
        prompt  = SUFFICIENCY_PROMPT.format(
            question=state["question"],
            context=context,
        )
        response = llm.invoke(prompt)
        try:
            data         = _parse_json(response.content)
            is_sufficient = data.get("is_sufficient", True)
            reason        = data.get("reason", "")
        except Exception:
            is_sufficient = True
            reason        = "Could not parse sufficiency check."

        icon = "✅" if is_sufficient else "⚠️"
        return {
            "is_sufficient": is_sufficient,
            "steps": [f"{icon} Sufficiency: {'sufficient' if is_sufficient else 'insufficient'} — {reason}"],
        }

    # ── Node 4: rewrite query ─────────────────────────────────────────────────
    def rewrite_node(state: AgentState) -> dict:
        prompt = REWRITE_PROMPT.format(
            question=state["question"],
            previous_queries=state.get("search_queries", []),
        )
        response = llm.invoke(prompt)
        try:
            data = _parse_json(response.content)
            new_queries = data.get("search_queries", [state["question"]])
        except Exception:
            new_queries = [state["question"] + " legal precedent statute"]

        return {
            "search_queries": new_queries,
            "steps": [f"✏️ Rewriting queries → {new_queries}"],
        }

    # ── Node 5: generate answer ───────────────────────────────────────────────
    def generate_node(state: AgentState) -> dict:
        docs    = state.get("retrieved_docs", [])
        context = _format_docs(docs)
        history = state.get("chat_history", [])
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
        )
        prompt  = ANSWER_PROMPT.format(
            question=state["question"],
            context=context,
            chat_history=history_text or "None",
        )
        response = llm.invoke(prompt)
        return {
            "answer": response.content,
            "steps": [f"💬 Answer generated ({len(response.content)} chars)"],
        }

    return {
        "classify":  classify_node,
        "retrieve":  retrieve_node,
        "assess":    assess_node,
        "rewrite":   rewrite_node,
        "generate":  generate_node,
    }
