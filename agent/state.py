"""
state.py — LangGraph AgentState definition.
"""

from __future__ import annotations
import operator
from typing import Annotated, List, TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    # Core conversation
    question: str
    chat_history: List[dict]           # [{role, content}, ...]

    # Routing
    query_type: str                    # "case_law" | "contracts" | "both" | "general"
    search_queries: List[str]          # Generated search queries

    # Retrieval
    retrieved_docs: List[Document]

    # Loop control
    loop_count: int
    is_sufficient: bool

    # Output
    answer: str

    # Observability — lists are appended across nodes
    steps: Annotated[List[str], operator.add]
