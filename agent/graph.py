"""
graph.py — Compile the LangGraph stateful agent.
"""

from __future__ import annotations
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import create_nodes, MAX_LOOPS


def _route_after_assess(state: AgentState) -> str:
    """Conditional edge: retry or proceed to generate."""
    if state.get("is_sufficient") or state.get("loop_count", 0) >= MAX_LOOPS:
        return "generate"
    return "rewrite"


def build_agent(
    case_law_retriever,
    contracts_retriever,
    general_retriever,
    llm,
):
    """Build and compile the LangGraph agent. Returns a CompiledGraph."""
    nodes = create_nodes(case_law_retriever, contracts_retriever, general_retriever, llm)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify",  nodes["classify"])
    graph.add_node("retrieve",  nodes["retrieve"])
    graph.add_node("assess",    nodes["assess"])
    graph.add_node("rewrite",   nodes["rewrite"])
    graph.add_node("generate",  nodes["generate"])

    # Edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges(
        "assess",
        _route_after_assess,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
