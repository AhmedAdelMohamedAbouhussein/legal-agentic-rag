"""
chunker.py — Text splitting strategies for legal documents.
"""

from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

STRATEGIES = ["recursive", "overlapping", "fixed", "token"]

DESCRIPTIONS = {
    "recursive": "Splits on paragraph/sentence/word boundaries. Best general-purpose choice.",
    "overlapping": "Fixed-size chunks with configurable overlap. Good for long clauses.",
    "fixed": "Strictly fixed character count. Fast but may split mid-sentence.",
    "token": "Splits on token boundaries aligned to LLM context windows.",
}


def get_strategy_description(strategy: str) -> str:
    return DESCRIPTIONS.get(strategy, "")


def chunk_documents(
    docs: List[Document],
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split documents into chunks using the chosen strategy."""

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif strategy == "overlapping":
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "fixed":
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=0,
        )
    elif strategy == "token":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size // 4,   # ~4 chars per token
            chunk_overlap=chunk_overlap // 4,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return splitter.split_documents(docs)
