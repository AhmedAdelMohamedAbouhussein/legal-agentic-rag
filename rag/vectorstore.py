"""
vectorstore.py — ChromaDB vector store management.
Two collections: 'case_law' and 'contracts'.
"""

from __future__ import annotations
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

CASE_LAW_COLLECTION = "legal_case_law"
CONTRACTS_COLLECTION = "legal_contracts"
GENERAL_COLLECTION   = "legal_general"


def get_embeddings(openai_api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )


def _split_by_category(docs: List[Document]):
    case_law, contracts, general = [], [], []
    for d in docs:
        cat = d.metadata.get("category", "general")
        if cat == "case_law":
            case_law.append(d)
        elif cat == "contracts":
            contracts.append(d)
        else:
            general.append(d)
    return case_law, contracts, general


def build_vector_stores(
    chunks: List[Document],
    embeddings: OpenAIEmbeddings,
    persist_dir: str = "./chroma_db",
) -> Tuple[Chroma, Chroma, Chroma]:
    """Embed and persist three Chroma collections."""
    case_law, contracts, general = _split_by_category(chunks)

    # Always build at least an empty store if no docs in that category
    cl_store = Chroma.from_documents(
        documents=case_law if case_law else [Document(page_content="placeholder", metadata={"category": "case_law"})],
        embedding=embeddings,
        collection_name=CASE_LAW_COLLECTION,
        persist_directory=persist_dir,
    )
    ct_store = Chroma.from_documents(
        documents=contracts if contracts else [Document(page_content="placeholder", metadata={"category": "contracts"})],
        embedding=embeddings,
        collection_name=CONTRACTS_COLLECTION,
        persist_directory=persist_dir,
    )
    gn_store = Chroma.from_documents(
        documents=general if general else [Document(page_content="placeholder", metadata={"category": "general"})],
        embedding=embeddings,
        collection_name=GENERAL_COLLECTION,
        persist_directory=persist_dir,
    )
    return cl_store, ct_store, gn_store


def load_vector_stores(
    embeddings: OpenAIEmbeddings,
    persist_dir: str = "./chroma_db",
) -> Tuple[Chroma, Chroma, Chroma]:
    """Load existing persisted Chroma collections."""
    cl_store = Chroma(
        collection_name=CASE_LAW_COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    ct_store = Chroma(
        collection_name=CONTRACTS_COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    gn_store = Chroma(
        collection_name=GENERAL_COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return cl_store, ct_store, gn_store
