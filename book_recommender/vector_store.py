"""Vector store creation and loading.

Wraps the Chroma vector store so the rest of the app does not need to know how
embeddings are produced or where the database lives. The store is persisted to
disk and reused on subsequent runs to avoid re-embedding the whole catalogue
(and re-paying for embeddings).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .config import settings
from .data import create_documents

logger = logging.getLogger(__name__)


def build_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    """Create the OpenAI embeddings client used for indexing and querying."""
    return OpenAIEmbeddings(model=model or settings.embedding_model)


def get_vector_store(
    books: pd.DataFrame,
    chroma_dir: Path | str | None = None,
    embeddings: OpenAIEmbeddings | None = None,
) -> Chroma:
    """Load an existing Chroma store, or build and persist one from ``books``.

    Args:
        books: Prepared books DataFrame (see :func:`book_recommender.data.load_books`).
        chroma_dir: Directory for the persisted store. Defaults to config.
        embeddings: Embeddings client. Defaults to a fresh OpenAI client.

    Returns:
        A ready-to-query Chroma vector store.
    """
    chroma_dir = Path(chroma_dir) if chroma_dir is not None else settings.chroma_dir
    embeddings = embeddings or build_embeddings()

    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        logger.info("Loading existing Chroma database from %s", chroma_dir)
        return Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
        )

    logger.info("Creating new Chroma database at %s", chroma_dir)
    documents = create_documents(books)
    if not documents:
        raise ValueError("No documents to index: the books dataset has no usable descriptions.")

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
    )
