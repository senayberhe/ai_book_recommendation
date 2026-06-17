"""Data loading and document preparation.

This module is responsible for reading the cleaned books dataset and turning
book descriptions into LangChain ``Document`` objects for embedding. It contains
no network or model dependencies, which keeps it fast and easy to unit-test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.documents import Document

from .config import settings


def load_books(
    csv_path: Path | str | None = None,
    cover_not_found: str | None = None,
) -> pd.DataFrame:
    """Load the books dataset and add display-ready columns.

    Args:
        csv_path: Path to the books CSV. Defaults to the configured location.
        cover_not_found: Placeholder image used when a thumbnail is missing.

    Returns:
        A DataFrame with a high-resolution ``large_thumbnail`` column and an
        ``isbn13`` column coerced to ``str`` for safe joins.
    """
    csv_path = Path(csv_path) if csv_path is not None else settings.books_csv
    cover_not_found = cover_not_found or settings.cover_not_found

    books = pd.read_csv(csv_path)
    return prepare_books(books, cover_not_found)


def prepare_books(books: pd.DataFrame, cover_not_found: str | None = None) -> pd.DataFrame:
    """Add derived display columns to a raw books DataFrame.

    Separated from :func:`load_books` so it can be tested without touching disk.
    """
    cover_not_found = cover_not_found or settings.cover_not_found
    books = books.copy()

    if "thumbnail" in books.columns:
        books["large_thumbnail"] = np.where(
            books["thumbnail"].notna(),
            books["thumbnail"].astype(str) + "&fife=w800",
            cover_not_found,
        )
    else:
        books["large_thumbnail"] = cover_not_found

    books["isbn13"] = books["isbn13"].astype(str)
    return books


def create_documents(books: pd.DataFrame) -> list[Document]:
    """Convert book descriptions into LangChain documents for embedding.

    Rows with an empty or missing description are skipped. Each document keeps
    the ISBN, title, and simplified category in its metadata so results can be
    mapped back to the full record after a semantic search.
    """
    documents: list[Document] = []

    for _, row in books.iterrows():
        description = str(row.get("description", "")).strip()
        if not description or description.lower() == "nan":
            continue

        documents.append(
            Document(
                page_content=description,
                metadata={
                    "isbn13": str(row["isbn13"]),
                    "title": str(row.get("title", "")),
                    "category": str(row.get("simple_categories", "")),
                },
            )
        )

    return documents
