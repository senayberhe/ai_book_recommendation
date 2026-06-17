"""The recommendation engine.

Given a free-text query, the engine performs semantic search over book
descriptions, maps the hits back to the full catalogue, optionally filters by
category, and re-ranks by emotional tone. The vector store is injected, so the
engine can be tested with a lightweight fake instead of a live OpenAI/Chroma
stack.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd
from langchain_core.documents import Document

from .config import TONE_TO_EMOTION, settings


class VectorStore(Protocol):
    """Minimal interface the recommender needs from a vector store."""

    def similarity_search(self, query: str, k: int) -> list[Document]:
        ...  # pragma: no cover - protocol method


class BookRecommender:
    """Produce book recommendations from a natural-language query."""

    def __init__(
        self,
        books: pd.DataFrame,
        vector_store: VectorStore,
        initial_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> None:
        self.books = books
        self.vector_store = vector_store
        self.initial_top_k = initial_top_k or settings.initial_top_k
        self.final_top_k = final_top_k or settings.final_top_k

    def recommend(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
    ) -> pd.DataFrame:
        """Return up to ``final_top_k`` recommended books as a DataFrame.

        Args:
            query: Free-text description of the kind of book wanted.
            category: A ``simple_categories`` value, or ``"All"`` for no filter.
            tone: One of the configured tones, or ``"All"`` to keep semantic order.

        Returns:
            A DataFrame of recommended books, ordered by relevance (and tone when
            requested). Empty if the query yields no hits.
        """
        if not query or not query.strip():
            return self.books.iloc[0:0].copy()

        hits = self.vector_store.similarity_search(query, k=self.initial_top_k)
        isbn_order = [doc.metadata["isbn13"] for doc in hits]
        if not isbn_order:
            return self.books.iloc[0:0].copy()

        filtered = self.books[self.books["isbn13"].isin(isbn_order)].copy()

        # Preserve the semantic ranking returned by the vector store.
        rank_map = {isbn: rank for rank, isbn in enumerate(isbn_order)}
        filtered["semantic_rank"] = filtered["isbn13"].map(rank_map)
        filtered = filtered.sort_values("semantic_rank")

        if category != "All" and "simple_categories" in filtered.columns:
            filtered = filtered[filtered["simple_categories"] == category]

        emotion_column = TONE_TO_EMOTION.get(tone)
        if emotion_column and emotion_column in filtered.columns:
            filtered = filtered.sort_values(by=emotion_column, ascending=False)

        return filtered.head(self.final_top_k)

    def categories(self) -> list[str]:
        """List of category choices for the UI, prefixed with ``"All"``."""
        if "simple_categories" not in self.books.columns:
            return ["All"]
        values = self.books["simple_categories"].dropna().unique().tolist()
        return ["All"] + sorted(values)
