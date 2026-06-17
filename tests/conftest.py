"""Shared pytest fixtures.

These fixtures provide a tiny in-memory books dataset and a fake vector store so
the recommendation logic can be tested deterministically, with no network calls,
no OpenAI key, and no Chroma database.
"""

from __future__ import annotations

import pandas as pd
import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_books() -> pd.DataFrame:
    """A small, realistic books DataFrame covering the columns the app uses."""
    return pd.DataFrame(
        {
            "isbn13": ["111", "222", "333", "444"],
            "title": ["Hopeful Dawn", "Dark Storm", "Quiet River", "No Description"],
            "authors": ["Ada Lovelace", "Alan Turing;Grace Hopper", "A;B;C", "Anon"],
            "description": [
                "A warm story of forgiveness and second chances.",
                "A tense thriller full of fear and suspense.",
                "A calm meditation on nature and stillness.",
                "",  # should be dropped when building documents
            ],
            "simple_categories": ["Fiction", "Fiction", "Nonfiction", "Fiction"],
            "thumbnail": ["http://img/1", "http://img/2", None, "http://img/4"],
            "joy": [0.9, 0.1, 0.5, 0.2],
            "fear": [0.1, 0.95, 0.2, 0.3],
            "sadness": [0.2, 0.4, 0.3, 0.1],
            "anger": [0.05, 0.6, 0.1, 0.1],
            "surprise": [0.3, 0.2, 0.1, 0.4],
        }
    )


class FakeVectorStore:
    """A stand-in for Chroma that returns documents in a fixed, controllable order."""

    def __init__(self, isbn_order: list[str]) -> None:
        self._isbn_order = isbn_order

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return [
            Document(page_content="", metadata={"isbn13": isbn})
            for isbn in self._isbn_order[:k]
        ]


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    """Vector store that ranks 222 first, then 111, then 333."""
    return FakeVectorStore(isbn_order=["222", "111", "333"])
