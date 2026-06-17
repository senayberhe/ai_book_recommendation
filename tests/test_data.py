"""Tests for data loading and document preparation."""

from __future__ import annotations

import pandas as pd

from book_recommender.data import create_documents, prepare_books


def test_prepare_books_builds_large_thumbnail(sample_books: pd.DataFrame) -> None:
    prepared = prepare_books(sample_books, cover_not_found="missing.jpg")

    # Present thumbnails get the high-res suffix.
    assert prepared.loc[0, "large_thumbnail"] == "http://img/1&fife=w800"
    # Missing thumbnails fall back to the placeholder.
    assert prepared.loc[2, "large_thumbnail"] == "missing.jpg"


def test_prepare_books_coerces_isbn_to_string(sample_books: pd.DataFrame) -> None:
    numeric = sample_books.copy()
    numeric["isbn13"] = [111, 222, 333, 444]
    prepared = prepare_books(numeric)
    assert prepared["isbn13"].map(type).eq(str).all()


def test_create_documents_skips_empty_descriptions(sample_books: pd.DataFrame) -> None:
    docs = create_documents(sample_books)
    # The 4th row has an empty description and must be dropped.
    assert len(docs) == 3
    assert all(doc.page_content for doc in docs)


def test_create_documents_carries_metadata(sample_books: pd.DataFrame) -> None:
    docs = create_documents(sample_books)
    first = docs[0]
    assert first.metadata["isbn13"] == "111"
    assert first.metadata["title"] == "Hopeful Dawn"
    assert first.metadata["category"] == "Fiction"
