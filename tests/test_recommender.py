"""Tests for the recommendation engine, using a fake vector store."""

from __future__ import annotations

import pandas as pd
import pytest

from book_recommender.data import prepare_books
from book_recommender.recommender import BookRecommender


def _make_recommender(sample_books: pd.DataFrame, store) -> BookRecommender:
    return BookRecommender(books=prepare_books(sample_books), vector_store=store)


def test_recommend_preserves_semantic_order(sample_books, fake_vector_store) -> None:
    rec = _make_recommender(sample_books, fake_vector_store)
    result = rec.recommend("anything")
    # Fake store ranks 222 first, then 111, then 333.
    assert result["isbn13"].tolist() == ["222", "111", "333"]


def test_recommend_filters_by_category(sample_books, fake_vector_store) -> None:
    rec = _make_recommender(sample_books, fake_vector_store)
    result = rec.recommend("anything", category="Nonfiction")
    assert result["isbn13"].tolist() == ["333"]


def test_recommend_reranks_by_tone(sample_books, fake_vector_store) -> None:
    rec = _make_recommender(sample_books, fake_vector_store)
    # "Happy" -> joy column; 111 has the highest joy and should come first.
    result = rec.recommend("anything", tone="Happy")
    assert result.iloc[0]["isbn13"] == "111"

    # "Suspenseful" -> fear column; 222 has the highest fear.
    result = rec.recommend("anything", tone="Suspenseful")
    assert result.iloc[0]["isbn13"] == "222"


def test_tone_blends_with_semantic_rank_instead_of_replacing_it(
    sample_books, fake_vector_store
) -> None:
    rec = _make_recommender(sample_books, fake_vector_store)
    result = rec.recommend("anything", tone="Happy")
    # 222 is the best semantic match (joy 0.1); 333 is the worst (joy 0.5). A pure
    # joy sort would put 333 above 222; blending keeps the closer match ahead.
    order = result["isbn13"].tolist()
    assert order == ["111", "222", "333"]


def test_tone_weight_extremes(sample_books, fake_vector_store) -> None:
    books = prepare_books(sample_books)
    semantic_only = BookRecommender(books, fake_vector_store, tone_weight=0.0)
    assert semantic_only.recommend("q", tone="Happy")["isbn13"].tolist() == ["222", "111", "333"]

    tone_only = BookRecommender(books, fake_vector_store, tone_weight=1.0)
    assert tone_only.recommend("q", tone="Happy")["isbn13"].tolist() == ["111", "333", "222"]


def test_invalid_tone_weight_rejected(sample_books, fake_vector_store) -> None:
    with pytest.raises(ValueError):
        BookRecommender(prepare_books(sample_books), fake_vector_store, tone_weight=1.5)


def test_recommend_respects_final_top_k(sample_books, fake_vector_store) -> None:
    rec = BookRecommender(
        books=prepare_books(sample_books),
        vector_store=fake_vector_store,
        final_top_k=1,
    )
    assert len(rec.recommend("anything")) == 1


def test_recommend_empty_query_returns_empty(sample_books, fake_vector_store) -> None:
    rec = _make_recommender(sample_books, fake_vector_store)
    assert rec.recommend("   ").empty


def test_categories_are_sorted_with_all_first(sample_books, fake_vector_store) -> None:
    rec = _make_recommender(sample_books, fake_vector_store)
    assert rec.categories() == ["All", "Fiction", "Nonfiction"]
