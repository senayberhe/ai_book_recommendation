"""Tests for the pure formatting helpers (no Gradio server involved)."""

from __future__ import annotations

import pandas as pd

from book_recommender.formatting import (
    build_caption,
    format_authors,
    recommendations_to_gallery,
    truncate_description,
)


def test_format_authors_single() -> None:
    assert format_authors("Ada Lovelace") == "Ada Lovelace"


def test_format_authors_two() -> None:
    assert format_authors("Alan Turing;Grace Hopper") == "Alan Turing and Grace Hopper"


def test_format_authors_many() -> None:
    assert format_authors("A;B;C") == "A, B, and C"


def test_format_authors_empty_is_unknown() -> None:
    assert format_authors("") == "Unknown Author"


def test_truncate_description_truncates_long_text() -> None:
    text = " ".join(str(i) for i in range(40))
    out = truncate_description(text, max_words=30)
    assert out.endswith("...")
    # 30 words kept; the ellipsis attaches to the last word, so 30 tokens total.
    assert len(out.split()) == 30
    assert out.startswith("0 1 2")


def test_truncate_description_keeps_short_text() -> None:
    assert truncate_description("short and sweet") == "short and sweet"


def test_build_caption_contains_title_and_author() -> None:
    row = pd.Series(
        {"title": "Hopeful Dawn", "authors": "Ada Lovelace", "description": "A warm story."}
    )
    caption = build_caption(row)
    assert "Hopeful Dawn" in caption
    assert "by Ada Lovelace" in caption


def test_recommendations_to_gallery_shape() -> None:
    df = pd.DataFrame(
        {
            "large_thumbnail": ["img1", "img2"],
            "title": ["A", "B"],
            "authors": ["X", "Y"],
            "description": ["d1", "d2"],
        }
    )
    items = recommendations_to_gallery(df)
    assert len(items) == 2
    assert items[0][0] == "img1"
    assert "A" in items[0][1]
