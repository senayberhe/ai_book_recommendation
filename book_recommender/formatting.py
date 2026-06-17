"""Pure presentation helpers.

These functions format recommender output for display. They have no Gradio or
model dependencies, so they are cheap to import and easy to unit-test.
"""

from __future__ import annotations

import pandas as pd

GalleryItem = tuple[str, str]


def format_authors(raw_authors: str) -> str:
    """Format a semicolon-separated author string for display.

    ``"A"`` -> ``"A"``; ``"A;B"`` -> ``"A and B"``; ``"A;B;C"`` -> ``"A, B, and C"``.
    """
    authors = [a.strip() for a in str(raw_authors).split(";") if a.strip()]
    if not authors:
        return "Unknown Author"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{', '.join(authors[:-1])}, and {authors[-1]}"


def truncate_description(description: str, max_words: int = 30) -> str:
    """Truncate a description to ``max_words`` words, adding an ellipsis."""
    words = str(description).split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return str(description)


def build_caption(row: pd.Series) -> str:
    """Build the gallery caption (title, authors, truncated description)."""
    title = str(row.get("title", "Untitled"))
    authors = format_authors(row.get("authors", ""))
    description = truncate_description(row.get("description", ""))
    return f"{title}\nby {authors}\n\n{description}"


def recommendations_to_gallery(recommendations: pd.DataFrame) -> list[GalleryItem]:
    """Turn a recommendations DataFrame into ``(image, caption)`` gallery items."""
    items: list[GalleryItem] = []
    for _, row in recommendations.iterrows():
        items.append((row["large_thumbnail"], build_caption(row)))
    return items
