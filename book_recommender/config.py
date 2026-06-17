"""Central configuration for the Semantic Book Recommender.

All tunable values live here so the rest of the code never hard-codes paths or
magic numbers. Values can be overridden with environment variables, which makes
the app easy to configure on different machines and on Hugging Face Spaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project root = the directory that contains this package.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Emotional tones offered in the UI mapped to the emotion columns in the
# dataset. Each value is a column produced by the sentiment-analysis notebook.
TONE_TO_EMOTION: dict[str, str] = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings, resolved from environment variables with defaults."""

    books_csv: Path
    chroma_dir: Path
    embedding_model: str
    initial_top_k: int
    final_top_k: int
    cover_not_found: str

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        return cls(
            books_csv=Path(
                os.getenv("BOOKS_CSV", str(ROOT_DIR / "data" / "books_with_emotions.csv"))
            ),
            chroma_dir=Path(os.getenv("CHROMA_DIR", str(ROOT_DIR / "chroma_db"))),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            initial_top_k=int(os.getenv("INITIAL_TOP_K", "50")),
            final_top_k=int(os.getenv("FINAL_TOP_K", "16")),
            cover_not_found=os.getenv("COVER_NOT_FOUND", "cover-not-found.jpg"),
        )


# Default, importable settings instance.
settings = Settings.from_env()
