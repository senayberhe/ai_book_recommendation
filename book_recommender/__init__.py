"""Semantic Book Recommender.

A small, well-structured package that powers a Gradio app for recommending
books from a natural-language description, with optional filtering by category
and emotional tone. Recommendations are produced by semantic search over book
descriptions (OpenAI embeddings + a Chroma vector store) followed by metadata
filtering and emotion-based re-ranking.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
