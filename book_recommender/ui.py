"""Gradio user interface.

Builds the dashboard and binds it to a recommender. All pure formatting logic
lives in :mod:`book_recommender.formatting` so it can be tested without Gradio.
"""

from __future__ import annotations

import logging

import gradio as gr

from .config import TONE_TO_EMOTION
from .formatting import GalleryItem, recommendations_to_gallery
from .recommender import BookRecommender

logger = logging.getLogger(__name__)

TONES = ["All", *TONE_TO_EMOTION.keys()]


def build_dashboard(recommender: BookRecommender) -> gr.Blocks:
    """Construct the Gradio dashboard bound to a recommender instance."""

    def recommend_books(query: str, category: str, tone: str) -> list[GalleryItem]:
        try:
            recommendations = recommender.recommend(query=query, category=category, tone=tone)
        except Exception:  # noqa: BLE001 - surface a friendly UI state, log the rest
            logger.exception("Recommendation failed for query=%r", query)
            return []
        if recommendations.empty:
            return []
        return recommendations_to_gallery(recommendations)

    categories = recommender.categories()

    with gr.Blocks(theme=gr.themes.Glass(), title="Semantic Book Recommender") as dashboard:
        gr.Markdown("# 📚 Semantic Book Recommender")
        gr.Markdown("Describe the kind of book you're looking for and get matching reads.")

        with gr.Row():
            user_query = gr.Textbox(
                label="Book Description",
                placeholder="A story about forgiveness and redemption...",
                lines=2,
            )

        with gr.Row():
            category_dropdown = gr.Dropdown(choices=categories, value="All", label="Category")
            tone_dropdown = gr.Dropdown(choices=TONES, value="All", label="Emotional Tone")

        submit_button = gr.Button("Find Recommendations", variant="primary")

        output = gr.Gallery(
            label="Recommended Books",
            columns=4,
            rows=4,
            height="auto",
            object_fit="contain",
        )

        submit_button.click(
            fn=recommend_books,
            inputs=[user_query, category_dropdown, tone_dropdown],
            outputs=output,
        )

    return dashboard
