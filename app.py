"""Application entrypoint for the Semantic Book Recommender.

Wires together the data, vector store, recommender, and Gradio UI, then launches
the dashboard. This is the file Hugging Face Spaces runs.
"""

from __future__ import annotations

import logging
import os

# Force the pure-Python protobuf implementation to avoid native/runtime
# incompatibilities with some Chroma builds. Must be set before heavy imports.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from dotenv import load_dotenv  # noqa: E402

from book_recommender.data import load_books  # noqa: E402
from book_recommender.recommender import BookRecommender  # noqa: E402
from book_recommender.ui import build_dashboard  # noqa: E402
from book_recommender.vector_store import get_vector_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("book_recommender")


def create_app():
    """Build the Gradio dashboard with a fully initialised recommender."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set it as a secret on Hugging Face Spaces."
        )

    logger.info("Loading books dataset...")
    books = load_books()

    logger.info("Initialising vector store (this can take a while on first run)...")
    vector_store = get_vector_store(books)

    recommender = BookRecommender(books=books, vector_store=vector_store)
    return build_dashboard(recommender)


def main() -> None:
    """Launch the dashboard."""
    create_app().launch()


if __name__ == "__main__":
    main()
