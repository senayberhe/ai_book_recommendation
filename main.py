"""Command-line entrypoint.

A thin wrapper so the app can be started with ``python main.py`` in addition to
``python app.py``. Hugging Face Spaces uses ``app.py``; this exists for local
convenience and for the ``book-recommender`` console script defined in
pyproject.toml.
"""

from __future__ import annotations

from app import main

if __name__ == "__main__":
    main()
