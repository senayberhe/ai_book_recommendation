# Semantic Book Recommender — container image.
#
# Build:  docker build -t book-recommender .
# Run:    docker run --rm -p 7860:7860 --env-file .env -v chroma_data:/data book-recommender
# (or just `docker compose up` — see docker-compose.yml)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Gradio binds to localhost by default, which is unreachable from outside a container.
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    # Persist the vector index on a volume so restarts don't re-embed (and re-pay).
    CHROMA_DIR=/data/chroma_db

WORKDIR /app

# Run as an unprivileged user; /data is owned by it so a named volume inherits access.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/chroma_db \
    && chown -R app:app /data /app

# Install dependencies first so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app app.py main.py ./
COPY --chown=app:app book_recommender ./book_recommender
COPY --chown=app:app data/books_with_emotions.csv ./data/books_with_emotions.csv

USER app
VOLUME ["/data"]
EXPOSE 7860

# First boot builds the Chroma index (minutes), hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860', timeout=4)"

CMD ["python", "app.py"]
