#!/usr/bin/env bash
#
# Deploy the Semantic Book Recommender to a (free) Hugging Face Space.
#
# Prerequisites:
#   1. A Hugging Face account:            https://huggingface.co/join
#   2. An access token with "write" role: https://huggingface.co/settings/tokens
#   3. git and python installed.
#
# Usage (run from the project folder):
#   export HF_TOKEN=hf_xxx                 # your write token (never commit this)
#   export HF_USERNAME=your-hf-username    # your HF username
#   export OPENAI_API_KEY=sk-your-new-key  # optional: sets the Space secret for you
#   ./deploy_to_hf.sh                      # optional arg: space name (default below)
#
set -euo pipefail

SPACE_NAME="${1:-semantic-book-recommender}"
: "${HF_TOKEN:?Set HF_TOKEN to your Hugging Face write token}"
: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face username}"

REPO_ID="${HF_USERNAME}/${SPACE_NAME}"
SPACE_URL="https://huggingface.co/spaces/${REPO_ID}"

echo "==> Installing huggingface_hub (if needed)"
python -m pip install --quiet --upgrade "huggingface_hub>=0.23"

echo "==> Creating Space ${REPO_ID} (Gradio SDK, free CPU) if it doesn't exist"
HF_TOKEN="${HF_TOKEN}" REPO_ID="${REPO_ID}" python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id=os.environ["REPO_ID"],
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
)
print("Space ready.")
PY

echo "==> Pushing code to the Space (respects .gitignore)"
# Token is embedded only transiently; we scrub it from git config at the end.
git remote remove space 2>/dev/null || true
git remote add space "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${REPO_ID}"
git push --force space main
git remote set-url space "${SPACE_URL}"   # scrub token from .git/config

echo
echo "==> Setting the OPENAI_API_KEY secret on the Space"
if [ -n "${OPENAI_API_KEY:-}" ]; then
  HF_TOKEN="${HF_TOKEN}" REPO_ID="${REPO_ID}" OPENAI_API_KEY="${OPENAI_API_KEY}" python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.add_space_secret(
    repo_id=os.environ["REPO_ID"],
    key="OPENAI_API_KEY",
    value=os.environ["OPENAI_API_KEY"],
)
print("OPENAI_API_KEY secret set.")
PY
else
  echo "OPENAI_API_KEY not set in this shell — add it in the web UI:"
  echo "   ${SPACE_URL}  ->  Settings  ->  Variables and secrets  ->  New secret"
  echo "   name: OPENAI_API_KEY   value: <your new key>"
fi

echo
echo "Done. Your Space is building at:"
echo "   ${SPACE_URL}"
echo "First boot builds the Chroma index (a few minutes); then the demo is live."
