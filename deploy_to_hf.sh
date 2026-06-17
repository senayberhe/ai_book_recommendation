#!/usr/bin/env bash
#
# Deploy the Semantic Book Recommender to a (free) Hugging Face Space.
#
# Prerequisites:
#   1. A Hugging Face account: https://huggingface.co/join
#   2. An access token with "write" role: https://huggingface.co/settings/tokens
#   3. git installed.
#
# Usage:
#   export HF_TOKEN=hf_xxx            # your write token (do NOT commit this)
#   export HF_USERNAME=your-username  # your HF username
#   ./deploy_to_hf.sh                 # optional arg: space name (default: semantic-book-recommender)
#
set -euo pipefail

SPACE_NAME="${1:-semantic-book-recommender}"
: "${HF_TOKEN:?Set HF_TOKEN to your Hugging Face write token}"
: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face username}"

echo "==> Installing huggingface_hub CLI (if needed)"
python -m pip install --quiet --upgrade "huggingface_hub[cli]"

echo "==> Creating Space ${HF_USERNAME}/${SPACE_NAME} (Gradio SDK)"
# --exist-ok makes re-runs safe.
huggingface-cli repo create "${SPACE_NAME}" \
  --repo-type space \
  --space_sdk gradio \
  --token "${HF_TOKEN}" \
  -y --exist-ok

SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
REMOTE="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo "==> Pointing 'space' git remote at ${SPACE_URL}"
git remote remove space 2>/dev/null || true
git remote add space "${REMOTE}"

echo "==> Pushing code to the Space"
git push --force space main

echo
echo "==> Setting the OPENAI_API_KEY secret"
if [ -n "${OPENAI_API_KEY:-}" ]; then
  huggingface-cli repo-files "${HF_USERNAME}/${SPACE_NAME}" --repo-type space >/dev/null 2>&1 || true
  python - <<PY
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
api.add_space_secret(repo_id="${HF_USERNAME}/${SPACE_NAME}", key="OPENAI_API_KEY", value="${OPENAI_API_KEY}")
print("OPENAI_API_KEY secret set on the Space.")
PY
else
  echo "OPENAI_API_KEY not set in this shell."
  echo "Add it manually: ${SPACE_URL} -> Settings -> Variables and secrets -> New secret"
  echo "   name: OPENAI_API_KEY   value: <your new key>"
fi

# Scrub the token from the remote URL so it isn't left in .git/config.
git remote set-url space "https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo
echo "Done. Your Space is building at:"
echo "   ${SPACE_URL}"
echo "First boot builds the Chroma index (a few minutes), then the demo is live."
