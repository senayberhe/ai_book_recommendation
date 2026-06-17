# Deploying to Hugging Face Spaces (free)

This app is ready to run as a free Gradio Space. There are two ways to deploy.
Either way, **rotate your OpenAI key first** and only ever store it as a Space
secret — never in the code or git history.

## Option A — One command (recommended)

1. Create a Hugging Face account: https://huggingface.co/join
2. Create a **write** access token: https://huggingface.co/settings/tokens
3. In a terminal, from the project folder:

   ```bash
   export HF_TOKEN=hf_your_write_token
   export HF_USERNAME=your-hf-username
   export OPENAI_API_KEY=sk-your-new-key      # optional; sets the Space secret
   ./deploy_to_hf.sh                          # or: ./deploy_to_hf.sh my-space-name
   ```

The script creates the Space, pushes the code, sets the `OPENAI_API_KEY`
secret, and prints the live URL. It scrubs the token from your git config when
it finishes.

> The token is read from an environment variable so it never lands in a file or
> in git. Don't paste it into any committed file.

## Option B — Web UI (no terminal)

1. Go to https://huggingface.co/new-space
2. **Owner**: you · **Space name**: `semantic-book-recommender`
3. **SDK**: Gradio · **Hardware**: CPU basic (free) · **Visibility**: Public
4. Click **Create Space**.
5. In the new Space: **Files** → **Add file** → upload the repo contents, or
   connect this GitHub repo. The `README.md` YAML header configures the Space
   automatically (it sets `app_file: app.py` and the Gradio SDK version).
6. **Settings → Variables and secrets → New secret**:
   - Name: `OPENAI_API_KEY`
   - Value: your new OpenAI key
7. The Space builds from `requirements.txt` and launches `app.py`.

## First boot

On first launch the app embeds the ~5,000 book descriptions and persists the
Chroma index, which takes a few minutes and uses the OpenAI embeddings API once.
After that, only short user queries are embedded, so it's fast and cheap.

## Notes

- The Space build uses `requirements.txt` (lean runtime deps), not the full
  `pyproject.toml` dependency set.
- Set a spend limit in the OpenAI dashboard for peace of mind.
- If the build fails on the Gradio version, edit `sdk_version` in the README
  YAML header to a version Hugging Face currently supports.
