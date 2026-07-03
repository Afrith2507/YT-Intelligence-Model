# Team environment setup

The `.env` file is **not on GitHub on purpose**. It contains secret API keys. Committing it would expose your keys to the internet (GitHub bots scan for keys within minutes).

## What each teammate needs

1. Clone the repo
2. Copy the template:
   ```powershell
   copy .env.example .env
   ```
3. Ask **Afrith** for the shared team keys on WhatsApp / Discord (do not post keys in GitHub issues or commits)
4. Edit `.env` and paste:
   ```
   GROQ_API_KEY=...
   YOUTUBE_API_KEY=...
   ```
5. Run:
   ```powershell
   python install.py
   .\start.ps1
   ```

## Required keys

| Variable | Used for |
|----------|----------|
| `GROQ_API_KEY` | Ask tab, Summarize, evaluation (LLM answers) |
| `YOUTUBE_API_KEY` | Scraping new comments only (optional if CSVs already in repo) |

## Data files

CSV datasets are in `data/processed/` and `data/raw/` on GitHub. Teammates do **not** need to scrape again unless adding new data.

## Dashboard shortcut

You can also paste `GROQ_API_KEY` in the **sidebar** of the Streamlit app (Settings section) without editing `.env` — useful for quick demos.

## If keys stop working

Regenerate at:
- Groq: https://console.groq.com/keys
- YouTube: https://console.cloud.google.com/apis/credentials

Then share the new keys privately with the team and update your local `.env`.
