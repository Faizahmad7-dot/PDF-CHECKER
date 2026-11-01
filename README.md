# 📚 PDF Chat

A Streamlit app that lets you chat with a PDF, summarise it, and quiz yourself on it,
powered by Google's Gemini Flash model.

## Features

- **Chat**: ask questions in plain English; answers cite page numbers like (p. 3)
- **Summary**: overview, key points and key terms
- **Quiz**: 3-10 multiple-choice questions with scoring and explanations
- Works with **PDF** and **TXT** files (up to 20 MB)

## How it works

1. The app extracts the text of each page with `pypdf`.
2. Small documents are sent to Gemini whole. For very large documents, the pages
   most relevant to your question are selected (chat), or every page is trimmed
   evenly (summary and quiz).
3. Gemini answers using only the document and cites pages. If the answer isn't
   in the document, it says so.
4. Busy or overloaded models are retried, then the app falls back to the next model.

> Scanned/image-only PDFs have no extractable text and won't work.

## Run locally

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

3. Provide the key by pasting it into the sidebar, **or** create
   `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
# Optional: force a specific model
# GEMINI_MODEL = "gemini-3.5-flash"
```

**Never commit `.streamlit/secrets.toml` or your key to GitHub.** Add it to `.gitignore`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`app.py` and `requirements.txt` in the root).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, paste `GEMINI_API_KEY = "your-key-here"`.
5. Click **Deploy**.

## Privacy

Document text is sent to the Gemini API. Don't upload anything confidential.
