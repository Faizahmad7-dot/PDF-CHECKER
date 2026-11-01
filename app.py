"""PDF Chat - Streamlit + Gemini Flash.

Upload a PDF (lecture notes, paper, book chapter) and:
  * chat with it, with answers that cite page numbers
  * get a summary with key points
  * generate a multiple-choice quiz to test yourself
"""

import io
import json
import os
import re
import time

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]
MAX_FILE_MB = 20
MAX_CONTEXT_CHARS = 150_000  # text sent to the model per request
RETRIES_PER_MODEL = 3
RETRY_DELAY_SECONDS = 2
MODEL_MISSING_MARKERS = ("404", "not found", "not supported")
TRANSIENT_MARKERS = (
    "503", "unavailable", "high demand", "overloaded", "500", "internal",
    "429", "resource_exhausted", "deadline", "timeout", "timed out",
)
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "what", "which", "who", "how", "why", "when", "where", "does", "do", "did",
    "this", "that", "with", "about", "from", "by", "it", "its", "be", "as", "at", "me",
    "tell", "explain", "give", "can", "you", "i", "my", "your",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def extract_pages(uploaded_file) -> list:
    """Return a list of page texts from a PDF or TXT upload."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
        # Split long text files into pseudo-pages of ~3000 chars.
        return [text[i : i + 3000] for i in range(0, len(text), 3000)] or [""]

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password-protected.")
        return [(page.extract_text() or "").strip() for page in reader.pages]

    raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")


def keywords(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def build_context(pages: list, query: str = "") -> str:
    """Build the document text sent to the model, labelled by page number.

    Small documents are sent whole. Large ones are reduced: pages most relevant
    to the question are kept (chat), or every page is trimmed evenly (summary/quiz).
    """
    labelled = [(i + 1, t) for i, t in enumerate(pages) if t]
    total = sum(len(t) for _, t in labelled)

    if total <= MAX_CONTEXT_CHARS:
        chosen = labelled
    elif query and keywords(query):
        q = keywords(query)
        scored = sorted(
            labelled,
            key=lambda p: len(q & keywords(p[1])),
            reverse=True,
        )
        chosen, used = [], 0
        for num, text in scored:
            if used + len(text) > MAX_CONTEXT_CHARS:
                continue
            chosen.append((num, text))
            used += len(text)
        chosen.sort(key=lambda p: p[0])  # restore page order
    else:
        per_page = max(300, MAX_CONTEXT_CHARS // max(1, len(labelled)))
        chosen = [(n, t[:per_page]) for n, t in labelled]

    return "\n\n".join(f"[Page {n}]\n{t}" for n, t in chosen)


def parse_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for open_c, close_c in (("[", "]"), ("{", "}")):
            s, e = text.find(open_c), text.rfind(close_c)
            if s != -1 and e > s:
                try:
                    return json.loads(text[s : e + 1])
                except json.JSONDecodeError:
                    continue
        raise


def call_gemini(prompt: str, api_key: str, json_mode: bool = False) -> str:
    """Call Gemini with retries on busy errors and fallback across models."""
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    custom = get_secret("GEMINI_MODEL")
    models = ([custom] if custom else []) + [m for m in DEFAULT_MODELS if m != custom]

    last_error = None
    for model in models:
        for attempt in range(RETRIES_PER_MODEL):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                text = response.text
                if not text:
                    raise ValueError("empty response")
                return text
            except Exception as exc:
                message = str(exc).lower()
                if any(k in message for k in MODEL_MISSING_MARKERS):
                    last_error = exc
                    break  # next model
                if isinstance(exc, ValueError) or any(k in message for k in TRANSIENT_MARKERS):
                    last_error = exc
                else:
                    raise  # e.g. invalid API key
            if attempt < RETRIES_PER_MODEL - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

    raise RuntimeError(
        "Gemini is busy right now and all models failed. Please wait a minute and "
        f"try again. Last error: {last_error}"
    )


def answer_question(pages: list, question: str, history: list, api_key: str) -> str:
    context = build_context(pages, question)
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-6:]
    )
    prompt = f"""You are a helpful study assistant. Answer the user's question using ONLY
the document below. Cite page numbers like (p. 3) after the facts you use.
If the answer is not in the document, say so clearly instead of guessing.
Keep answers clear and concise; use short bullet points where helpful.

DOCUMENT:
\"\"\"
{context}
\"\"\"

CONVERSATION SO FAR:
{convo or "(none)"}

User: {question}
Assistant:"""
    return call_gemini(prompt, api_key)


def summarise(pages: list, api_key: str) -> str:
    context = build_context(pages)
    prompt = f"""Summarise the document below for a student.
Format in Markdown with these parts:
## Overview  (3-4 sentences)
## Key points  (6-10 bullets, cite pages like (p. 3))
## Key terms  (up to 8 terms with one-line definitions)

DOCUMENT:
\"\"\"
{context}
\"\"\""""
    return call_gemini(prompt, api_key)


def generate_quiz(pages: list, n: int, api_key: str) -> list:
    context = build_context(pages)
    prompt = f"""Create {n} multiple-choice questions that test understanding of the document below.
Return ONLY a JSON array. Each item must be:
{{"question": "...", "options": ["A text", "B text", "C text", "D text"],
  "answer_index": <0-3>, "explanation": "<why, with page reference>"}}
Exactly 4 options per question, only one correct, plausible wrong options.

DOCUMENT:
\"\"\"
{context}
\"\"\""""
    data = parse_json(call_gemini(prompt, api_key, json_mode=True))
    if isinstance(data, dict):
        data = data.get("questions", [])

    quiz = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        options = item.get("options")
        try:
            idx = int(item.get("answer_index"))
        except (TypeError, ValueError):
            continue
        if (
            isinstance(options, list)
            and len(options) == 4
            and 0 <= idx < 4
            and str(item.get("question", "")).strip()
        ):
            quiz.append(
                {
                    "question": str(item["question"]).strip(),
                    "options": [str(o).strip() for o in options],
                    "answer_index": idx,
                    "explanation": str(item.get("explanation", "")).strip(),
                }
            )
    if not quiz:
        raise ValueError("The model returned no usable questions. Please try again.")
    return quiz


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def reset_state():
    for key in ("messages", "summary", "quiz", "quiz_submitted"):
        st.session_state.pop(key, None)


def main():
    st.set_page_config(page_title="PDF Chat", page_icon="📚", layout="wide")
    st.title("📚 PDF Chat")
    st.caption("Upload a document, then chat with it, summarise it, or quiz yourself.")

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        with st.sidebar:
            st.header("Settings")
            api_key = st.text_input(
                "Gemini API key",
                type="password",
                help="Get a free key at https://aistudio.google.com/apikey",
            )

    uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded is None:
        reset_state()
        st.session_state.pop("doc_id", None)
        st.info("Upload a file to get started.")
        return

    if uploaded.size > MAX_FILE_MB * 1024 * 1024:
        st.error(f"File is too large. Maximum size is {MAX_FILE_MB} MB.")
        return

    doc_id = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.get("doc_id") != doc_id:
        try:
            with st.spinner("Reading your document..."):
                pages = extract_pages(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return
        if sum(len(p) for p in pages) < 100:
            st.error(
                "Almost no text could be extracted. The file may be a scanned image. "
                "Try a text-based PDF."
            )
            return
        reset_state()
        st.session_state.update(doc_id=doc_id, pages=pages)

    pages = st.session_state["pages"]
    st.success(f"Loaded **{uploaded.name}**: {len(pages)} pages, "
               f"{sum(len(p) for p in pages):,} characters.")

    if not api_key:
        st.warning("Add your Gemini API key in the sidebar to continue.")
        return

    tab_chat, tab_summary, tab_quiz = st.tabs(["💬 Chat", "📝 Summary", "🧠 Quiz"])

    # ---- Chat ---------------------------------------------------------------
    with tab_chat:
        messages = st.session_state.setdefault("messages", [])
        for m in messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        question = st.chat_input("Ask something about the document...")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        reply = answer_question(pages, question, messages, api_key)
                    st.markdown(reply)
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": reply})
            except Exception as exc:
                st.error(f"Could not get an answer: {exc}")

    # ---- Summary ------------------------------------------------------------
    with tab_summary:
        if st.button("Generate summary"):
            try:
                with st.spinner("Summarising..."):
                    st.session_state["summary"] = summarise(pages, api_key)
            except Exception as exc:
                st.error(f"Could not summarise: {exc}")
        if st.session_state.get("summary"):
            st.markdown(st.session_state["summary"])

    # ---- Quiz ---------------------------------------------------------------
    with tab_quiz:
        n = st.slider("Number of questions", 3, 10, 5)
        if st.button("Generate quiz"):
            try:
                with st.spinner("Writing questions..."):
                    st.session_state["quiz"] = generate_quiz(pages, n, api_key)
                    st.session_state["quiz_submitted"] = False
                    for k in [k for k in st.session_state if k.startswith("q_")]:
                        del st.session_state[k]
            except Exception as exc:
                st.error(f"Could not create the quiz: {exc}")

        quiz = st.session_state.get("quiz")
        if quiz:
            for i, q in enumerate(quiz):
                st.markdown(f"**{i + 1}. {q['question']}**")
                st.radio(
                    "Choose one",
                    range(4),
                    index=None,
                    format_func=lambda j, q=q: q["options"][j],
                    key=f"q_{i}",
                    label_visibility="collapsed",
                )
            if st.button("Check answers"):
                st.session_state["quiz_submitted"] = True

            if st.session_state.get("quiz_submitted"):
                score = 0
                for i, q in enumerate(quiz):
                    picked = st.session_state.get(f"q_{i}")
                    correct = picked == q["answer_index"]
                    score += correct
                    right = q["options"][q["answer_index"]]
                    if correct:
                        st.success(f"Q{i + 1}: Correct. {q['explanation']}")
                    else:
                        st.error(f"Q{i + 1}: Answer is **{right}**. {q['explanation']}")
                st.metric("Score", f"{score} / {len(quiz)}")

    st.caption("Answers are AI-generated from your document. Double-check important facts.")


if __name__ == "__main__":
    main()
