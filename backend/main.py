from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os, io, re
from typing import Optional
from pathlib import Path
from database import db as history_db

load_dotenv()

app = FastAPI(title="Text Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("GROQ_API_KEY", "").strip() or "missing-set-GROQ_API_KEY-in-env"
OPENAI_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.groq.com/openai/v1")

# Multi-provider chat helper with Groq → Claude fallback on rate limits.
from llm_client import chat_with_fallback

class SummarizeRequest(BaseModel):
    text: str
    length: str = "2_paragraphs"
    grade_level: str = "grade8"
    language: str = "English"
    length_instruction: Optional[str] = None
    grade_instruction: Optional[str] = None

LENGTH_INSTRUCTIONS = {
    "1_paragraph":  "Summarize in exactly 1 paragraph.",
    "2_paragraphs": "Summarize in exactly 2 paragraphs.",
    "3_paragraphs": "Summarize in exactly 3 paragraphs.",
    "4_paragraphs": "Summarize in exactly 4 paragraphs.",
    "5_paragraphs": "Summarize in exactly 5 paragraphs.",
    "bullets":      "Summarize as a list of concise bullet points (use - for each point). Include 5–8 bullets.",
    "notes":        "Summarize in notes format: use short labeled section headings followed by brief bullet points under each, like structured study notes.",
    "short":        "Summarize in 2–3 sentences only. Be extremely concise.",
    "medium":       "Summarize in 1–2 paragraphs (5–8 sentences). Capture key ideas.",
    "long":         "Write a detailed summary in 3–4 paragraphs covering all main points.",
}

GRADE_INSTRUCTIONS = {
    "k":          "Write for a Kindergarten student. Use very simple words and very short sentences.",
    "grade1":     "Write for a Grade 1 student. Use simple words a 6-year-old would understand.",
    "grade2":     "Write for a Grade 2 student. Keep sentences short and vocabulary basic.",
    "grade3":     "Write for a Grade 3 student. Use simple but complete sentences.",
    "grade4":     "Write for a Grade 4 student. Use clear language appropriate for a 9-10 year old.",
    "grade5":     "Write for a Grade 5 student. Use straightforward language for a 10-11 year old.",
    "grade6":     "Write for a Grade 6 student. Use moderately simple language for an 11-12 year old.",
    "grade7":     "Write for a Grade 7 student. Use clear language for a 12-13 year old.",
    "grade8":     "Write for a Grade 8 student. Use accessible language for a 13-14 year old.",
    "grade9":     "Write for a Grade 9 student. Use standard academic language for a 14-15 year old.",
    "grade10":    "Write for a Grade 10 student. Use confident academic language for a 15-16 year old.",
    "grade11":    "Write for a Grade 11 student. Use mature academic language for a 16-17 year old.",
    "grade12":    "Write for a Grade 12 student. Use advanced language appropriate for a 17-18 year old.",
    "elementary": "Use very simple words and short sentences suitable for a 3rd-5th grade student.",
    "middle":     "Use clear, accessible language for a middle school student.",
    "high":       "Use standard academic language appropriate for a high school student.",
    "college":    "Use sophisticated vocabulary and complex sentence structures for a college student.",
    "general":    "Use clear, accessible language for a general adult audience.",
}

BLOCKED_KEYWORDS = [
    "pornography","pornographic","pornographer","pornographers",
    "porn ","porn,","porn.","porn\n","/porn","#porn",
    "sexually explicit","sex video","sex tape","nude video",
    "child sexual","child pornography","csam","child abuse material",
    "rape video","rape porn","incest","bestiality",
    "hentai","xxx ","xxx.","xxx\n",
    "onlyfans","camgirl","camboy","strip club","escort service",
    "sex worker","prostitution","brothel",
    "erectile dysfunction medication","penis enlargement",
    "drug trafficking","how to make meth","how to make bomb",
    "how to synthesize","drug manufacturing","crystal meth recipe",
    "kill yourself","suicide method","self harm method",
]

def _keyword_blocked(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in BLOCKED_KEYWORDS)

@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(req.text) > 15000:
        raise HTTPException(status_code=400, detail="Text too long. Max 15,000 characters.")

    # Fast keyword pre-filter — blocks before any AI call
    if _keyword_blocked(req.text):
        raise HTTPException(
            status_code=422,
            detail="INAPPROPRIATE_CONTENT: This content is not suitable for educational use. Please only summarize appropriate study materials such as textbooks, articles, research papers, or educational content."
        )

    length_note = req.length_instruction or LENGTH_INSTRUCTIONS.get(req.length, LENGTH_INSTRUCTIONS["2_paragraphs"])
    grade_note  = req.grade_instruction  or GRADE_INSTRUCTIONS.get(req.grade_level, GRADE_INSTRUCTIONS["grade8"])
    lang_note   = f"Write the entire summary in {req.language}." if req.language != "English" else ""

    system_prompt = (
        "You are a world-class text summarizer for an educational platform used by teachers and students.\n\n"
        "CONTENT SAFETY CHECK (do this first):\n"
        "If the text contains ANY of the following, respond with EXACTLY the word CONTENT_BLOCKED and nothing else:\n"
        "- Pornographic or sexually explicit material\n"
        "- Graphic violence, gore, or self-harm instructions\n"
        "- Hate speech or content targeting people by race, religion, gender, sexuality\n"
        "- Instructions for illegal activities (drug manufacturing, weapons, hacking for harm, etc.)\n"
        "- Adult content not suitable for an educational setting\n\n"
        "If the content is appropriate (educational, academic, news, science, history, literature, business, technology, etc.), "
        "summarize it following these rules:\n"
        "- Only include information present in the original text — never add outside knowledge\n"
        "- Never repeat the same idea twice, even in different words\n"
        "- Use active voice and specific, concrete language — avoid vague filler phrases\n"
        "- Start directly with the content — never begin with 'This text discusses...' or 'The article explains...'\n"
        "- Match the exact format requested (paragraphs / bullets / notes format)\n"
        "- For bullet format: start each bullet with a strong action verb or key fact\n"
        "- For notes format: use clear section headings in ALL CAPS followed by bullet points\n"
        "- Return ONLY the summary — no labels, no preamble, no sign-off, no meta-commentary"
    )

    user_prompt = (
        f"Summarize the text below.\n\n"
        f"FORMAT: {length_note}\n"
        f"READING LEVEL: {grade_note}\n"
        + (f"LANGUAGE: {lang_note}\n" if lang_note else "")
        + f"RULES: No repetition. No filler phrases. Be specific and precise.\n\n"
        f"TEXT:\n{req.text}"
    )

    try:
        response = chat_with_fallback(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        summary = response.choices[0].message.content.strip()

        # Check if AI flagged content as inappropriate
        if summary.upper().startswith("CONTENT_BLOCKED"):
            raise HTTPException(
                status_code=422,
                detail="INAPPROPRIATE_CONTENT: This content is not suitable for educational use. Please only summarize appropriate study materials such as textbooks, articles, research papers, or educational content."
            )

        return {
            "summary":      summary,
            "input_words":  len(req.text.split()),
            "output_words": len(summary.split()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")


@app.post("/api/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        import pdfplumber
        content = await file.read()
        text_parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages[:30]:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        text = "\n".join(text_parts).strip()
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. The file may be image-based.")
        return {"text": text[:15000], "pages": len(pdf.pages) if hasattr(pdf, 'pages') else 0}
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF processing library not available.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")


@app.post("/api/extract-url")
async def extract_url(payload: dict):
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TextSummarizer/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            tag.decompose()
        # Try article/main first, fallback to body
        main = soup.find("article") or soup.find("main") or soup.find("body")
        text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
        # Clean up whitespace
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
        text = "\n".join(lines)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract readable text from this URL.")
        return {"text": text[:15000], "title": soup.title.string.strip() if soup.title else url}
    except ImportError:
        raise HTTPException(status_code=500, detail="URL processing library not available.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL extraction failed: {str(e)}")


# ── Frontend serving ──
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

NO_CACHE_HEADERS = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}

# ─── HISTORY (SQLite, per-user, with date + session filters) ────────────────

class SaveSummaryRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    source_preview: str = ""
    summary: str
    length: str = ""
    grade_level: str = ""
    language: str = ""
    word_count: int = 0


class HistoryRequest(BaseModel):
    user_id: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    session_id: Optional[str] = None
    limit: int = 100


class SessionListRequest(BaseModel):
    user_id: str


@app.post("/api/save-summary")
def save_summary_endpoint(req: SaveSummaryRequest):
    try:
        sid = history_db.save_summary(
            req.user_id,
            req.source_preview,
            req.summary,
            length=req.length,
            grade_level=req.grade_level,
            language=req.language,
            word_count=req.word_count,
            session_id=req.session_id,
        )
        return {"success": True, "id": sid}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/summary-history")
def get_history_endpoint(req: HistoryRequest):
    try:
        items = history_db.get_history(
            req.user_id,
            date_from=req.date_from,
            date_to=req.date_to,
            session_id=req.session_id,
            limit=req.limit,
        )
        return {"success": True, "items": items, "count": len(items)}
    except Exception as e:
        return {"success": False, "error": str(e), "items": []}


@app.post("/api/summary-sessions")
def list_sessions_endpoint(req: SessionListRequest):
    try:
        sessions = history_db.list_sessions(req.user_id)
        return {"success": True, "sessions": sessions}
    except Exception as e:
        return {"success": False, "error": str(e), "sessions": []}


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html", headers=NO_CACHE_HEADERS)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
