from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Optional
from pathlib import Path

load_dotenv()

app = FastAPI(title="Text Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SummarizeRequest(BaseModel):
    text: str
    length: str = "2_paragraphs"
    grade_level: str = "grade8"
    length_instruction: Optional[str] = None
    grade_instruction: Optional[str] = None

LENGTH_INSTRUCTIONS = {
    "1_paragraph":  "Summarize in exactly 1 paragraph.",
    "2_paragraphs": "Summarize in exactly 2 paragraphs.",
    "3_paragraphs": "Summarize in exactly 3 paragraphs.",
    "4_paragraphs": "Summarize in exactly 4 paragraphs.",
    "5_paragraphs": "Summarize in exactly 5 paragraphs.",
    "bullets":      "Summarize as a list of concise bullet points (use - for each point). Include 5–8 bullets.",
    "notes":        "Summarize in notes format: use short labeled section headings followed by brief bullet points under each, like structured study notes. Make it easy to review quickly.",
    "short":   "Summarize in 2–3 sentences only. Be extremely concise.",
    "medium":  "Summarize in 1–2 paragraphs (5–8 sentences). Capture key ideas.",
    "long":    "Write a detailed summary in 3–4 paragraphs covering all main points.",
}

GRADE_INSTRUCTIONS = {
    "k":       "Write for a Kindergarten student. Use very simple words and very short sentences.",
    "grade1":  "Write for a Grade 1 student. Use simple words a 6-year-old would understand.",
    "grade2":  "Write for a Grade 2 student. Keep sentences short and vocabulary basic.",
    "grade3":  "Write for a Grade 3 student. Use simple but complete sentences.",
    "grade4":  "Write for a Grade 4 student. Use clear language appropriate for a 9-10 year old.",
    "grade5":  "Write for a Grade 5 student. Use straightforward language for a 10-11 year old.",
    "grade6":  "Write for a Grade 6 student. Use moderately simple language for an 11-12 year old.",
    "grade7":  "Write for a Grade 7 student. Use clear language for a 12-13 year old.",
    "grade8":  "Write for a Grade 8 student. Use accessible language for a 13-14 year old.",
    "grade9":  "Write for a Grade 9 student. Use standard academic language for a 14-15 year old.",
    "grade10": "Write for a Grade 10 student. Use confident academic language for a 15-16 year old.",
    "grade11": "Write for a Grade 11 student. Use mature academic language for a 16-17 year old.",
    "grade12": "Write for a Grade 12 student. Use advanced language appropriate for a 17-18 year old.",
    "elementary": "Use very simple words and short sentences suitable for a 3rd-5th grade student.",
    "middle":     "Use clear, accessible language for a middle school student.",
    "high":       "Use standard academic language appropriate for a high school student.",
    "college":    "Use sophisticated vocabulary and complex sentence structures for a college student.",
    "general":    "Use clear, accessible language for a general adult audience.",
}

@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(req.text) > 15000:
        raise HTTPException(status_code=400, detail="Text too long. Max 15,000 characters.")

    length_note = req.length_instruction or LENGTH_INSTRUCTIONS.get(req.length, LENGTH_INSTRUCTIONS["2_paragraphs"])
    grade_note  = req.grade_instruction  or GRADE_INSTRUCTIONS.get(req.grade_level, GRADE_INSTRUCTIONS["grade8"])

    system_prompt = (
        "You are an expert text summarizer. Your job is to read the provided text and create an accurate, "
        "clear, and well-structured summary. Do NOT add your own opinions or information not present in the text. "
        "Only return the summary — no preamble, no meta-commentary, no labels."
    )

    user_prompt = (
        f"Please summarize the following text.\n\n"
        f"Length instruction: {length_note}\n"
        f"Reading level: {grade_note}\n\n"
        f"TEXT TO SUMMARIZE:\n{req.text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        summary = response.choices[0].message.content.strip()
        return {
            "summary":      summary,
            "input_words":  len(req.text.split()),
            "output_words": len(summary.split()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# ── Frontend serving ──
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
