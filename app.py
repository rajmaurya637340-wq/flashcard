from fastapi import FastAPI
from fastapi import HTTPException, Query
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
load_dotenv()

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    token=HF_TOKEN,
    provider="featherless-ai"
)


@app.get("/")
def home():
    return {"message": "Flashcard Generator API is running"}


@app.post("/generate-flashcards")
def generate_flashcards(
    topic: str = Query(..., min_length=1, max_length=200),
    count: int = Query(5, ge=1, le=20),
):
    if not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Flashcard generation is unavailable because HF_TOKEN is not configured.",
        )

    response = client.chat.completions.create(
        model="google/gemma-2-2b-it",
        messages=[
            {
                "role": "user",
                "content": f"""
Act as a knowledgeable, patient study tutor. Create exactly {count} useful flashcards about:
{topic}

Make each question test one important idea. Give a clear, accurate answer in plain language,
with a brief explanation or example when useful. Avoid duplicate questions and filler.

Return only a valid JSON array in this exact format:
[
    {{
        "question": "Question here",
        "answer": "Answer here"
    }}
]
"""
            }
        ],
        max_tokens=min(2000, count * 100),
    )

    content = response.choices[0].message.content
    try:
        flashcards = json.loads(content or "")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="The AI provider returned flashcards in an invalid format. Please try again.",
        ) from exc

    if (
        not isinstance(flashcards, list)
        or len(flashcards) != count
        or any(
            not isinstance(card, dict)
            or not isinstance(card.get("question"), str)
            or not card["question"].strip()
            or not isinstance(card.get("answer"), str)
            or not card["answer"].strip()
            for card in flashcards
        )
    ):
        raise HTTPException(
            status_code=502,
            detail="The AI provider returned incomplete flashcards. Please try again.",
        )

    return {
        "topic": topic,
        "count": len(flashcards),
        "flashcards": flashcards,
    }