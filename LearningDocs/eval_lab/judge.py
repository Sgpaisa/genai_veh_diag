"""
"LLM-as-judge" -- grading free-text output that a simple == can't check.

`risk_level` is a hard label -> exact match, done in offline_eval.py directly.
`summary` / `root_causes` / `immediate_actions` are free text -> you need a
judge to score them against a rubric.

Two backends, same idea as model.py:
  - REAL: sends the output + rubric to Gemini and asks it to score 1-5.
  - MOCK: keyword-coverage heuristic (fraction of rubric criteria whose
          keywords show up in the text). Good enough to demonstrate the
          pattern without needing an API key.

In a real production system you'd use a real judge model (often a stronger
model than the one being evaluated), log its reasoning, and periodically
spot-check the judge itself against human graders -- judges drift too.
"""

import os
from dotenv import load_dotenv

load_dotenv()
USE_REAL = bool(os.getenv("GEMINI_API_KEY"))


def _mock_judge(output_text: str, rubric: list[str]) -> float:
    text = output_text.lower()
    hits = 0
    for criterion in rubric:
        # crude keyword-overlap check: does the output touch on the topic
        # the criterion names? (real judge would use semantic understanding)
        tokens = [w for w in criterion.lower().replace("or", " ").split() if len(w) > 3]
        if any(t in text for t in tokens):
            hits += 1
    coverage = hits / len(rubric) if rubric else 1.0
    return round(1 + coverage * 4, 2)  # map 0..1 coverage -> 1..5 score


def _real_judge(output_text: str, rubric: list[str]) -> float:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel
    from config import GEMINI_MODEL

    class JudgeScore(BaseModel):
        score: int  # 1-5
        reasoning: str

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"""You are grading an automotive diagnostic explanation against a rubric.

Explanation to grade:
\"\"\"{output_text}\"\"\"

Rubric (the explanation should satisfy each point):
{chr(10).join(f"- {c}" for c in rubric)}

Score 1-5: 5 = fully satisfies rubric, 1 = ignores it entirely. Be strict."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_schema=JudgeScore,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return float(response.parsed.score)


def judge_output(prediction: dict, rubric: list[str]) -> float:
    text = " ".join([
        prediction.get("summary", ""),
        " ".join(prediction.get("root_causes", [])),
        " ".join(prediction.get("immediate_actions", [])),
    ])
    if USE_REAL:
        return _real_judge(text, rubric)
    return _mock_judge(text, rubric)
