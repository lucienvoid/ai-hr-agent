import core.config
import json
import re

from langchain_openai import ChatOpenAI
from core.vector_store import load_vector_store

ROLE_ADJUSTMENT = {
    "Junior": 0,
    "Mid": -15,
    "Senior": -30
}



def _clean_json(raw: str):
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())


def evaluate_interview(question, answer, job_description, role_level):
    if not answer.strip():
        return {
            "overall_score": 0,
            "verdict": "Fail",
            "strengths": [],
            "weaknesses": ["No answer provided"],
            "reasoning": ["Empty answer detected"]
        }

    vectordb = load_vector_store()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    docs = vectordb.similarity_search(job_description, k=4)
    context = "\n".join(d.page_content for d in docs)

    base_prompt = f"""
You are an HR interviewer.

Evaluate the answer as if it was given by a JUNIOR-level candidate.

Scoring Rules:
- 70–85 → Good Junior answer
- 50–69 → Weak Junior answer
- <50 → Poor Junior answer

Do NOT apply senior expectations.

Context:
{context}

Interview Question:
{question}

Candidate Answer:
{answer}

Return ONLY valid JSON:
{{
  "base_score": number,
  "strengths": [string],
  "weaknesses": [string],
  "reasoning": string
}}
"""


    for attempt in range(2):
        raw = llm.invoke(base_prompt).content
        try:
            data = _clean_json(raw)
            break
        except Exception:
            if attempt == 1:
                return {
                    "overall_score": 0,
                    "verdict": "Fail",
                    "strengths": [],
                    "weaknesses": ["Invalid model output"],
                    "reasoning": ["JSON parsing failed safely"]
                }
            base_prompt = "RETURN JSON ONLY. NO TEXT.\n" + base_prompt

    adjusted = max(
        0,
        min(100, data["base_score"] + ROLE_ADJUSTMENT.get(role_level, 0))
    )

    verdict = "Pass" if adjusted >= 70 else "Borderline" if adjusted >= 50 else "Fail"

    return {
        "overall_score": adjusted,
        "verdict": verdict,
        "strengths": data["strengths"],
        "weaknesses": data["weaknesses"],
        "reasoning": [
            "Evaluated absolute answer quality using LLM",
            f"Adjusted score for {role_level} role expectations",
            data["reasoning"]
        ]
    }
