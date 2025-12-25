import core.config
import json
import re

from langchain_openai import ChatOpenAI
from core.vector_store import load_vector_store


def _clean_json(raw: str):
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())


def generate_interview_questions(job_description: str, role_level: str) -> dict:
    vectordb = load_vector_store()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    docs = vectordb.similarity_search(job_description, k=4)
    context = "\n".join(d.page_content for d in docs)

    base_prompt = f"""
You are a professional interviewer.

Role Level: {role_level}

RULES:
- Junior → fundamentals
- Mid → trade-offs & debugging
- Senior → architecture & scale

IGNORE context that conflicts with role.

Context:
{context}

Job Description:
{job_description}

Return ONLY valid JSON:
{{
  "technical": [string, string, string],
  "behavioral": [string, string]
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
                    "questions": [],
                    "reasoning": [
                        "LLM failed to produce valid JSON after retry",
                        "Generation aborted safely"
                    ]
                }
            base_prompt = "RETURN JSON ONLY. NO TEXT.\n" + base_prompt

    questions = []

    for q in data.get("technical", []):
        questions.append({
            "category": "Technical",
            "question": q,
            "intent": f"{role_level} level technical assessment"
        })

    for q in data.get("behavioral", []):
        questions.append({
            "category": "Behavioral",
            "question": q,
            "intent": f"{role_level} level behavioral assessment"
        })

    return {
        "questions": questions,
        "reasoning": [
            "Retrieved interview context from vector database",
            f"Generated role-specific questions for {role_level} using LLM"
        ]
    }
