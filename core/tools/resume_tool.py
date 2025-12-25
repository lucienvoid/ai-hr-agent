import core.config  # noqa

from langchain_openai import ChatOpenAI
from core.vector_store import load_vector_store

USE_LLM = True

def run_resume_screening(resume_text: str, job_description: str) -> dict:
    if not resume_text or not job_description:
        return {"error": "Resume and Job Description required."}

    # -----------------------------
    # Deterministic skill logic
    # -----------------------------
    jd_skills = {s.strip().lower() for s in job_description.split(",")}
    resume_skills = {s.strip().lower() for s in resume_text.split(",")}

    matched = jd_skills & resume_skills
    match_percentage = round((len(matched) / max(len(jd_skills), 1)) * 100, 2)

    recommendation = (
        "Shortlist" if match_percentage >= 75
        else "Hold" if match_percentage >= 50
        else "Reject"
    )

    # -----------------------------
    # Vector DB context
    # -----------------------------
    vectordb = load_vector_store()
    docs = vectordb.similarity_search(job_description, k=2)
    context = "\n".join([d.page_content for d in docs])

    # -----------------------------
    # LLM explanation
    # -----------------------------
    if USE_LLM:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        explanation = llm.invoke(f"""
Explain the resume screening decision.

Context:
{context}

Job Description:
{job_description}

Resume:
{resume_text}

Match Percentage: {match_percentage}
Recommendation: {recommendation}
""").content
    else:
        explanation = "Resume matched against job skills using deterministic logic."

    return {
        "match_percentage": match_percentage,
        "recommendation": recommendation,
        "explanation": explanation,
        "reasoning": [
            "Extracted skills from job description",
            "Extracted skills from resume",
            "Calculated overlap",
            "Retrieved HR evaluation context from vector DB",
            "Used LLM for explanation"
        ]
    }
