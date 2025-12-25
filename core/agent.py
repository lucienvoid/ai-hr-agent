from enum import Enum

from core.tools.resume_tool import run_resume_screening
from core.tools.interview_generator import generate_interview_questions
from core.tools.interview_evaluator import evaluate_interview
from core.tools.hr_qa_tool import answer_hr_question


class Intent(Enum):
    RESUME_SCREENING = "resume_screening"
    INTERVIEW_GENERATION = "interview_generation"
    INTERVIEW_EVALUATION = "interview_evaluation"
    HR_QA = "hr_qa"


def run_agent(intent: Intent, payload: dict) -> dict:
    try:
        if intent == Intent.RESUME_SCREENING:
            return run_resume_screening(
                resume_text=payload.get("resume_text", ""),
                job_description=payload.get("job_description", "")
            )

        if intent == Intent.INTERVIEW_GENERATION:
            return generate_interview_questions(
                job_description=payload.get("job_description", ""),
                role_level=payload.get("role_level", "Junior")
            )

        if intent == Intent.INTERVIEW_EVALUATION:
            return evaluate_interview(
                question=payload.get("question", ""),
                answer=payload.get("answer", ""),  # ✅ FIXED
                job_description=payload.get("job_description", ""),
                role_level=payload.get("role_level", "Junior")
            )

        if intent == Intent.HR_QA:
            return answer_hr_question(
                question=payload.get("question", "")
            )

        return {"error": "Unknown intent"}

    except Exception as e:
        return {"error": f"Agent execution error: {str(e)}"}
