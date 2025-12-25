"""
Interview Evaluation Tool
Evaluates candidate responses to interview questions
"""

import logging
import json
from typing import Dict, Any
from pathlib import Path
import core.config  # noqa

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class EvaluationTool:
    """Tool for evaluating interview answers"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.last_reasoning = ""
        self._load_prompt()

    def _load_prompt(self):
        """Load evaluation prompt"""
        prompt_path = Path(__file__).parent.parent / "prompts" / "evaluation_prompt.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                self.prompt_template = f.read()
        else:
            # Fallback prompt
            self.prompt_template = """You are an expert interviewer. Evaluate the candidate's answer to the interview question.

Job Title: {job_title}
Question: {question}
Candidate's Answer: {candidate_answer}

Evaluate based on:
1. Technical correctness
2. Completeness of answer
3. Communication clarity
4. Problem-solving approach
5. Relevance to the role

Provide evaluation in JSON format:
{{
    "score": <0-10>,
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "feedback": "<constructive feedback>",
    "improvement_suggestions": ["<suggestion1>", "<suggestion2>"],
    "reasoning": "<why this score was given>",
    "pass_fail": "<Pass / Fail / Borderline>"
}}

Be fair but objective. Provide detailed reasoning for the score."""

    def execute(self, question: str, candidate_answer: str, job_title: str = "") -> Dict[str, Any]:
        """
        Evaluate a candidate's interview answer

        Args:
            question: The interview question
            candidate_answer: The candidate's response
            job_title: Optional job title for context

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Interview Evaluation Tool - START")

        try:
            # Create prompt
            prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["job_title", "question", "candidate_answer"]
            )

            # Format prompt
            formatted_prompt = prompt.format(
                job_title=job_title or "General Position",
                question=question,
                candidate_answer=candidate_answer
            )

            logger.info(f"Evaluating answer for job: {job_title or 'General'}")

            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content

            # Parse JSON response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse JSON response")
                result = {
                    "raw_response": response_text,
                    "score": 5,
                    "pass_fail": "Borderline"
                }

            score = result.get("score", 5)
            self.last_reasoning = f"Answer evaluated with score {score}/10. Pass/Fail: {result.get('pass_fail', 'Unknown')}"

            logger.info(f"Interview Evaluation Tool - END")
            return {
                "success": True,
                "result": result
            }

        except Exception as e:
            logger.error(f"Answer evaluation failed: {str(e)}")
            self.last_reasoning = f"Error during evaluation: {str(e)}"
            return {
                "success": False,
                "error": str(e)
            }
