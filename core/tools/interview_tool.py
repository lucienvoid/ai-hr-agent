"""
Interview Question Generator Tool
Generates role-based interview questions with difficulty levels
"""

import logging
import json
from typing import Dict, Any, List
from pathlib import Path
import core.config
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class InterviewTool:
    """Tool for generating interview questions"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.last_reasoning = ""
        self._load_prompt()

    def _load_prompt(self):
        """Load interview generation prompt"""
        prompt_path = Path(__file__).parent.parent / "prompts" / "interview_prompt.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                self.prompt_template = f.read()
        else:
            # Fallback prompt
            self.prompt_template = """You are an expert interviewer. Generate interview questions for the specified role.

Job Title: {job_title}
Difficulty Level: {difficulty_level}
Number of Questions: 5

Generate questions that assess:
1. Technical skills
2. Problem-solving ability
3. Communication skills
4. Team collaboration
5. Experience and background

Return the response as JSON in this format:
{{
    "questions": [
        {{
            "id": 1,
            "question": "<question text>",
            "difficulty": "{difficulty_level}",
            "category": "<Technical/Behavioral/Situational>",
            "expected_qualities": ["<quality1>", "<quality2>"]
        }},
        ...
    ],
    "interview_focus": "<brief description of what this interview should assess>"
}}

Ensure questions are:
- Specific to the role
- Open-ended to encourage detailed answers
- Fair and objective
- Professional"""

    def execute(self, job_title: str, difficulty_level: str = "medium") -> Dict[str, Any]:
        """
        Generate interview questions for a role

        Args:
            job_title: Title of the position
            difficulty_level: Level of difficulty (easy, medium, hard)

        Returns:
            Dictionary with generated questions
        """
        logger.info("Interview Generator Tool - START")

        # Validate difficulty level
        valid_levels = ["easy", "medium", "hard"]
        if difficulty_level.lower() not in valid_levels:
            difficulty_level = "medium"
            logger.warning(f"Invalid difficulty level, defaulting to medium")

        try:
            # Create prompt
            prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["job_title", "difficulty_level"]
            )

            # Format prompt
            formatted_prompt = prompt.format(
                job_title=job_title,
                difficulty_level=difficulty_level
            )

            logger.info(f"Generating questions for role: {job_title}")

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
                    "questions": []
                }

            self.last_reasoning = f"Generated questions for role: {job_title} at {difficulty_level} difficulty"

            logger.info(f"Interview Generator Tool - END")
            return {
                "success": True,
                "result": result
            }

        except Exception as e:
            logger.error(f"Interview generation failed: {str(e)}")
            self.last_reasoning = f"Error during question generation: {str(e)}"
            return {
                "success": False,
                "error": str(e)
            }
