"""
HR Chatbot Tool
Answers employee questions using FAISS vector search over company documents
"""

import logging
import json
from typing import Dict, Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class HRChatTool:
    """Tool for HR chatbot queries"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.last_reasoning = ""
        self._load_prompt()
        self._initialize_vectorstore()

    def _load_prompt(self):
        """Load HR chat prompt"""
        self.prompt_template = """You are an HR Assistant. Answer employee questions based on company documents.

Company Information:
{company_context}

Employee Question: {query}

Respond in JSON format:
{{
    "answer": "<direct answer to the question>",
    "confidence": <0.0-1.0>,
    "sources": ["<source1>", "<source2>"],
    "additional_info": "<any helpful additional information>",
    "requires_human_review": <true/false>
}}

IMPORTANT RULES:
1. If the answer is NOT found in company documents, clearly state: "This information is not available in our knowledge base."
2. Do NOT hallucinate or make up information.
3. If unsure, set confidence < 0.5 and flag for human review.
4. Always cite sources when available."""

    def _initialize_vectorstore(self):
        """Initialize FAISS vector store"""
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings

            # Check if index exists
            index_path = Path("./vectorstore/faiss_index")
            if index_path.exists():
                logger.info("Loading existing FAISS index...")
                embeddings = OpenAIEmbeddings()
                self.vectorstore = FAISS.load_local(
                    str(index_path),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully")
            else:
                logger.warning("No existing FAISS index found. Will use direct LLM responses.")
                self.vectorstore = None
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {str(e)}. Will use direct responses.")
            self.vectorstore = None

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from vector store"""
        if not self.vectorstore:
            return "No company documents available."

        try:
            # Search for relevant documents
            docs = self.vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context if context else "No relevant documents found."
        except Exception as e:
            logger.warning(f"Vector search failed: {str(e)}")
            return "Search failed. Please try again."

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Answer HR-related employee questions

        Args:
            query: Employee question

        Returns:
            Dictionary with answer and metadata
        """
        logger.info("HR Chatbot Tool - START")

        try:
            # Retrieve context from vector store
            context = self._retrieve_context(query)

            # Create prompt
            prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["company_context", "query"]
            )

            # Format prompt
            formatted_prompt = prompt.format(
                company_context=context,
                query=query
            )

            logger.info(f"Processing HR query: {query[:100]}...")

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
                    "answer": response_text,
                    "confidence": 0.5,
                    "requires_human_review": True
                }

            confidence = result.get("confidence", 0)
            self.last_reasoning = f"Query processed. Confidence: {confidence}. Requires review: {result.get('requires_human_review', False)}"

            logger.info(f"HR Chatbot Tool - END")
            return {
                "success": True,
                "result": result
            }

        except Exception as e:
            logger.error(f"HR chat failed: {str(e)}")
            self.last_reasoning = f"Error during chat: {str(e)}"
            return {
                "success": False,
                "error": str(e)
            }
