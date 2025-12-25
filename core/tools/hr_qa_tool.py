import core.config  # forces env load

from langchain_openai import ChatOpenAI
from core.vector_store import load_vector_store


def answer_hr_question(question: str) -> dict:
    """
    Answer HR-related questions using:
    1. Vector DB + LLM (preferred)
    2. LLM-only fallback (never silent fail)
    """

    if not question.strip():
        return {
            "answer": "Please enter a valid HR-related question.",
            "confidence": "Low",
            "source": "Validation",
            "reasoning": ["Empty input provided"]
        }

    # Initialize Vector DB and LLM
    vectordb = load_vector_store()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    # -----------------------------
    # 1. Try Vector DB Retrieval
    # -----------------------------
    docs = vectordb.similarity_search(question, k=4)

    if docs and len(docs) > 0:
        context = "\n\n".join(doc.page_content for doc in docs)

        answer = llm.invoke(f"""
You are an HR assistant.

Use the following internal HR policy context to answer the question.
If the context is partially relevant, still provide the best possible answer
without inventing facts.

HR Policy Context:
{context}

Question:
{question}
""").content

        return {
            "answer": answer,
            "confidence": "High",
            "source": "Vector DB + LLM",
            "reasoning": [
                "Retrieved relevant HR policy documents using vector similarity search",
                "Used LLM to generate a grounded response based on retrieved context"
            ]
        }

    # -----------------------------
    # 2. Controlled LLM Fallback
    # -----------------------------
    answer = llm.invoke(f"""
You are an HR assistant.

No internal HR policy documents were retrieved for this question.
Answer using general HR best practices.
Clearly state that the response is based on general knowledge.

Question:
{question}
""").content

    return {
        "answer": answer,
        "confidence": "Medium",
        "source": "LLM (No Retrieval)",
        "reasoning": [
            "No relevant documents retrieved from vector database",
            "Used LLM with general HR domain knowledge as a safe fallback"
        ]
    }
