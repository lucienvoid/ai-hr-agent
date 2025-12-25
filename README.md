🧠 AI-Powered HR Agent

An AI-powered HR system that automates resume screening, interview question generation, interview evaluation, and HR Q&A, while remaining explainable, cost-controlled, and demo-safe.

This project was built with a deterministic-first design, using AI only where it adds real value.

🚀 Features
1️⃣ Resume Screening

Matches resume skills against job requirements

Uses deterministic logic for fairness

AI is used only to explain decisions, not to decide them

2️⃣ Interview Question Generator

Generates role-specific interview questions

Supports Junior / Mid / Senior roles

Uses Vector DB + LLM to stay grounded in job context

Prevents role collapse (no junior questions for senior roles)

3️⃣ Interview Evaluation

Evaluates candidate answers relative to role expectations

Same answer → different score for different roles

Robust JSON-safe LLM handling

Produces:

Score

Verdict (Pass / Borderline / Fail)

Strengths, weaknesses, reasoning

4️⃣ HR Q&A Chatbot

Answers employee HR questions using internal policy documents

Grounded via Vector DB (no hallucinations)

Falls back to general HR knowledge when needed

Provides confidence level and source attribution

🧩 Tech Stack
Layer	Technology
Frontend	Streamlit
Backend	Python
AI	OpenAI (GPT models)
Agent Routing	LangChain-style agent logic
Vector Database	Chroma
Embeddings	OpenAI embeddings
Environment	Python 3.12
🏗 Architecture Overview
Streamlit UI
     ↓
Agent Router (Intent-based)
     ↓
Feature Tool
     ↓
Vector DB (Chroma) + LLM (OpenAI)
     ↓
Structured Output + Reasoning

Design Principles

Deterministic first (AI never blindly decides)

Explainability everywhere

Role-aware evaluation

Cost-controlled LLM usage

Fail-safe JSON parsing

📁 Project Structure
Operation Black Widow/
│
├── main.py                 # Streamlit UI (all features)
├── build_vectors.py        # Builds vector database
│
├── core/
│   ├── agent.py            # Central agent router
│   ├── vector_store.py     # Chroma vector DB logic
│   ├── config.py           # Environment loading
│   │
│   └── tools/
│       ├── resume_tool.py
│       ├── interview_generator.py
│       ├── interview_evaluator.py
│       └── hr_qa_tool.py
│
├── data/
│   └── hr_policies.txt     # HR knowledge base
│
├── requirements.txt
├── .env                    # OpenAI API key (not committed)
└── README.md

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone <repo-url>
cd Operation-Black-Widow

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Environment Variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here

5️⃣ Build Vector Database
python build_vectors.py

6️⃣ Run the App
streamlit run main.py

🧪 Sample Test Case (Interview Evaluation)

Job Description

Backend Python Developer working on REST APIs, authentication,
databases, and performance optimization.


Question

How would you design a REST API in Python for a production system?


Candidate Answer

I would use FastAPI, define endpoints, use JWT authentication,
connect to a database with an ORM, and write unit tests.


Expected Output

Junior → PASS (~80)

Mid → BORDERLINE (~65)

Senior → FAIL (~50)

💰 Cost Control Strategy

LLM calls are scoped and intentional

Deterministic logic used where possible

No continuous chat loops

Vector retrieval limits token usage

Safe for demo and hackathon budgets

🛡 Safety & Reliability

Robust JSON parsing with retries

Safe fallbacks on model failure

No hard crashes during demo

Explainable outputs for every decision

🎯 Problem Statement Addressed

AI-Powered HR Agent: Create an AI to handle resumes, interviews, and employee questions automatically

✔ Resume handling
✔ Interview automation
✔ Employee HR Q&A
✔ Explainable & role-aware decisions

🧠 Why This Project Stands Out

Not prompt-only

No hallucinated decisions

Real HR logic

Role-calibrated evaluation

Production-minded architecture

📌 Note

This project is built for demonstration, evaluation, and educational purposes.
It showcases how AI can assist HR without replacing human judgment.