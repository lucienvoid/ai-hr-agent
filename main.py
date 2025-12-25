import sys
import os
import streamlit as st

# -------------------------------------------------
# Ensure project root is on Python path
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.agent import run_agent, Intent

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(
    page_title="AI HR Agent",
    layout="wide"
)

st.title("AI-Powered HR Agent")
st.caption("Deterministic core • Vector DB • LLM-powered reasoning")

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------
page = st.sidebar.radio(
    "Select Feature",
    [
        "Resume Screening",
        "Interview Question Generator",
        "Interview Evaluation",
        "HR Q&A Chatbot"
    ]
)

# =================================================
# Feature 1: Resume Screening
# =================================================
if page == "Resume Screening":
    st.header("📄 Resume Screening")

    resume_text = st.text_area(
        "Paste Resume Text",
        height=200
    )

    job_description = st.text_area(
        "Paste Job Description",
        height=200
    )

    if st.button("Analyze Resume"):
        result = run_agent(
            intent=Intent.RESUME_SCREENING,
            payload={
                "resume_text": resume_text,
                "job_description": job_description
            }
        )

        if "error" in result:
            st.error(result["error"])
        else:
            st.metric("Skill Match Percentage", f"{result['match_percentage']}%")
            st.write("**AI Recommendation:**", result["recommendation"])

            st.subheader("Explanation")
            st.write(result["explanation"])

            st.subheader("Reasoning Trace")
            for r in result["reasoning"]:
                st.write("•", r)

# =================================================
# Feature 2: Interview Question Generator
# =================================================
elif page == "Interview Question Generator":
    st.header("🧠 Interview Question Generator")

    job_description = st.text_area(
        "Paste Job Description",
        height=200
    )

    role_level = st.selectbox(
        "Role Level",
        ["Junior", "Mid", "Senior"]
    )

    if st.button("Generate Questions"):
        result = run_agent(
            intent=Intent.INTERVIEW_GENERATION,
            payload={
                "job_description": job_description,
                "role_level": role_level   # ✅ FIXED KEY
            }
        )

        if "error" in result:
            st.error(result["error"])
        else:
            if not result["questions"]:
                st.warning("No questions generated.")
            else:
                st.subheader("Generated Interview Questions")
                for q in result["questions"]:
                    st.markdown(f"**[{q['category']}]** {q['question']}")
                    st.caption(f"Purpose: {q['intent']}")

            st.subheader("Reasoning")
            for r in result["reasoning"]:
                st.write("•", r)

# =================================================
# Feature 3: Interview Evaluation
# =================================================
elif page == "Interview Evaluation":
    st.header("⭐ Interview Evaluation")

    job_description = st.text_area(
        "Paste Job Description",
        height=150
    )

    role_level = st.selectbox(
        "Role Level",
        ["Junior", "Mid", "Senior"]
    )

    question = st.text_area(
        "Interview Question",
        height=100
    )

    answer = st.text_area(
        "Candidate Answer",
        height=200
    )

    if st.button("Evaluate Interview"):
        result = run_agent(
            intent=Intent.INTERVIEW_EVALUATION,
            payload={
                "question": question,
                "answer": answer,              # ✅ FIXED
                "job_description": job_description,
                "role_level": role_level       # ✅ FIXED
            }
        )

        if "error" in result:
            st.error(result["error"])
        else:
            st.metric("Overall Score", f"{result['overall_score']}/100")
            st.write("**Verdict:**", result["verdict"])

            st.subheader("Strengths")
            if result["strengths"]:
                for s in result["strengths"]:
                    st.write("•", s)
            else:
                st.write("None")

            st.subheader("Weaknesses")
            if result["weaknesses"]:
                for w in result["weaknesses"]:
                    st.write("•", w)
            else:
                st.write("None")

            st.subheader("Reasoning")
            for r in result["reasoning"]:
                st.write("•", r)

# =================================================
# Feature 4: HR Q&A Chatbot
# =================================================
elif page == "HR Q&A Chatbot":
    st.header("💬 HR Q&A Assistant")

    question = st.text_input(
        "Ask an HR-related question",
        placeholder="How does resume screening work?"
    )

    if st.button("Ask"):
        result = run_agent(
            intent=Intent.HR_QA,
            payload={"question": question}
        )

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Answer")
            st.write(result["answer"])

            st.metric("Confidence", result["confidence"])
            st.write("**Source:**", result["source"])

            st.subheader("Reasoning")
            for r in result["reasoning"]:
                st.write("•", r)
