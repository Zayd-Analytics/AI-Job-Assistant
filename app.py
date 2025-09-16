import os
import json
import streamlit as st
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from docx import Document
from PyPDF2 import PdfReader
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import openai
from openai import OpenAI

st.set_page_config(page_title="AI Job Search Assistant", page_icon="🧠")


# Load environment variables
load_dotenv()
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=DEFAULT_API_KEY)

# --- Prompt builder ---
def make_prompt(resume_text, job_description=None, user_question=None, mode="analyze"):
    base = f"You are an expert career assistant. Mode: {mode}.\nPlease keep your response as concise as possible."
    if mode == "analyze":
        base += (
            "Analyze the resume and return ONLY valid JSON (no markdown, no explanation, no code block) with keys: assessment, strengths, weaknesses, rewritten_bullets, "
            "job_titles, freelance_ideas, keywords, rewritten_resume, bullet_resume, email_summary.\n"
        )
    elif mode == "cover_letter":
        base += (
            "Generate a tailored cover letter based on the resume and job description. Use a professional tone.\n"
        )
    elif mode == "linkedin":
        base += (
            "Suggest LinkedIn headline, summary, and experience section updates based on the resume.\n"
        )
    elif mode == "interview":
        base += (
            "Generate 5 personalized interview questions and suggested answers based on the resume and job description.\n"
        )
    if job_description:
        base += f"Job Description:\n{job_description}\n"
    if user_question:
        base += f"User Question:\n{user_question}\n"
    base += f"Resume Text:\n{resume_text}"
    return base

# --- Resume scoring ---
def resume_score(text):
    score = {
        'Contact': 10 if any(word in text.lower() for word in ['email', 'phone', 'contact']) else 0,
        'Skills': 10 if 'skills' in text.lower() else 0,
        'Experience': 10 if 'experience' in text.lower() else 0,
    }
    score['Total'] = sum(score.values())
    return score

# --- Extract text from file ---
def extract_text(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif uploaded_file.name.lower().endswith(".docx"):
            doc = Document(uploaded_file)
            return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return ""

# --- Sidebar: API Key ---
st.sidebar.header("API Settings")
api_key_input = st.sidebar.text_input("Open AI API Key", type="password", value=DEFAULT_API_KEY)
# (API key input is handled by openai.api_key assignment above)

# --- Session state for versioning ---
if "versions" not in st.session_state:
    st.session_state.versions = {}

# --- Main UI ---
st.title("AI Job Search Assistant")

uploaded = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description (Optional)", height=150)
user_question = st.text_input("Ask a question about your resume (Optional)")

if uploaded:
    text = extract_text(uploaded)
    if not text.strip():
        st.error("Could not extract text. Try a different file or paste below.")
    else:
        st.subheader("Extracted Resume Text")
        with st.expander("Show Raw Text"):
            st.text_area("Resume Text", value=text, height=300)

        # Scorecard
        scores = resume_score(text)
        st.subheader("Scorecard")
        cols = st.columns(4)
        cols[0].metric("Total", f"{scores['Total']}/100")
        cols[1].metric("Contact", f"{scores['Contact']}/20")
        cols[2].metric("Skills", f"{scores['Skills']}/20")
        cols[3].metric("Experience", f"{scores['Experience']}/25")

        st.markdown("**Breakdown**")
        df = pd.DataFrame([{'Category': k, 'Score': v} for k, v in scores.items() if k != 'Total'])
        st.table(df)

        # --- Resume Analysis ---
        if st.button("Analyze & Rewrite Resume"):
            with st.spinner("Analyzing resume..."):
                prompt = make_prompt(text, job_description, user_question, mode="analyze")
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=512
                    )
                    result = json.loads(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"OpenAI API call failed: {e}")
                    result = {}

            if result:
                st.subheader("Assessment")
                st.write(result.get("assessment", ""))

                st.subheader("Strengths")
                for s in result.get("strengths", []):
                    st.write(f"- {s}")

                st.subheader("Weaknesses")
                for w in result.get("weaknesses", []):
                    st.write(f"- {w}")

                st.subheader("Rewritten Bullets")
                for b in result.get("rewritten_bullets", []):
                    st.write(f"- {b}")

                st.subheader("Job Titles")
                st.write(", ".join(result.get("job_titles", [])))

                st.subheader("Freelance Ideas")
                for f in result.get("freelance_ideas", []):
                    st.write(f"- {f}")

                st.subheader("ATS Keywords")
                st.write(", ".join(result.get("keywords", [])))

                st.subheader("Email Summary")
                st.text_area("Professional Summary", value=result.get("email_summary", ""), height=100)

                bullet_resume = result.get("bullet_resume", "")
                if bullet_resume:
                    st.subheader("Professional Resume (Bullets)")
                    st.text_area("Bullet Resume", value=bullet_resume, height=400)

                rewritten_resume = result.get("rewritten_resume", "")
                if rewritten_resume:
                    st.subheader("Rewritten Resume (Full)")
                    st.text_area("Full Resume", value=rewritten_resume, height=400)

                    version_name = st.text_input("Label this resume version", value="Version 1")
                    if st.button("Save Resume Version"):
                        st.session_state.versions[version_name] = rewritten_resume
                        st.success(f"Saved as '{version_name}'")

        # --- Cover Letter Generator ---
        if st.button("Generate Cover Letter"):
            with st.spinner("Creating cover letter..."):
                prompt = make_prompt(text, job_description, mode="cover_letter")
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512
                    )
                    st.subheader("Cover Letter")
                    st.text_area("Cover Letter", value=response.choices[0].message.content, height=400)
                except Exception as e:
                    st.error(f"OpenAI API call failed: {e}")

        # --- LinkedIn Optimizer ---
        if st.button("Optimize LinkedIn Profile"):
            with st.spinner("Optimizing LinkedIn profile..."):
                prompt = make_prompt(text, job_description, mode="linkedin")
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512
                    )
                    st.subheader("LinkedIn Profile Suggestions")
                    st.text_area("LinkedIn Suggestions", value=response.choices[0].message.content, height=400)
                except Exception as e:
                    st.error(f"OpenAI API call failed: {e}")

        # --- Interview Q&A Generator ---
        if st.button("Generate Interview Q&A"):
            with st.spinner("Preparing interview questions..."):
                prompt = make_prompt(text, job_description, mode="interview")
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512
                    )
                    st.subheader("Interview Questions & Answers")
                    st.text_area("Interview Prep", value=response.choices[0].message.content, height=400)
                except Exception as e:
                    st.error(f"OpenAI API call failed: {e}")

        # --- Resume Version Viewer ---
        if st.session_state.versions:
            st.subheader("Saved Resume Versions")
            version_keys = list(st.session_state.versions.keys())
            selected = st.selectbox("Select a version to view", version_keys)
            st.text_area("Saved Version", value=st.session_state.versions[selected], height=300)

else:
    st.info("Upload a resume file to get started, or paste demo text below.")
    demo = st.text_area("Demo Resume Text", height=250)
    if demo and st.button("Analyze Demo Text"):
        scores = resume_score(demo)
        st.subheader("Scorecard")
        st.metric("Total", f"{scores['Total']}/100")
        st.table(pd.DataFrame([{"Category": k, "Score": v} for k, v in scores.items() if k != 'Total']))

  # --- Chat-style Q&A ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.subheader("Ask Me Anything About Your Resume 🗨️")

        user_input = st.chat_input("Ask me anything, bro...")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Thinking like your career wingman..."):
                    prompt = make_prompt(demo, job_description, user_input, mode="analyze")
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=512
                        )
                        reply = response.choices[0].message.content
                    except Exception as e:
                        reply = f"Oops, something went wrong: {e}"
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # Show full chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
if st.button("Clear Chat"):
    st.session_state.chat_history = []
