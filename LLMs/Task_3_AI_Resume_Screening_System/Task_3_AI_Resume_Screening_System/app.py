"""
app.py — Streamlit Web Interface for AI Resume Screening System
Provides a user-friendly UI to run the pipeline interactively.

Run with:
    streamlit run app.py
"""

import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load project modules
from utils.file_loader import load_all_resumes, load_job_description
from utils.langsmith_tracer import setup_langsmith_tracing
from chains.extraction_chain import run_extraction
from chains.matching_chain import run_matching
from chains.scoring_chain import run_scoring
from chains.explanation_chain import run_explanation

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Load Environment
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🤖 AI Resume Screening System")
st.markdown("**Task 3 | Innomatics Research Labs | Agentic AI Internship**")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    
    
    enable_tracing = st.checkbox("Enable LangSmith Tracing", value=True)
    
    candidate_choice = st.selectbox(
        "Select Candidate to Screen",
        ["strong", "average", "weak", "all"]
    )
    
    st.markdown("---")
    st.markdown("**Pipeline Steps:**")
    st.markdown("1. 📋 Skill Extraction")
    st.markdown("2. 🔍 Matching Analysis")
    st.markdown("3. 📊 Scoring (0–100)")
    st.markdown("4. 📝 Explanation Report")

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Job Description")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    try:
        jd_text = load_job_description(data_dir)
        st.text_area("", value=jd_text, height=300, disabled=True)
    except Exception as e:
        st.error(f"Could not load job description: {e}")
        jd_text = ""

with col2:
    st.subheader("👤 Candidate Resumes")
    try:
        resumes = load_all_resumes(data_dir)
        tab_labels = list(resumes.keys())
        tabs = st.tabs([t.capitalize() for t in tab_labels])
        for tab, label in zip(tabs, tab_labels):
            with tab:
                st.text_area("", value=resumes[label], height=250, disabled=True)
    except Exception as e:
        st.error(f"Could not load resumes: {e}")
        resumes = {}

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Run Pipeline Button
# ─────────────────────────────────────────────────────────────────────────────
run_button = st.button("🚀 Run Screening Pipeline", type="primary", use_container_width=True)

if run_button:
    if not openai_key:
        st.error("❌ Please provide your OpenAI API Key in the sidebar.")
        st.stop()
    
    if not resumes or not jd_text:
        st.error("❌ Could not load required files. Check the data/ directory.")
        st.stop()
    
    # Setup LangSmith if enabled
    if enable_tracing:
        if langsmith_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
            try:
                setup_langsmith_tracing()
                st.success("✅ LangSmith tracing enabled")
            except Exception as e:
                st.warning(f"⚠️ LangSmith setup warning: {e}")
        else:
            st.warning("⚠️ LangSmith API Key not provided — tracing disabled")
    
    # Initialize LLM
    llm = ChatOpenAI(
        api_key=openai_key,
        model="gpt-3.5-turbo",
        temperature=0
    )
    
    # Determine which candidates to screen
    candidates_to_run = (
        list(resumes.items()) if candidate_choice == "all"
        else [(candidate_choice, resumes[candidate_choice])]
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Run Pipeline for Each Candidate
    # ─────────────────────────────────────────────────────────────────────
    for label, resume_text in candidates_to_run:
        
        st.markdown(f"## 👤 {label.capitalize()} Candidate")
        
        progress = st.progress(0, text="Starting pipeline...")
        
        with st.expander(f"📋 Step 1: Skill Extraction — {label.capitalize()}", expanded=True):
            with st.spinner("Extracting skills..."):
                extracted = run_extraction(resume_text, llm)
            progress.progress(25, text="Extraction complete")
            st.json(extracted)
        
        with st.expander(f"🔍 Step 2: Match Analysis — {label.capitalize()}", expanded=True):
            with st.spinner("Matching against JD..."):
                match_result = run_matching(extracted, jd_text, llm)
            progress.progress(50, text="Matching complete")
            st.json(match_result)
        
        with st.expander(f"📊 Step 3: Score — {label.capitalize()}", expanded=True):
            with st.spinner("Calculating score..."):
                score_result = run_scoring(match_result, jd_text, llm)
            progress.progress(75, text="Scoring complete")
            
            # Display score prominently
            score = score_result.get("total_score", 0)
            category = score_result.get("score_category", "N/A")
            
            score_col1, score_col2 = st.columns(2)
            with score_col1:
                st.metric("Final Score", f"{score}/100")
            with score_col2:
                st.metric("Category", category)
            
            st.json(score_result)
        
        with st.expander(f"📝 Step 4: Explanation — {label.capitalize()}", expanded=True):
            with st.spinner("Generating explanation..."):
                explanation = run_explanation(extracted, match_result, score_result, llm)
            progress.progress(100, text="Pipeline complete ✓")
            st.markdown(explanation)
        
        st.markdown("---")
    
    st.success("✅ Pipeline completed for all selected candidates!")
    
    if enable_tracing and langsmith_key:
        st.info("📊 View LangSmith traces at: https://smith.langchain.com")
