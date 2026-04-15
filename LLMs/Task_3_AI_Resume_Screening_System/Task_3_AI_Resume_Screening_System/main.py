"""
main.py — CLI Entry Point for AI Resume Screening System

This script runs the complete 4-step resume screening pipeline for:
  - Strong candidate
  - Average candidate
  - Weak candidate

Pipeline Flow:
  Resume → Extract → Match → Score → Explain → Trace (LangSmith)

Usage:
    python main.py

Environment Variables Required (.env file):
    OPENAI_API_KEY=<your_openai_key>
    LANGSMITH_API_KEY=<your_langsmith_key>
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

# ── Project Utilities ─────────────────────────────────────────────────────────
from utils.file_loader import load_all_resumes, load_job_description
from utils.report_formatter import format_pipeline_result, save_report, print_summary_table
from utils.langsmith_tracer import setup_langsmith_tracing, get_run_tags, get_run_metadata

# ── Pipeline Chain Modules ────────────────────────────────────────────────────
from chains.extraction_chain import run_extraction
from chains.matching_chain import run_matching
from chains.scoring_chain import run_scoring
from chains.explanation_chain import run_explanation


# ─────────────────────────────────────────────────────────────────────────────
# Load environment variables from .env file
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# @traceable decorator for LangSmith tracing — marks each candidate run
# ─────────────────────────────────────────────────────────────────────────────
@traceable(
    name="resume_screening_pipeline",
    run_type="chain",
)
def run_pipeline_for_candidate(
    candidate_label: str,
    resume_text: str,
    job_description: str,
    llm: ChatOpenAI
) -> dict:
    """
    Execute the complete 4-step screening pipeline for one candidate.
    
    This function is decorated with @traceable so every sub-step is
    visible in LangSmith as a nested run.
    
    Args:
        candidate_label: 'strong', 'average', or 'weak'
        resume_text: Raw resume text
        job_description: Raw job description text
        llm: Initialized ChatOpenAI instance
        
    Returns:
        Dict with all pipeline outputs
    """
    print(f"\n{'─'*60}")
    print(f"  Processing: {candidate_label.upper()} CANDIDATE")
    print(f"{'─'*60}")
    
    # ── STEP 1: Skill Extraction ──────────────────────────────────────────
    print("  [1/4] Extracting skills, tools, experience...")
    extracted_profile = run_extraction(resume_text, llm)
    candidate_name = extracted_profile.get("name", "Unknown")
    print(f"       → Candidate: {candidate_name}")
    print(f"       → Skills found: {len(extracted_profile.get('skills', []))}")
    
    # ── STEP 2: Matching ─────────────────────────────────────────────────
    print("  [2/4] Matching against job description...")
    match_analysis = run_matching(extracted_profile, job_description, llm)
    matched = len(match_analysis.get("matched_skills", []))
    missing = len(match_analysis.get("missing_skills", []))
    print(f"       → Matched skills: {matched} | Missing skills: {missing}")
    
    # ── STEP 3: Scoring ──────────────────────────────────────────────────
    print("  [3/4] Calculating score...")
    score_data = run_scoring(match_analysis, job_description, llm)
    total_score = score_data.get("total_score", 0)
    category = score_data.get("score_category", "N/A")
    print(f"       → Score: {total_score}/100  |  Category: {category}")
    
    # ── STEP 4: Explanation ──────────────────────────────────────────────
    print("  [4/4] Generating explanation report...")
    explanation = run_explanation(extracted_profile, match_analysis, score_data, llm)
    print(f"       → Report generated ({len(explanation)} chars)")
    
    return {
        "label": candidate_label,
        "name": candidate_name,
        "score": total_score,
        "category": category,
        "extracted_profile": extracted_profile,
        "match_analysis": match_analysis,
        "score_data": score_data,
        "explanation": explanation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  AI RESUME SCREENING SYSTEM")
    print("  Task 3 | Innomatics Research Labs")
    print("="*60)
    
    # ── 1. Validate API Keys ─────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Add it to your .env file."
        )
    
    # ── 2. Setup LangSmith Tracing ───────────────────────────────────────
    print("\n[*] Configuring LangSmith tracing...")
    try:
        setup_langsmith_tracing(project_name="Task3-AI-Resume-Screening")
    except EnvironmentError as e:
        print(f"  [!] Warning: {e}")
        print("  [!] Continuing without LangSmith tracing...\n")
    
    # ── 3. Initialize LLM ────────────────────────────────────────────────
    print("\n[*] Initializing LLM (GPT-3.5-turbo, temperature=0)...")
    llm = ChatOpenAI(
        api_key=openai_key,
        model="gpt-3.5-turbo",
        temperature=0,          # Deterministic outputs for consistency
        max_tokens=2000,
    )
    print("  [✓] LLM ready")
    
    # ── 4. Load Data ─────────────────────────────────────────────────────
    print("\n[*] Loading data files...")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    resumes = load_all_resumes(data_dir)
    job_description = load_job_description(data_dir)
    
    # ── 5. Run Pipeline for All Candidates ───────────────────────────────
    print("\n[*] Starting pipeline for all candidates...\n")
    
    all_results = []
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    
    for label, resume_text in resumes.items():
        # Run the full 4-step pipeline (traced via @traceable)
        result = run_pipeline_for_candidate(
            candidate_label=label,
            resume_text=resume_text,
            job_description=job_description,
            llm=llm,
        )
        
        all_results.append(result)
        
        # Format and save individual report
        report = format_pipeline_result(
            candidate_label=result["label"],
            extracted_profile=result["extracted_profile"],
            match_analysis=result["match_analysis"],
            score_data=result["score_data"],
            explanation=result["explanation"],
        )
        save_report(report, output_dir, label)
    
    # ── 6. Print Final Summary Table ─────────────────────────────────────
    print_summary_table(all_results)
    
    print(f"\n[✓] All reports saved to: {output_dir}/")
    print("[✓] LangSmith traces: https://smith.langchain.com\n")


if __name__ == "__main__":
    main()
