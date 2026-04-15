"""
Report Formatter Utility
Formats and saves the final evaluation reports for each candidate.
"""

import os
import json
from datetime import datetime


def format_pipeline_result(
    candidate_label: str,
    extracted_profile: dict,
    match_analysis: dict,
    score_data: dict,
    explanation: str
) -> str:
    """
    Combine all pipeline outputs into a single formatted report string.
    
    Args:
        candidate_label: One of 'strong', 'average', 'weak'
        extracted_profile: Output from extraction chain
        match_analysis: Output from matching chain
        score_data: Output from scoring chain
        explanation: Output from explanation chain
        
    Returns:
        Formatted report as a Markdown string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
================================================================================
  AI RESUME SCREENING SYSTEM — CANDIDATE EVALUATION REPORT
  Generated: {timestamp}
  Candidate Type: {candidate_label.upper()}
================================================================================

{'='*80}
STEP 1: EXTRACTED PROFILE
{'='*80}
{json.dumps(extracted_profile, indent=2)}

{'='*80}
STEP 2: MATCH ANALYSIS
{'='*80}
{json.dumps(match_analysis, indent=2)}

{'='*80}
STEP 3: SCORE BREAKDOWN
{'='*80}
{json.dumps(score_data, indent=2)}

{'='*80}
STEP 4: EXPLANATION REPORT
{'='*80}
{explanation}

================================================================================
END OF REPORT
================================================================================
"""
    return report


def save_report(report: str, output_dir: str, candidate_label: str):
    """
    Save the formatted report to a text file.
    
    Args:
        report: Formatted report string
        output_dir: Directory to save the report
        candidate_label: Label used for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"report_{candidate_label}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"  [✓] Report saved: {filepath}")


def print_summary_table(results: list):
    """
    Print a clean summary table of all candidate scores.
    
    Args:
        results: List of dicts with candidate results
                 Each dict: {"label": str, "name": str, "score": int, "category": str}
    """
    print("\n")
    print("=" * 70)
    print("  FINAL SCREENING SUMMARY")
    print("=" * 70)
    print(f"  {'Candidate':<12} {'Name':<20} {'Score':>6}  {'Category':<20}")
    print("-" * 70)
    
    for r in results:
        label = r.get("label", "N/A").capitalize()
        name = r.get("name", "Unknown")[:20]
        score = r.get("score", 0)
        category = r.get("category", "N/A")
        
        # Color-code output based on score
        if score >= 80:
            indicator = "🟢"
        elif score >= 50:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"  {label:<12} {name:<20} {score:>6}  {indicator} {category}")
    
    print("=" * 70)
    print()
