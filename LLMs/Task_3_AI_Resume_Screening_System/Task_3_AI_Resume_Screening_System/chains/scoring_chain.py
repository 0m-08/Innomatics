"""
Scoring Chain — Step 3 of the Resume Screening Pipeline.
Assigns a numeric score (0–100) to the candidate based on match analysis.
Uses few-shot prompting + LCEL with .invoke()
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.scoring_prompt import scoring_prompt


def build_scoring_chain(llm: ChatOpenAI):
    """
    Build and return the scoring LCEL chain.
    
    Pipeline:
        scoring_prompt | llm | StrOutputParser
    
    Args:
        llm: Initialized ChatOpenAI instance
        
    Returns:
        LCEL chain for candidate scoring
    """
    chain = scoring_prompt | llm | StrOutputParser()
    return chain


def run_scoring(match_analysis: dict, job_description: str, llm: ChatOpenAI) -> dict:
    """
    Execute the scoring chain for a candidate.
    
    Args:
        match_analysis: Dict output from matching chain
        job_description: Raw job description text
        llm: Initialized ChatOpenAI instance
        
    Returns:
        Parsed dict with score breakdown
    """
    chain = build_scoring_chain(llm)
    
    # Convert match analysis dict to string for prompt
    match_str = json.dumps(match_analysis, indent=2)
    
    # Invoke with LCEL
    raw_output = chain.invoke({
        "match_analysis": match_str,
        "job_description": job_description
    })
    
    # Clean and parse JSON
    cleaned = re.sub(r"```json|```", "", raw_output).strip()
    
    try:
        score_result = json.loads(cleaned)
    except json.JSONDecodeError:
        score_result = {"raw_output": raw_output, "parse_error": True}
    
    return score_result
