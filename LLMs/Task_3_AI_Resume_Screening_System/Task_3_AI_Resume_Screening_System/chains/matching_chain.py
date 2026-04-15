"""
Matching Chain — Step 2 of the Resume Screening Pipeline.
Compares extracted candidate profile against the job description requirements.
Uses LCEL (LangChain Expression Language) with .invoke()
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.matching_prompt import matching_prompt


def build_matching_chain(llm: ChatOpenAI):
    """
    Build and return the matching LCEL chain.
    
    Pipeline:
        matching_prompt | llm | StrOutputParser
    
    Args:
        llm: Initialized ChatOpenAI instance
        
    Returns:
        LCEL chain for skill matching
    """
    chain = matching_prompt | llm | StrOutputParser()
    return chain


def run_matching(extracted_profile: dict, job_description: str, llm: ChatOpenAI) -> dict:
    """
    Execute the matching chain given an extracted profile and JD.
    
    Args:
        extracted_profile: Dict output from extraction chain
        job_description: Raw job description text
        llm: Initialized ChatOpenAI instance
        
    Returns:
        Parsed dict with match analysis results
    """
    chain = build_matching_chain(llm)
    
    # Convert extracted profile dict to formatted string for prompt
    profile_str = json.dumps(extracted_profile, indent=2)
    
    # Invoke with LCEL
    raw_output = chain.invoke({
        "extracted_profile": profile_str,
        "job_description": job_description
    })
    
    # Clean and parse JSON
    cleaned = re.sub(r"```json|```", "", raw_output).strip()
    
    try:
        match_result = json.loads(cleaned)
    except json.JSONDecodeError:
        match_result = {"raw_output": raw_output, "parse_error": True}
    
    return match_result
