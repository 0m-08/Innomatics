"""
Explanation Chain — Step 4 of the Resume Screening Pipeline.
Generates a human-readable evaluation report explaining the candidate's score.
Uses LCEL (LangChain Expression Language) with .invoke()
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.explanation_prompt import explanation_prompt


def build_explanation_chain(llm: ChatOpenAI):
    """
    Build and return the explanation LCEL chain.
    
    Pipeline:
        explanation_prompt | llm | StrOutputParser
    
    Args:
        llm: Initialized ChatOpenAI instance
        
    Returns:
        LCEL chain for generating explanation reports
    """
    chain = explanation_prompt | llm | StrOutputParser()
    return chain


def run_explanation(
    extracted_profile: dict,
    match_analysis: dict,
    score_data: dict,
    llm: ChatOpenAI
) -> str:
    """
    Execute the explanation chain to generate a recruiter-style report.
    
    Args:
        extracted_profile: Dict from extraction chain
        match_analysis: Dict from matching chain
        score_data: Dict from scoring chain
        llm: Initialized ChatOpenAI instance
        
    Returns:
        Formatted string report (Markdown)
    """
    chain = build_explanation_chain(llm)
    
    # Invoke with LCEL — pass all context as stringified JSON
    explanation = chain.invoke({
        "extracted_profile": json.dumps(extracted_profile, indent=2),
        "match_analysis": json.dumps(match_analysis, indent=2),
        "score_data": json.dumps(score_data, indent=2)
    })
    
    return explanation
