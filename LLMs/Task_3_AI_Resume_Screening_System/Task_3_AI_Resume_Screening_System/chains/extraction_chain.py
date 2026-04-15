"""
Extraction Chain — Step 1 of the Resume Screening Pipeline.
Extracts structured skills, tools, and experience from raw resume text.
Uses LCEL (LangChain Expression Language) with .invoke()
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Import our custom prompt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.extraction_prompt import extraction_prompt


def build_extraction_chain(llm: ChatOpenAI):
    """
    Build and return the extraction LCEL chain.
    
    Pipeline:
        extraction_prompt | llm | StrOutputParser
    
    Args:
        llm: Initialized ChatOpenAI instance
        
    Returns:
        LCEL chain for skill extraction
    """
    # LCEL chain: prompt → LLM → string output parser
    chain = extraction_prompt | llm | StrOutputParser()
    return chain


def run_extraction(resume_text: str, llm: ChatOpenAI) -> dict:
    """
    Execute the extraction chain on a given resume.
    
    Args:
        resume_text: Raw resume text string
        llm: Initialized ChatOpenAI instance
        
    Returns:
        Parsed dict with extracted candidate profile
    """
    chain = build_extraction_chain(llm)
    
    # Invoke chain with LangChain Expression Language
    raw_output = chain.invoke({"resume_text": resume_text})
    
    # Clean up the output (strip markdown code fences if present)
    cleaned = re.sub(r"```json|```", "", raw_output).strip()
    
    # Parse JSON output
    try:
        extracted = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: return raw string in a dict if JSON parsing fails
        extracted = {"raw_output": raw_output, "parse_error": True}
    
    return extracted
