"""
LangSmith Tracing Utility
Configures LangSmith tracing for all pipeline runs.
Provides helper functions for tagging and annotating runs.
"""

import os


def setup_langsmith_tracing(project_name: str = "AI-Resume-Screening"):
    """
    Configure environment variables to enable LangSmith tracing.
    
    Reads API key from environment. Raises error if not set.
    
    Args:
        project_name: LangSmith project name for grouping runs
        
    Raises:
        EnvironmentError: If LANGSMITH_API_KEY is not set
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "LANGSMITH_API_KEY is not set in environment variables.\n"
            "Please add it to your .env file:\n"
            "  LANGSMITH_API_KEY=<your_key_here>"
        )
    
    # Enable LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    
    print(f"  [✓] LangSmith tracing enabled → Project: '{project_name}'")
    print(f"      View runs at: https://smith.langchain.com")


def get_run_tags(candidate_label: str) -> list:
    """
    Generate LangSmith tags for a specific candidate run.
    
    Args:
        candidate_label: 'strong', 'average', or 'weak'
        
    Returns:
        List of tag strings for LangSmith
    """
    base_tags = ["resume-screening", "task-3", "innomatics"]
    
    candidate_tags = {
        "strong": ["strong-candidate", "high-score"],
        "average": ["average-candidate", "mid-score"],
        "weak": ["weak-candidate", "low-score"],
    }
    
    return base_tags + candidate_tags.get(candidate_label, [])


def get_run_metadata(candidate_label: str, candidate_name: str) -> dict:
    """
    Generate metadata dict for LangSmith run annotation.
    
    Args:
        candidate_label: 'strong', 'average', or 'weak'
        candidate_name: Full name of the candidate
        
    Returns:
        Metadata dict for LangSmith
    """
    return {
        "candidate_type": candidate_label,
        "candidate_name": candidate_name,
        "pipeline": "resume-screening-v1",
        "steps": ["extraction", "matching", "scoring", "explanation"],
    }
