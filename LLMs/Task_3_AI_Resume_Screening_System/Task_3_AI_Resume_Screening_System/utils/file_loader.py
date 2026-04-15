"""
File Loader Utility
Handles reading resume and job description text files from disk.
"""

import os


def load_text_file(filepath: str) -> str:
    """
    Load and return the contents of a text file.
    
    Args:
        filepath: Absolute or relative path to the text file
        
    Returns:
        File contents as a string
        
    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the file cannot be read
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    return content.strip()


def load_all_resumes(data_dir: str) -> dict:
    """
    Load all resume files from the data directory.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Dict mapping label to resume text:
        {
            "strong": "<resume text>",
            "average": "<resume text>",
            "weak": "<resume text>"
        }
    """
    resumes = {}
    
    resume_files = {
        "strong": "resume_strong.txt",
        "average": "resume_average.txt",
        "weak": "resume_weak.txt",
    }
    
    for label, filename in resume_files.items():
        filepath = os.path.join(data_dir, filename)
        resumes[label] = load_text_file(filepath)
        print(f"  [✓] Loaded resume: {filename}")
    
    return resumes


def load_job_description(data_dir: str) -> str:
    """
    Load the job description file.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Job description as a string
    """
    filepath = os.path.join(data_dir, "job_description.txt")
    jd = load_text_file(filepath)
    print(f"  [✓] Loaded job description: job_description.txt")
    return jd
