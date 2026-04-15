"""
Prompt template for matching extracted resume data against job requirements.
"""

from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Matching Prompt
# ---------------------------------------------------------------------------

MATCHING_TEMPLATE = """
You are a senior technical recruiter comparing a candidate's profile against a job description.

Your task is to identify matched and missing requirements.

**STRICT RULES:**
- Base your analysis ONLY on the provided resume data and job description.
- Do NOT assume skills the candidate has not listed.
- Be objective and precise.

**CANDIDATE PROFILE (Extracted):**
{extracted_profile}

**JOB DESCRIPTION:**
{job_description}

**OUTPUT FORMAT (JSON only, no extra text):**
{{
  "matched_skills": ["<skill present in both resume and JD>", ...],
  "missing_skills": ["<skill required by JD but absent in resume>", ...],
  "matched_tools": ["<tool present in both>", ...],
  "missing_tools": ["<tool required but missing>", ...],
  "experience_match": {{
    "required_years": <number>,
    "candidate_years": <number>,
    "meets_requirement": <true/false>
  }},
  "education_match": {{
    "required": "<education requirement from JD>",
    "candidate_has": "<candidate education>",
    "meets_requirement": <true/false>
  }},
  "overall_match_summary": "<2-3 sentence summary of overall match>"
}}

Return ONLY the JSON object. No explanation or extra text.
"""

matching_prompt = PromptTemplate(
    input_variables=["extracted_profile", "job_description"],
    template=MATCHING_TEMPLATE,
)
