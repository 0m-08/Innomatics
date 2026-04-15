"""
Prompt template for extracting skills, experience, and tools from a resume.
"""

from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Skill Extraction Prompt
# ---------------------------------------------------------------------------

SKILL_EXTRACTION_TEMPLATE = """
You are a professional resume parser with expertise in technical hiring.

Your task is to extract structured information from the resume below.

**STRICT RULES:**
- Extract ONLY information explicitly mentioned in the resume.
- Do NOT infer, assume, or hallucinate any skills, tools, or experience.
- If a field has no information, return an empty list [].
- Do NOT add skills that are implied but not stated.

**RESUME:**
{resume_text}

**OUTPUT FORMAT (JSON only, no extra text):**
{{
  "name": "<candidate name>",
  "skills": ["<skill1>", "<skill2>", ...],
  "tools_and_technologies": ["<tool1>", "<tool2>", ...],
  "total_experience_years": <number or 0>,
  "education": "<highest degree and institution>",
  "certifications": ["<cert1>", "<cert2>", ...],
  "notable_projects": ["<project1>", "<project2>", ...]
}}

Return ONLY the JSON object. No explanation or extra text.
"""

extraction_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template=SKILL_EXTRACTION_TEMPLATE,
)
