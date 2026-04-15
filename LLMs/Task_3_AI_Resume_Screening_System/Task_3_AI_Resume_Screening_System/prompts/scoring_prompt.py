"""
Prompt template for scoring the candidate based on match analysis.
Uses few-shot prompting for consistent, calibrated scoring.
"""

from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Scoring Prompt with Few-Shot Examples
# ---------------------------------------------------------------------------

SCORING_TEMPLATE = """
You are an AI hiring evaluator. Your job is to assign a numeric score (0-100) to a
candidate based on how well they match a job description.

**SCORING CRITERIA:**
- Core Technical Skills (ML/DL, Python, SQL): 35 points
- Cloud & MLOps (AWS, Docker, MLflow, etc.): 20 points
- Experience & Projects (years, real-world work): 25 points
- Education & Certifications: 10 points
- Communication & Soft Skills (inferred from resume quality): 10 points

**STRICT RULES:**
- Score must be an integer between 0 and 100.
- Do NOT inflate scores. Be strict and realistic.
- A score of 90+ means near-perfect match.
- A score below 40 means the candidate lacks most requirements.

---
**FEW-SHOT EXAMPLES:**

Example 1 (Strong Candidate):
Match Analysis: Has Python, PyTorch, TensorFlow, AWS SageMaker, Docker, MLflow,
5 years experience, NLP projects, M.Tech from IIT. Missing: Kafka.
Score: 91
Reasoning: Meets almost all technical and experience requirements. Minor gap in Kafka.

Example 2 (Average Candidate):
Match Analysis: Has Python, Scikit-learn, SQL, Power BI, 2 years experience,
basic ML projects. Missing: PyTorch, TensorFlow, AWS deployment, Docker, Spark.
Score: 52
Reasoning: Covers foundational skills but lacks deep learning, cloud, and MLOps experience.

Example 3 (Weak Candidate):
Match Analysis: Basic Python only, no ML frameworks, no cloud, no real datasets,
0 relevant experience, low CGPA.
Score: 12
Reasoning: Almost no overlap with job requirements. Only basic programming knowledge.
---

**NOW EVALUATE THIS CANDIDATE:**

**Match Analysis:**
{match_analysis}

**Job Description Summary:**
{job_description}

**OUTPUT FORMAT (JSON only, no extra text):**
{{
  "total_score": <integer 0-100>,
  "breakdown": {{
    "core_ml_skills": <integer 0-35>,
    "cloud_mlops": <integer 0-20>,
    "experience_projects": <integer 0-25>,
    "education_certifications": <integer 0-10>,
    "soft_skills": <integer 0-10>
  }},
  "score_category": "<Excellent | Good | Average | Below Average | Poor>"
}}

Return ONLY the JSON object. No explanation or extra text.
"""

scoring_prompt = PromptTemplate(
    input_variables=["match_analysis", "job_description"],
    template=SCORING_TEMPLATE,
)
