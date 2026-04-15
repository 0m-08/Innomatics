"""
Prompt template for generating a human-readable explanation of the candidate's score.
"""

from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Explanation Prompt
# ---------------------------------------------------------------------------

EXPLANATION_TEMPLATE = """
You are an experienced recruiter writing a professional candidate evaluation report.

Based on the information below, write a clear, structured explanation of why this
candidate received their score.

**STRICT RULES:**
- Be honest and specific.
- Do NOT mention skills the candidate does not have.
- Support every statement with evidence from the extracted profile or match analysis.
- Use professional, neutral language.
- Structure your response with clear sections.

**CANDIDATE PROFILE:**
{extracted_profile}

**MATCH ANALYSIS:**
{match_analysis}

**SCORE:**
{score_data}

**OUTPUT FORMAT:**
Write the explanation in the following structure:

## Candidate Evaluation Report

**Candidate Name:** <name>
**Final Score:** <score>/100 (<category>)

---

### Strengths
(List 3-5 specific strengths backed by the resume content)

### Gaps / Areas of Concern
(List 3-5 specific missing skills or experience gaps)

### Score Justification
(2-3 paragraphs explaining why this exact score was given)

### Hiring Recommendation
(One of: Strongly Recommended | Recommended | Borderline | Not Recommended)
(1-2 sentences with rationale)

---

Return the formatted report. Do NOT add anything outside this structure.
"""

explanation_prompt = PromptTemplate(
    input_variables=["extracted_profile", "match_analysis", "score_data"],
    template=EXPLANATION_TEMPLATE,
)
