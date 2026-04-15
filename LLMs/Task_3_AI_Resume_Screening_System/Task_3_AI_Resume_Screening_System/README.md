# 🤖 AI Resume Screening System
### Task 3 | Innomatics Research Labs | Agentic AI Internship

---

## 📌 Overview

An end-to-end AI-powered resume screening pipeline built with **LangChain**, **OpenAI GPT**, and **LangSmith** tracing.

Given a Job Description and multiple resumes, the system:
1. **Extracts** skills, tools, and experience from each resume
2. **Matches** the candidate profile against the JD requirements
3. **Scores** the candidate from 0–100 with a structured breakdown
4. **Explains** the score with a detailed recruiter-style report

---

## 🗂️ Project Structure

```
Task_3_AI_Resume_Screening_System/
│
├── chains/
│   ├── extraction_chain.py      # Step 1: Skill extraction LCEL chain
│   ├── matching_chain.py        # Step 2: JD matching LCEL chain
│   ├── scoring_chain.py         # Step 3: Scoring LCEL chain (few-shot)
│   └── explanation_chain.py     # Step 4: Explanation LCEL chain
│
├── data/
│   ├── resume_strong.txt        # Strong candidate resume
│   ├── resume_average.txt       # Average candidate resume
│   ├── resume_weak.txt          # Weak candidate resume
│   └── job_description.txt      # Data Scientist JD
│
├── prompts/
│   ├── extraction_prompt.py     # PromptTemplate for extraction
│   ├── matching_prompt.py       # PromptTemplate for matching
│   ├── scoring_prompt.py        # PromptTemplate for scoring (few-shot)
│   └── explanation_prompt.py    # PromptTemplate for explanation
│
├── templates/
│   └── report_template.md       # Markdown template for output reports
│
├── utils/
│   ├── file_loader.py           # Load resumes & JD from disk
│   ├── report_formatter.py      # Format & save pipeline results
│   └── langsmith_tracer.py      # LangSmith setup & tagging helpers
│
├── app.py                       # Streamlit web UI
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone & Navigate
```bash
git clone https://github.com/<your-username>/Task_3_AI_Resume_Screening_System.git
cd Task_3_AI_Resume_Screening_System
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-...your_key_here...
LANGSMITH_API_KEY=lsv2_...your_key_here...
```

---

## 🚀 Running the System

### Option A: CLI (Recommended)
```bash
python main.py
```
Processes all 3 candidates and saves reports to `outputs/`.

### Option B: Streamlit Web UI
```bash
streamlit run app.py
```
Opens a browser-based interface at `http://localhost:8501`.

---

## 🔁 Pipeline Architecture

```
Resume Text
    │
    ▼
[Step 1] Extraction Chain
    └─► Skills, Tools, Experience, Education → JSON
    │
    ▼
[Step 2] Matching Chain
    └─► Matched Skills, Missing Skills, Experience Match → JSON
    │
    ▼
[Step 3] Scoring Chain  ← Few-Shot Examples
    └─► Score (0–100), Breakdown by Category → JSON
    │
    ▼
[Step 4] Explanation Chain
    └─► Recruiter-style Report (Strengths, Gaps, Recommendation)
    │
    ▼
LangSmith Tracing (all 4 steps visible per run)
```

---

## 🧠 LangChain Implementation Details

| Feature | Implementation |
|---|---|
| Prompt Templates | `PromptTemplate` from `langchain.prompts` |
| LCEL Chains | `prompt \| llm \| StrOutputParser()` |
| Chain Invocation | `.invoke({"key": value})` |
| Tracing | `@traceable` decorator + `LANGCHAIN_TRACING_V2=true` |
| Few-Shot Prompting | Embedded in `scoring_prompt.py` |
| Structured Output | JSON-constrained prompts, parsed with `json.loads()` |

---

## 📊 LangSmith Tracing

All runs are automatically traced to LangSmith with:
- **Project name:** `Task3-AI-Resume-Screening`
- **Tags:** `resume-screening`, `strong-candidate` / `average-candidate` / `weak-candidate`
- **3 runs visible:** Strong, Average, Weak
- **All 4 pipeline steps** visible as nested spans per run

View traces at: [https://smith.langchain.com](https://smith.langchain.com)

---

## 📋 Prompt Engineering Rules

All prompts follow strict anti-hallucination rules:

- ❌ Do NOT assume skills not present in the resume
- ❌ Do NOT inflate scores
- ✅ Base all outputs ONLY on the provided text
- ✅ Return structured JSON for machine-readability
- ✅ Few-shot examples ensure calibrated scoring

---

## 🏆 Sample Output

| Candidate | Score | Category |
|---|---|---|
| Strong (Arjun Mehta) | 91/100 | 🟢 Excellent |
| Average (Priya Sharma) | 52/100 | 🟡 Average |
| Weak (Rahul Patil) | 12/100 | 🔴 Poor |

---

## 👤 Author

**Om** | Agentic AI Intern @ Innomatics Research Labs  
GH Raisoni College of Engineering & Management, Pune  
GitHub:https://github.com/0m-08
