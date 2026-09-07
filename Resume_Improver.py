from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_resume(resume):

    prompt = f"""
You are a professional resume reviewer.

Analyze the resume below.

Return your analysis using exactly these five sections:

1. WEAKNESSES
- List the main weaknesses.

2. MISSING INFORMATION
- List important information that is missing.

3. POORLY WRITTEN SECTIONS
- Identify sections or statements that need improvement.

4. SKILLS TO HIGHLIGHT
- Identify relevant skills already present in the resume.

5. IMPROVEMENT SUGGESTIONS
- Give specific suggestions for improving the resume.

Important:
- Only identify information that can be supported by the resume.
- Do not invent facts.
- Be specific and concise.

Resume:
{resume}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

resume = input("Paste your resume:\n")

analysis = analyze_resume(resume)

def rewrite_resume(resume, analysis):

    prompt = f"""

You are a professional resume writer.

Your job is to rewrite the original resume based on the professional analysis from Step 1.

Use the analysis as a guide, but only use facts that appear in the original resume.

Do not invent:
- Work experience
- Job titles
- Education
- Skills
- Certifications
- Achievements
- Numbers or statistics

Improve:
- Professional language
- Clarity
- Structure
- Action verbs
- Conciseness
- Presentation of existing skills and experience

Original Resume:
{resume}

STEP 1 — RESUME ANALYSIS:
{analysis}

Return only the improved resume.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

analysis = analyze_resume(resume)

print("\n===== RESUME ANALYSIS =====")
print(analysis)

improved_resume = rewrite_resume(resume, analysis)

print("\n===== IMPROVED RESUME =====")
print(improved_resume)