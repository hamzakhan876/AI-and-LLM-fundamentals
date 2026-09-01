from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

topic = input("Enter your Topic:")
tone = input("Enter Tone (e.g., professional, casual, humorous):")
length = input("Enter length in words:")

prompt = f"""
You are a professional AI writing assistant.

TASK:
Write an article about the following topic:
{topic}

TONE:
Use a {tone} tone.

LENGTH:
Write approximately {length} words.

REQUIREMENTS:
- Stay focused on the topic.
- Use clear and natural language.
- Match the requested tone.
- Avoid unnecessary repetition.
- Make the content informative and engaging.

OUTPUT FORMAT:
- Give the article a suitable title.
- Use short paragraphs.
- Do not include explanations about how you generated the content.
- Output only the final article.
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)



