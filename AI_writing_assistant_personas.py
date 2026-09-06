import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("AI Writing Assistant")
st.write("Generate content using AI with your preferred tone and length.")



# User inputs
topic = st.text_input("Enter your topic")

tone = st.selectbox(
    "Choose a tone",
    ["Professional", "Casual", "Humorous"]
)

length = st.slider(
    "Choose length (words)",
    min_value=50,
    max_value=1000,            
    value=250,
    step=50
)

# Generate button
if st.button("Generate Content"):

    if not topic:
        st.warning("Please enter a topic.")

    else:
        prompt = f"""
Write an article about:
{topic}

Tone: {tone}

Length: approximately {length} words.

Requirements:
- Stay focused on the topic.
- Use clear and natural language.
- Match the requested tone.
- Avoid unnecessary repetition.
- Give the article a suitable title.
- Output only the final article.
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a Dubai Business Writer.

Your writing style is:
- Professional and polished
- Business-focused
- Clear and concise
- Culturally aware of the UAE and Dubai
- Suitable for business professionals and organizations

Rules:
- Respect UAE culture and business etiquette.
- Avoid stereotypes and culturally insensitive language.
- Focus on practical business value.
- Do not invent statistics, companies, laws, or facts.
- Match the user's requested tone while remaining professional.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = response.choices[0].message.content

        st.subheader("Generated Content")
        st.write(result)

        st.download_button(
            label="⬇️ Download Content",
            data=result,
            file_name="generated_content.txt",
            mime="text/plain"
        )