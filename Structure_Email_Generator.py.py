import streamlit as st
import json
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import os


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------------
# Pydantic Model
# -----------------------------

class EmailResponse(BaseModel):
    subject: str
    greeting: str
    body: str
    closing: str


# -----------------------------
# Generate Email
# -----------------------------

def generate_email(topic, tone, recipient):

    prompt = f"""
You are a professional email writing assistant.

Create a professional email based on the following information:

Recipient: {recipient}
Topic: {topic}
Tone: {tone}

Return ONLY valid JSON.

The JSON must follow exactly this structure:

{{
    "subject": "email subject",
    "greeting": "email greeting",
    "body": "email body",
    "closing": "email closing"
}}

Do not add markdown.
Do not add explanations outside the JSON.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": "You generate professional emails in valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        response_format={
            "type": "json_object"
        },

        temperature=0.7
    )

    raw_response = response.choices[0].message.content

    # Convert JSON string into Python dictionary
    email_data = json.loads(raw_response)

    # Validate using Pydantic
    validated_email = EmailResponse(**email_data)

    return validated_email


# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Email Generator",
    page_icon="📧",
    layout="centered"
)


# -----------------------------
# Header
# -----------------------------

st.title("📧 AI Email Generator")

st.write(
    "Generate professional emails using AI with structured JSON outputs "
    "and Pydantic validation."
)


st.divider()


# -----------------------------
# User Inputs
# -----------------------------

recipient = st.text_input(
    "👤 Recipient",
    placeholder="e.g. Hiring Manager"
)

topic = st.text_area(
    "📝 What is the email about?",
    placeholder="e.g. I want to apply for a Junior AI Engineer position."
)

tone = st.selectbox(
    "🎯 Email Tone",
    [
        "Professional",
        "Friendly",
        "Formal",
        "Persuasive",
        "Casual"
    ]
)


# -----------------------------
# Generate Button
# -----------------------------

if st.button("✨ Generate Email", use_container_width=True):

    if not recipient or not topic:
        st.warning("Please enter both the recipient and email topic.")

    else:

        with st.spinner("AI is writing your email..."):

            try:

                email = generate_email(
                    topic,
                    tone,
                    recipient
                )

                st.success("Email generated successfully!")

                # -----------------------------
                # Display Subject
                # -----------------------------

                st.subheader("📌 Subject")

                st.text_input(
                    "Email Subject",
                    value=email.subject,
                    key="generated_subject"
                )

                # -----------------------------
                # Display Email
                # -----------------------------

                st.subheader("📨 Email")

                full_email = f"""{email.greeting}

{email.body}

{email.closing}"""

                st.text_area(
                    "Generated Email",
                    value=full_email,
                    height=300
                )

                # -----------------------------
                # Structured JSON
                # -----------------------------

                with st.expander("🔍 View Structured JSON"):

                    st.json(email.model_dump())

            except json.JSONDecodeError:

                st.error(
                    "The AI returned an invalid JSON response. Please try again."
                )

            except ValidationError as e:

                st.error(
                    "The AI response did not match the required structure."
                )

                st.code(str(e))

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Built with Python • Streamlit • Groq • Pydantic"
)