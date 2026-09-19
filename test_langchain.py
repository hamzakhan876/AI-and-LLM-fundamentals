import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv()


# -----------------------------
# LangChain Model
# -----------------------------

model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# -----------------------------
# Prompt Template
# -----------------------------

prompt = PromptTemplate.from_template(
    """
    You are a professional email writing assistant.

    Write a clear and professional email based on the information below.

    Purpose:
    {purpose}

    Recipient:
    {recipient}

    Tone:
    {tone}

    Additional details:
    {details}

    Requirements:
    - Include a suitable subject line.
    - Write a professional greeting.
    - Clearly communicate the purpose.
    - Keep the email concise and natural.
    - End with a professional closing.
    """
)


# -----------------------------
# Create LangChain Chain
# -----------------------------

chain = prompt | model


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="AI Email Generator",
    page_icon="✉️",
    layout="centered"
)

st.title("✉️ AI Email Generator")
st.write("Generate professional emails using LangChain and AI.")


purpose = st.text_input(
    "Email Purpose",
    placeholder="e.g. Request a job interview"
)

recipient = st.text_input(
    "Recipient",
    placeholder="e.g. Hiring Manager"
)

tone = st.selectbox(
    "Tone",
    [
        "Professional",
        "Friendly",
        "Formal",
        "Polite",
        "Confident"
    ]
)

details = st.text_area(
    "Additional Details",
    placeholder="Enter any important information you want to include..."
)


if st.button("Generate Email"):

    if not purpose:
        st.warning("Please enter the email purpose.")

    else:

        with st.spinner("Generating email..."):

            response = chain.invoke({
                "purpose": purpose,
                "recipient": recipient,
                "tone": tone,
                "details": details
            })

        st.subheader("Generated Email")

        st.write(response.content)