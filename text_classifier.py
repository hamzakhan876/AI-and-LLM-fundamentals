import os
import json

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY was not found in the .env file.")
    st.stop()

client = Groq(api_key=api_key)


# ==========================================
# Pydantic Response Model
# ==========================================

class ClassificationResult(BaseModel):
    category: str
    confidence: float = Field(ge=0, le=100)
    explanation: str


# ==========================================
# Classification Categories
# ==========================================

CATEGORIES = {
    "Sentiment": [
        "Positive",
        "Neutral",
        "Negative"
    ],

    "Topic": [
        "Technology",
        "Business",
        "Sports",
        "Politics",
        "Entertainment",
        "Education",
        "Other"
    ],

    "Urgency": [
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
}


# ==========================================
# AI Classification Function
# ==========================================

def classify_text(text, classification_type):

    selected_categories = CATEGORIES[classification_type]

    prompt = f"""
You are a precise AI text classification system.

Classification type:
{classification_type}

Available categories:
{", ".join(selected_categories)}

Your task is to classify the user's text into EXACTLY ONE
of the available categories.

You must also provide:

1. confidence:
   A number between 0 and 100 representing your estimated
   confidence in the classification.

2. explanation:
   A short and clear explanation of why the text belongs
   to the selected category.

Important rules:

- The category MUST be one of the available categories.
- Do not create a new category.
- Confidence MUST be between 0 and 100.
- Return ONLY valid JSON.
- Do not include markdown.
- Do not include any text outside the JSON.

Required JSON structure:

{{
    "category": "category name",
    "confidence": 95,
    "explanation": "Short explanation"
}}

Text to classify:

{text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise text classification assistant. "
                    "Always follow the requested JSON structure."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        response_format={
            "type": "json_object"
        },

        temperature=0
    )

    raw_result = response.choices[0].message.content

    # Convert JSON string to Python object
    result_data = json.loads(raw_result)

    # Validate with Pydantic
    result = ClassificationResult.model_validate(result_data)

    # Extra safety check for category
    if result.category not in selected_categories:
        raise ValueError(
            f"AI returned invalid category: {result.category}"
        )

    return result


# ==========================================
# Streamlit Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI Text Classifier",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# Custom CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-top: 20px;
    }

    .category {
        font-size: 30px;
        font-weight: 700;
        text-align: center;
    }

    .confidence {
        font-size: 22px;
        font-weight: 600;
        text-align: center;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Header
# ==========================================

st.markdown(
    '<div class="main-title">🤖 AI Text Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Classify any text using AI-powered zero-shot classification.'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# ==========================================
# Classification Type
# ==========================================

classification_type = st.selectbox(
    "🎯 Classification Type",
    list(CATEGORIES.keys())
)


# ==========================================
# Show Available Categories
# ==========================================

st.caption(
    "Available categories: "
    + " • ".join(CATEGORIES[classification_type])
)


# ==========================================
# Text Input
# ==========================================

text = st.text_area(
    "📝 Enter text to classify",
    height=180,
    placeholder=(
        "Example: Our production server is down "
        "and customers cannot access the application."
    )
)


# ==========================================
# Classify Button
# ==========================================

if st.button(
    "🚀 Classify Text",
    use_container_width=True
):

    if not text.strip():

        st.warning("Please enter some text first.")

    else:

        with st.spinner("AI is analyzing your text..."):

            try:

                result = classify_text(
                    text,
                    classification_type
                )

                st.success("Classification completed!")


                # ==================================
                # Result
                # ==================================

                st.markdown(
                    '<div class="result-box">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="category">'
                    f'{result.category}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'Confidence: {result.confidence:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


                # ==================================
                # Confidence Progress Bar
                # ==================================

                st.write("### 📊 Confidence")

                st.progress(
                    int(result.confidence)
                )


                # ==================================
                # Explanation
                # ==================================

                st.write("### 💡 Explanation")

                st.info(result.explanation)


                # ==================================
                # Structured JSON
                # ==================================

                with st.expander(
                    "🔍 View Structured JSON"
                ):

                    st.json(
                        result.model_dump()
                    )


            except json.JSONDecodeError:

                st.error(
                    "The AI returned invalid JSON. "
                    "Please try again."
                )

            except ValueError as e:

                st.error(
                    f"Classification error: {e}"
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Built with Python • Streamlit • Groq • "
    "GPT-OSS-120B • Pydantic"
)