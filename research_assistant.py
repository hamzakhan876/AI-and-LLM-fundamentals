import os

import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
from dotenv import load_dotenv


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# GROQ CLIENT
# ==================================================

if not api_key:
    st.error(
        "GROQ_API_KEY was not found. "
        "Please add it to your .env file."
    )
    st.stop()

client = Groq(api_key=api_key)


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):
    """
    Clean extracted PDF text.
    """

    text = text.replace("\x00", " ")

    # Remove unnecessary whitespace
    text = " ".join(text.split())

    return text.strip()


# ==================================================
# TEXT CHUNKING
# ==================================================

def chunk_text(text, chunk_size=2000):
    """
    Divide large document text into smaller chunks.
    """

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        chunks.append(chunk)

    return chunks


# ==================================================
# RETRIEVAL
# ==================================================

def retrieve_relevant_chunks(
    question,
    chunks,
    top_k=3
):
    """
    Find chunks containing the most words
    related to the user's question.
    """

    question_words = set(
        question.lower().split()
    )

    # Common words that are not useful for retrieval
    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "for",
        "and",
        "or",
        "this",
        "that",
        "document",
        "does",
        "do",
        "how",
        "why",
        "which",
        "who"
    }

    # Remove stop words
    question_words = {
        word
        for word in question_words
        if word not in stop_words
    }

    scored_chunks = []

    for chunk in chunks:

        chunk_words = set(
            chunk.lower().split()
        )

        score = len(
            question_words.intersection(
                chunk_words
            )
        )

        scored_chunks.append(
            (score, chunk)
        )

    # Highest score first
    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Select the best chunks
    relevant_chunks = [
        chunk
        for score, chunk in scored_chunks[:top_k]
        if score > 0
    ]

    return relevant_chunks


# ==================================================
# ASK GROQ
# ==================================================

def ask_groq(question, relevant_chunks):
    """
    Send the question and document context to Groq.
    """

    context = "\n\n".join(
        relevant_chunks
    )

    system_prompt = """
You are an AI Research Assistant.

Your job is to answer questions using ONLY
the information provided in the document context.

Rules:

1. Use the provided document context.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer cannot be found in the context,
   clearly say that the information is not available
   in the provided document.
5. Give a clear and concise answer.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


QUESTION:

{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# ==================================================
# STREAMLIT APPLICATION
# ==================================================

st.title("📚 AI Research Assistant")

st.write(
    "Upload a research paper and ask questions "
    "about its content."
)


# ==================================================
# PDF UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload your PDF research paper",
    type=["pdf"]
)


if uploaded_file is not None:

    # ==================================================
    # EXTRACT PDF TEXT
    # ==================================================

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        try:
            page_text = page.extract_text(
                extraction_mode="layout"
            )

        except TypeError:
            page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    # ==================================================
    # CHECK EXTRACTED TEXT
    # ==================================================

    if not text.strip():

        st.error(
            "No readable text was found in this PDF."
        )

        st.stop()


    # ==================================================
    # CLEAN TEXT
    # ==================================================

    cleaned_text = clean_text(text)


    # ==================================================
    # CREATE CHUNKS
    # ==================================================

    chunks = chunk_text(
        cleaned_text,
        chunk_size=2000
    )


    # ==================================================
    # DOCUMENT INFORMATION
    # ==================================================

    st.success(
        "PDF processed successfully!"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            len(reader.pages)
        )

    with col2:

        st.metric(
            "Characters",
            len(cleaned_text)
        )

    with col3:

        st.metric(
            "Chunks",
            len(chunks)
        )


    # ==================================================
    # DOCUMENT PREVIEW
    # ==================================================

    with st.expander(
        "📄 View extracted document text"
    ):

        st.text_area(
            "Document",
            cleaned_text,
            height=300
        )


    # ==================================================
    # ASK QUESTION
    # ==================================================

    st.subheader("🔎 Ask a Question")

    question = st.text_input(
        "What would you like to know about the document?"
    )


    if question:

        # ==================================================
        # RETRIEVE RELEVANT CHUNKS
        # ==================================================

        relevant_chunks = retrieve_relevant_chunks(
            question,
            chunks,
            top_k=3
        )


        # ==================================================
        # CHECK RETRIEVAL
        # ==================================================

        if not relevant_chunks:

            st.warning(
                "I could not find relevant information "
                "in the document."
            )

        else:

            # ==================================================
            # SHOW RETRIEVED CONTEXT
            # ==================================================

            with st.expander(
                "🎯 View retrieved document context"
            ):

                for i, chunk in enumerate(
                    relevant_chunks
                ):

                    st.markdown(
                        f"**Relevant Chunk {i + 1}**"
                    )

                    st.write(chunk)


            # ==================================================
            # ASK GROQ
            # ==================================================

            with st.spinner(
                "🤖 Reading the document..."
            ):

                try:

                    answer = ask_groq(
                        question,
                        relevant_chunks
                    )

                    st.subheader(
                        "💡 Answer"
                    )

                    st.write(answer)

                except Exception as e:

                    st.error(
                        f"An error occurred: {e}"
                    )