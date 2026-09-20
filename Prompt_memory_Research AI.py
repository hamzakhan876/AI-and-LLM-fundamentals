import os

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


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
# GROQ / LANGCHAIN MODEL
# ==================================================

if not api_key:
    st.error(
        "GROQ_API_KEY was not found. "
        "Please add it to your .env file."
    )
    st.stop()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.2
)


# ==================================================
# STREAMLIT CONVERSATION MEMORY
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
# LANGCHAIN PROMPT
# ==================================================

research_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI Research Assistant.

Your job is to answer questions using ONLY
the information provided in the document context
and the conversation history.

Rules:

1. Use the provided document context.
2. Use conversation history to understand follow-up questions.
3. Do not invent information.
4. Do not use outside knowledge.
5. If the answer cannot be found in the document context,
   clearly say that the information is not available
   in the provided document.
6. Give a clear and concise answer.
"""
        ),
        MessagesPlaceholder(
            variable_name="history"
        ),
        (
            "human",
            """
DOCUMENT CONTEXT:

{context}

QUESTION:

{question}
"""
        )
    ]
)


# ==================================================
# ASK AI USING LANGCHAIN
# ==================================================

def ask_ai(question, relevant_chunks):
    """
    Send the question, document context, and
    conversation history through a LangChain prompt
    and Groq model.
    """

    context = "\n\n".join(
        relevant_chunks
    )

    response = research_prompt | llm

    result = response.invoke(
        {
            "context": context,
            "question": question,
            "history": st.session_state.chat_history
        }
    )

    return result.content


# ==================================================
# STREAMLIT APPLICATION
# ==================================================

st.title("📚 AI Research Assistant")

st.write(
    "Upload a research paper and ask questions "
    "about its content. The assistant remembers "
    "your conversation."
)


# ==================================================
# CLEAR CONVERSATION
# ==================================================

if st.button("🗑️ Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()


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
    # CONVERSATION HISTORY
    # ==================================================

    if st.session_state.chat_history:

        st.subheader("💬 Conversation")

        for message in st.session_state.chat_history:

            if isinstance(message, HumanMessage):

                with st.chat_message("user"):
                    st.write(message.content)

            elif isinstance(message, AIMessage):

                with st.chat_message("assistant"):
                    st.write(message.content)


    # ==================================================
    # ASK QUESTION
    # ==================================================

    st.subheader("🔎 Ask a Question")

    question = st.chat_input(
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
            # ASK AI
            # ==================================================

            with st.spinner(
                "🤖 Reading the document..."
            ):

                try:

                    answer = ask_ai(
                        question,
                        relevant_chunks
                    )

                    # Save conversation to memory
                    st.session_state.chat_history.append(
                        HumanMessage(content=question)
                    )

                    st.session_state.chat_history.append(
                        AIMessage(content=answer)
                    )

                    # Display current exchange
                    with st.chat_message("user"):
                        st.write(question)

                    with st.chat_message("assistant"):
                        st.write(answer)

                except Exception as e:

                    st.error(
                        f"An error occurred: {e}"
                    )
