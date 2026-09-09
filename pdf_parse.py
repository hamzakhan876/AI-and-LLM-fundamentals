import streamlit as st
import re
from PyPDF2 import PdfReader


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Text Extractor",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# TEXT CLEANING FUNCTION
# --------------------------------------------------

def clean_text(text):
    """
    Clean unnecessary whitespace from extracted PDF text.
    """

    # Replace multiple whitespace characters
    # such as spaces, tabs and newlines with one space
    text = re.sub(r"\s+", " ", text)

    # Remove whitespace from beginning and end
    text = text.strip()

    return text


# --------------------------------------------------
# TEXT STATISTICS FUNCTION
# --------------------------------------------------

def get_text_statistics(text):
    """
    Calculate basic statistics about the extracted text.
    """

    characters = len(text)
    words = len(text.split())
    lines = len(text.splitlines())

    return characters, words, lines


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📄 PDF Text Extractor")

st.write(
    "Upload a PDF file and extract, clean, analyze, and download its text."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Options")

cleaning_enabled = st.sidebar.checkbox(
    "Clean whitespace",
    value=True
)

show_raw_text = st.sidebar.checkbox(
    "Show raw extracted text",
    value=False
)


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF file",
    type=["pdf"]
)


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if uploaded_file is not None:

    try:

        # Read the uploaded PDF
        reader = PdfReader(uploaded_file)

        # Get total number of pages
        total_pages = len(reader.pages)

        st.success(
            f"PDF uploaded successfully! Total pages: {total_pages}"
        )


        # --------------------------------------------------
        # PDF INFORMATION
        # --------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "📑 Total Pages",
                total_pages
            )

        with col2:
            st.metric(
                "📦 File Size",
                f"{uploaded_file.size / 1024:.2f} KB"
            )


        # --------------------------------------------------
        # PAGE SELECTION
        # --------------------------------------------------

        st.subheader("📖 Select Pages")

        page_option = st.radio(
            "What would you like to extract?",
            ["All Pages", "Single Page"]
        )


        # --------------------------------------------------
        # EXTRACT TEXT
        # --------------------------------------------------

        extracted_text = ""


        if page_option == "All Pages":

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    extracted_text += page_text + "\n"


        else:

            selected_page = st.number_input(
                "Enter page number",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1
            )

            page = reader.pages[selected_page - 1]

            page_text = page.extract_text()

            if page_text:
                extracted_text = page_text


        # --------------------------------------------------
        # CHECK EXTRACTION
        # --------------------------------------------------

        if not extracted_text.strip():

            st.warning(
                "No readable text was found in this PDF."
            )

        else:

            # --------------------------------------------------
            # CLEAN TEXT
            # --------------------------------------------------

            if cleaning_enabled:

                final_text = clean_text(
                    extracted_text
                )

            else:

                final_text = extracted_text


            # --------------------------------------------------
            # STATISTICS
            # --------------------------------------------------

            st.subheader("📊 Text Statistics")

            characters, words, lines = get_text_statistics(
                final_text
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Characters",
                    characters
                )

            with col2:
                st.metric(
                    "Words",
                    words
                )

            with col3:
                st.metric(
                    "Lines",
                    lines
                )


            # --------------------------------------------------
            # RAW TEXT
            # --------------------------------------------------

            if show_raw_text:

                st.subheader(
                    "🔎 Raw Extracted Text"
                )

                st.text_area(
                    "Original PDF text",
                    extracted_text,
                    height=250
                )


            # --------------------------------------------------
            # CLEANED TEXT
            # --------------------------------------------------

            st.subheader(
                "✨ Extracted Text"
            )

            st.text_area(
                "Text output",
                final_text,
                height=400
            )


            # --------------------------------------------------
            # DOWNLOAD TEXT
            # --------------------------------------------------

            st.download_button(
                label="⬇️ Download Text File",
                data=final_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )


    except Exception as e:

        st.error(
            f"An error occurred while processing the PDF: {e}"
        )


else:

    st.info(
        "👆 Upload a PDF file to get started."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PDF Text Extractor | Built with Python, PyPDF2 and Streamlit"
)
