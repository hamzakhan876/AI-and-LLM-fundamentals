import streamlit as st
import editdistance
import pyarabic.araby as araby

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB


# --------------------------------------------------
# CAMeL Tools Analyzer
# --------------------------------------------------

@st.cache_resource
def load_analyzer():
    db = MorphologyDB.builtin_db()
    return Analyzer(db)


# --------------------------------------------------
# Arabic Text Normalization
# --------------------------------------------------

def normalize_arabic(text):
    # Remove Arabic diacritics
    text = araby.strip_tashkeel(text)

    # Normalize different forms of Arabic letters
    text = araby.normalize_hamza(text)

    return text


# --------------------------------------------------
# Arabic Text Analysis
# --------------------------------------------------

def analyze_arabic(text):
    analyzer = load_analyzer()

    # Normalize text
    normalized_text = normalize_arabic(text)

    # Tokenize sentence
    tokens = simple_word_tokenize(normalized_text)

    results = []

    for token in tokens:

        # Skip punctuation
        if not any(char.isalpha() for char in token):
            continue

        try:
            analysis = analyzer.analyze(token)

            if analysis:
                best_analysis = analysis[0]

                results.append({
                    "Word": token,
                    "Lemma": best_analysis.get("lex", "N/A"),
                    "Root": best_analysis.get("root", "N/A"),
                    "POS": best_analysis.get("pos", "N/A")
                })

            else:
                results.append({
                    "Word": token,
                    "Lemma": "N/A",
                    "Root": "N/A",
                    "POS": "N/A"
                })

        except Exception as e:
            results.append({
                "Word": token,
                "Lemma": "Error",
                "Root": "Error",
                "POS": str(e)
            })

    return normalized_text, tokens, results


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

st.set_page_config(
    page_title="Arabic NLP Analyzer",
    page_icon="🔤",
    layout="wide"
)

st.title("🔤 Arabic NLP Analyzer")

st.write(
    "Analyze Arabic text using PyArabic, CAMeL Tools, "
    "and Edit Distance."
)

st.divider()


# --------------------------------------------------
# Input
# --------------------------------------------------

text = st.text_area(
    "Enter Arabic text:",
    placeholder="مثال: هذا كتاب جميل",
    height=120
)


# --------------------------------------------------
# Analyze Button
# --------------------------------------------------

if st.button("Analyze Text"):

    if not text.strip():

        st.warning("Please enter some Arabic text first.")

    else:

        with st.spinner("Analyzing Arabic text..."):

            normalized_text, tokens, results = analyze_arabic(text)

        # ------------------------------------------
        # Normalized Text
        # ------------------------------------------

        st.subheader("1. Normalized Text")

        st.write(normalized_text)


        # ------------------------------------------
        # Tokens
        # ------------------------------------------

        st.subheader("2. Tokens")

        st.write(tokens)


        # ------------------------------------------
        # Morphological Analysis
        # ------------------------------------------

        st.subheader("3. Morphological Analysis")

        if results:
            st.dataframe(
                results,
                use_container_width=True
            )
        else:
            st.info("No analyzable Arabic words were found.")


        # ------------------------------------------
        # Edit Distance Demonstration
        # ------------------------------------------

        st.subheader("4. Edit Distance")

        if len(tokens) >= 2:

            word1 = tokens[0]
            word2 = tokens[1]

            distance = editdistance.eval(word1, word2)

            st.write(
                f"Edit distance between **{word1}** "
                f"and **{word2}**: **{distance}**"
            )

        else:

            st.info(
                "Enter at least two words to calculate "
                "edit distance."
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python, Streamlit, PyArabic, "
    "CAMeL Tools, and EditDistance."
)
