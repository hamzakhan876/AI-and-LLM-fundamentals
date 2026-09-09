import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# --------------------------------------------------
# Load the embedding model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Semantic Similarity Tool",
    page_icon="🔍",
    layout="centered"
)


# --------------------------------------------------
# App title
# --------------------------------------------------

st.title("🔍 Semantic Similarity Tool")

st.write(
    "Compare the semantic meaning of two pieces of text "
    "using embeddings and cosine similarity."
)


# --------------------------------------------------
# Text inputs
# --------------------------------------------------

text1 = st.text_area(
    "Enter the first text:",
    placeholder="Example: I love learning Python."
)

text2 = st.text_area(
    "Enter the second text:",
    placeholder="Example: I enjoy programming in Python."
)


# --------------------------------------------------
# Calculate similarity
# --------------------------------------------------

if st.button("Calculate Similarity"):

    if text1.strip() and text2.strip():

        # Convert both texts into embeddings
        embedding1 = model.encode(text1)
        embedding2 = model.encode(text2)

        # Calculate cosine similarity
        score = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        # Make sure the value stays between 0 and 1
        score = max(0.0, min(1.0, float(score)))


        # --------------------------------------------------
        # Similarity score
        # --------------------------------------------------

        st.subheader("📊 Similarity Score")

        st.metric(
            "Cosine Similarity",
            f"{score:.2f}"
        )

        # Progress bar
        st.progress(score)


        # --------------------------------------------------
        # Explain the result
        # --------------------------------------------------

        st.subheader("💡 Interpretation")

        if score >= 0.80:

            st.success(
                "🟢 Very Similar — The two texts have highly similar meanings."
            )

        elif score >= 0.60:

            st.info(
                "🟡 Moderately Similar — The two texts share some semantic meaning."
            )

        elif score >= 0.40:

            st.warning(
                "🟠 Somewhat Similar — The texts have limited semantic similarity."
            )

        else:

            st.error(
                "🔴 Not Very Similar — The two texts have different meanings."
            )


        # --------------------------------------------------
        # Similarity bar chart
        # --------------------------------------------------

        st.subheader("📈 Similarity Visualization")

        fig, ax = plt.subplots()

        ax.barh(
            ["Similarity"],
            [score]
        )

        ax.set_xlim(0, 1)

        ax.set_xlabel("Cosine Similarity")

        ax.set_title("Semantic Similarity Score")

        st.pyplot(fig)


        # --------------------------------------------------
        # PCA visualization
        # --------------------------------------------------

        st.subheader("🧠 Embedding Visualization")

        st.write(
            "The model creates high-dimensional embeddings. "
            "PCA reduces those dimensions to 2D so we can visualize "
            "the two embeddings as points."
        )

        # Combine embeddings
        embeddings = [embedding1, embedding2]

        # Reduce dimensions to 2
        pca = PCA(n_components=2)

        reduced_embeddings = pca.fit_transform(embeddings)


        # Create graph
        fig2, ax2 = plt.subplots()

        ax2.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            s=100
        )

        # Add labels
        ax2.annotate(
            "Text 1",
            (
                reduced_embeddings[0, 0],
                reduced_embeddings[0, 1]
            )
        )

        ax2.annotate(
            "Text 2",
            (
                reduced_embeddings[1, 0],
                reduced_embeddings[1, 1]
            )
        )

        ax2.set_xlabel("PCA Dimension 1")

        ax2.set_ylabel("PCA Dimension 2")

        ax2.set_title("2D Visualization of Embeddings")

        st.pyplot(fig2)


        # --------------------------------------------------
        # Show embeddings information
        # --------------------------------------------------

        st.subheader("🔢 Embedding Information")

        st.write(
            f"Each text was converted into a vector with "
            f"**{len(embedding1)} dimensions**."
        )

        st.write(
            "The embedding contains numerical values that represent "
            "semantic information about the text."
        )


    else:

        st.warning(
            "⚠️ Please enter text in both boxes."
        )