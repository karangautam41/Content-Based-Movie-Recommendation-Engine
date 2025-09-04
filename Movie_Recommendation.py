# --- Consolidated Imports (No Transformers) ---
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Simplified Configuration ---
DATA_DIR = 'data'
RAW_DATASET_PATH = os.path.join(DATA_DIR, 'netflix_titles.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.pkl')
# We now save a TF-IDF matrix instead of embeddings
TFIDF_MATRIX_PATH = os.path.join(DATA_DIR, 'tfidf_matrix.pkl')
VECTORIZER_PATH = os.path.join(DATA_DIR, 'vectorizer.pkl')


# --- 2. Preprocessing Logic (Using TF-IDF) ---
def run_preprocessing():
    """
    Loads raw data, processes it, generates a TF-IDF matrix, and saves the artifacts.
    This is the main function to be run once for setup.
    """
    print("Starting data preprocessing with TF-IDF...")

    # Load and clean data
    if not os.path.exists(RAW_DATASET_PATH):
        print(f"ERROR: Dataset not found at {RAW_DATASET_PATH}.")
        print("Please download the 'netflix_titles.csv' file from Kaggle and place it in the 'data' directory.")
        return

    df = pd.read_csv(RAW_DATASET_PATH)
    df = df[df['type'] == 'Movie'].copy()
    df.dropna(subset=['description'], inplace=True)
    df['description'] = df['description'].fillna('')
    df.reset_index(drop=True, inplace=True)
    print(f"Data loaded and cleaned. Shape: {df.shape}")

    # Generate TF-IDF matrix
    print("Generating TF-IDF matrix...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

    # Save artifacts
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(df, f)
    with open(TFIDF_MATRIX_PATH, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    print(f"Preprocessing complete. Artifacts saved to '{DATA_DIR}'.")

# --- 3. Recommendation Engine Class (Adapted for TF-IDF) ---
class RecommendationEngine:
    """
    Handles the core logic for generating recommendations using the TF-IDF matrix.
    """
    def __init__(self):
        with open(PROCESSED_DATA_PATH, 'rb') as f:
            self.df = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, 'rb') as f:
            self.tfidf_matrix = pickle.load(f)

    def get_recommendations(self, title, top_n=5):
        """
        Finds movies similar to the given title based on TF-IDF similarity.
        """
        try:
            idx = self.df[self.df['title'] == title].index[0]
        except IndexError:
            return pd.DataFrame()

        query_vector = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        similar_movie_indices = sim_scores.argsort()[::-1][1:top_n+1]

        return self.df.iloc[similar_movie_indices]

# --- 4. Streamlit Application (Corrected Logic) ---
def run_app():
    """
    Defines and runs the Streamlit user interface.
    """
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide"
    )

    @st.cache_resource
    def load_engine():
        """Loads the recommendation engine using Streamlit's cache."""
        try:
            return RecommendationEngine()
        except FileNotFoundError:
            st.error(
                "Model artifacts not found! Please run the preprocessing step first. "
                "From your terminal, run: `python app.py --preprocess`"
            )
            return None # Return None on failure

    engine = load_engine()

    # --- Main App Logic: Only runs if the engine loaded successfully ---
    if engine:
        movie_titles = engine.df['title'].tolist()

        st.sidebar.header("🎬 Movie Recommender")
        selected_movie = st.sidebar.selectbox("Choose a movie you like:", options=movie_titles)
        num_recommendations = st.sidebar.slider("Number of recommendations:", 3, 10, 5)

        st.title("Content-Based Movie Recommender")
        st.markdown("---")

        if st.sidebar.button("Get Recommendations", use_container_width=True):
            st.markdown(f"### Because you liked '{selected_movie}', you might also like...")
            with st.spinner('Finding similar movies...'):
                recommendations = engine.get_recommendations(selected_movie, top_n=num_recommendations)

            if not recommendations.empty:
                cols = st.columns(num_recommendations)
                for i, (_, row) in enumerate(recommendations.iterrows()):
                    with cols[i]:
                        st.markdown(f"**{row['title']}** ({row['release_year']})")
                        st.markdown(
                            f'<div style="background-color: #262730; border-radius: 10px; padding: 10px; height: 200px; display: flex; align-items: center; justify-content: center; text-align: center;">'
                            f'<span style="color: #FAFAFA; font-size: 16px;">{row["title"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        with st.expander("See description"):
                            st.write(row['description'])
            else:
                st.warning("Could not find recommendations.")
        else:
            st.info("Select a movie from the sidebar to get started.")

# --- 5. Main Execution Block (Controller) ---
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--preprocess':
        run_preprocessing()
    else:
        run_app()

