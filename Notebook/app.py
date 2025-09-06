import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy.sparse import load_npz
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediRec AI Recommender",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- STYLING ---
def load_css():
    """Inject custom CSS for styling, including the animated background."""
    st.markdown("""
    <style>
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .stApp {
            background: linear-gradient(-45deg, #e0f7fa, #e8eaf6, #d1c4e9, #c5cae9);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            color: #333;
        }
        .stTextInput > div > div > input {
            border-radius: 20px;
            border: 2px solid #b0bec5;
            padding: 10px 15px;
        }
        .stButton > button {
            border-radius: 20px;
            border: 2px solid #3f51b5;
            background-color: #3f51b5;
            color: white;
        }
        [data-testid="stExpander"] {
            border-radius: 15px;
            border: 1px solid #ddd;
            background-color: rgba(255, 255, 255, 0.6);
        }
        .result-card {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model_components(vectorizer_path, matrix_path, df_path):
    """
    Loads the pre-trained model components from disk.
    Uses Streamlit's caching to load only once.
    """
    if not all(os.path.exists(p) for p in [vectorizer_path, matrix_path, df_path]):
        st.error("Model files not found! Please run 'train_model.py' first.", icon="🚨")
        return None, None, None
        
    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
        
    tfidf_matrix = load_npz(matrix_path)
    df = pd.read_pickle(df_path)
    
    return tfidf_vectorizer, tfidf_matrix, df

# --- RECOMMENDATION LOGIC ---
def get_recommendations_from_query(query, df, tfidf_vectorizer, tfidf_matrix):
    """
    Finds medicines similar to a given text query.
    """
    with st.spinner('Analyzing your query...'):
        query_tfidf = tfidf_vectorizer.transform([query])
        sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[-5:][::-1]
    return df.iloc[top_indices]

def get_substitutes(medicine_name, df):
    """
    Retrieves the list of substitute medicines.
    """
    try:
        medicine_row = df[df['name'] == medicine_name].iloc[0]
        sub_cols = ['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']
        substitutes = [sub for sub in medicine_row[sub_cols] if sub]
        return substitutes
    except IndexError:
        return []

# --- UI RENDERING ---
def main():
    load_css()
    
    st.title("MediRec AI 💊")
    st.markdown("Your AI-Powered Medicine Recommendation Assistant")
    
    # Load model
    vectorizer, matrix, df = load_model_components(
        'tfidf_vectorizer.pkl', 
        'tfidf_matrix.npz', 
        'processed_data.pkl'
    )

    if df is not None:
        user_input = st.text_input(
            "Enter a medicine name, symptom, or description:",
            placeholder="e.g., 'headache' or 'Augmentin'",
            key="search_input"
        )

        if user_input:
            recommended_medicines = get_recommendations_from_query(user_input, df, vectorizer, matrix)
            
            if recommended_medicines.empty:
                st.warning("Could not find any matching medicines. Please try another query.", icon="⚠️")
            else:
                top_recommendation = recommended_medicines.iloc[0]

                st.markdown("---")
                st.subheader("✅ Best Match Found")
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>{top_recommendation['name']}</h4>
                        <p><strong>Description:</strong> {top_recommendation['description']}</p>
                        <p><strong>Reason for Use:</strong> {top_recommendation['reason_cleaned']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                substitutes = get_substitutes(top_recommendation['name'], df)
                if substitutes:
                    st.subheader("💊 Available Brand Substitutes")
                    cols = st.columns(len(substitutes) if len(substitutes) <= 4 else 4)
                    for i, sub in enumerate(substitutes):
                        with cols[i % 4]:
                            st.info(sub)
                
                other_recommendations = recommended_medicines.iloc[1:]
                if not other_recommendations.empty:
                    with st.expander("🔍 See other similar medicines"):
                        for _, row in other_recommendations.iterrows():
                            st.markdown(f"""
                            <div class="result-card">
                                <h5>{row['name']}</h5>
                                <p><strong>Description:</strong> {row['description']}</p>
                                <p><strong>Reason for Use:</strong> {row['reason_cleaned']}</p>
                            </div>
                            """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
