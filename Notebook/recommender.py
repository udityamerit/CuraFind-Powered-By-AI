import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy.sparse import load_npz
import os

def load_model_components(vectorizer_path, matrix_path, df_path):
    """
    Loads the pre-trained model components from disk.
    """
    print("--- Initializing Recommender ---")
    if not all(os.path.exists(p) for p in [vectorizer_path, matrix_path, df_path]):
        print("\nFATAL ERROR: Model components not found.")
        print("Please run 'train_model.py' first to train the model and create the necessary files.")
        return None, None, None

    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        matrix = load_npz(matrix_path)
        df = pd.read_pickle(df_path)
        print("Recommender initialized successfully.")
        return vectorizer, matrix, df
    except Exception as e:
        print(f"\nERROR loading model components: {e}")
        return None, None, None

def get_recommendations(query, df, vectorizer, matrix):
    """
    Gets medicine recommendations based on a query.
    First, it tries a direct, case-insensitive search.
    If that fails, it uses the TF-IDF model to find similar medicines.
    """
    # Direct search (case-insensitive)
    direct_match = df[df['name'].str.lower() == query.lower()]
    if not direct_match.empty:
        return direct_match

    # Cosine similarity search
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 results
    medicine_indices = [i[0] for i in sim_scores]
    return df.iloc[medicine_indices]

def get_substitutes(medicine_name, df):
    """
    Gets substitutes for a given medicine.
    """
    substitutes = df[df['name'] == medicine_name][['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']]
    return substitutes.values.flatten().tolist()

if __name__ == '__main__':
    # --- Load Model Components ---
    VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
    MATRIX_FILE = 'tfidf_matrix.npz'
    DATAFRAME_FILE = 'processed_data.pkl'

    vectorizer, matrix, df = load_model_components(VECTORIZER_FILE, MATRIX_FILE, DATAFRAME_FILE)

    if df is not None:
        # --- Get Recommendations ---
        query = "Paracetamol"
        recommendations = get_recommendations(query, df, vectorizer, matrix)
        print(f"\nRecommendations for '{query}':")
        print(recommendations[['name', 'description', 'reason']])

        # --- Get Substitutes ---
        if not recommendations.empty:
            medicine_name = recommendations.iloc[0]['name']
            substitutes = get_substitutes(medicine_name, df)
            print(f"\nSubstitutes for '{medicine_name}':")
            print(substitutes)