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
        print("\nFATAL ERROR: Model files not found.")
        print("Please run the 'train.py' script first to generate the necessary files.")
        return None, None, None
        
    print("Loading pre-trained model components...")
    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
        
    tfidf_matrix = load_npz(matrix_path)
    df = pd.read_pickle(df_path)
    
    print("Recommender is ready.\n")
    return tfidf_vectorizer, tfidf_matrix, df

def get_recommendations_from_query(query, df, tfidf_vectorizer, tfidf_matrix):
    """
    Finds medicines similar to a given text query.
    """
    query_tfidf = tfidf_vectorizer.transform([query])
    sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-5:][::-1]
    return df.iloc[top_indices]

def get_substitutes(medicine_name, df):
    """
    Retrieves the list of substitute medicines for a given medicine name.
    """
    try:
        medicine_row = df[df['name'] == medicine_name].iloc[0]
        sub_cols = ['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']
        substitutes = [sub for sub in medicine_row[sub_cols] if sub]
        return substitutes
    except IndexError:
        return []