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
    # Check if all necessary files exist
    if not all(os.path.exists(p) for p in [vectorizer_path, matrix_path, df_path]):
        print("\nFATAL ERROR: Model files not found.")
        print("Please run the 'train_model.py' script first to generate the necessary files.")
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
    Finds medicines similar to a given text query by calculating similarity on-the-fly.
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

def run_interactive_recommender(df, tfidf_vectorizer, tfidf_matrix):
    """
    Main interactive loop for the recommendation system.
    """
    print("You can now enter a medicine name, symptom (e.g., 'headache'), or description.")
    
    while True:
        print("\n======================================================")
        user_input = input("Enter your query (or type 'exit' to quit):\n> ")
        if user_input.lower() == 'exit':
            print("\nThank you for using the Recommender. Goodbye!")
            break
        
        recommended_medicines = get_recommendations_from_query(user_input, df, tfidf_vectorizer, tfidf_matrix)
        
        if recommended_medicines.empty:
            print("\nSorry, we couldn't find any matching medicines for your query.")
            continue
            
        top_recommendation = recommended_medicines.iloc[0]

        print("\n------------------------------------------------------")
        print("✅ Best Match Found For Your Query:")
        print("------------------------------------------------------")
        print(f"Medicine: {top_recommendation['name']}")
        print(f"  Description: {top_recommendation['description']}")
        print(f"  Reason for Use: {top_recommendation['reason_cleaned']}")
        
        substitutes = get_substitutes(top_recommendation['name'], df)
        if substitutes:
            print("\n💊 Available Brand Substitutes:")
            for i, sub in enumerate(substitutes, 1):
                print(f"  {i}. {sub}")
        else:
            print("\nNo brand substitutes listed for this medicine.")

        # Ask the user if they want to see other similar medicines
        other_recommendations = recommended_medicines.iloc[1:]
        if not other_recommendations.empty:
            show_alternatives = input("\nWould you like to see other similar medicines? (yes/no): ").lower()
            if show_alternatives in ['yes', 'y']:
                print("\n------------------------------------------------------")
                print("🔍 Other Similar Medicines You Might Consider:")
                print("------------------------------------------------------")
                for index, row in other_recommendations.iterrows():
                    print(f"\nMedicine: {row['name']}")
                    print(f"  Description: {row['description']}")
                    print(f"  Reason for Use: {row['reason_cleaned']}")

if __name__ == '__main__':
    VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
    MATRIX_FILE = 'tfidf_matrix.npz'
    DATAFRAME_FILE = 'processed_data.pkl'
    
    vectorizer, matrix, medicine_df = load_model_components(VECTORIZER_FILE, MATRIX_FILE, DATAFRAME_FILE)
    
    if medicine_df is not None:
        run_interactive_recommender(medicine_df, vectorizer, matrix)

