import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import pickle
from scipy.sparse import save_npz

def train_and_save_model(data_filepath, vectorizer_path, matrix_path, df_path):
    """
    Loads data, trains the TF-IDF model, and saves the necessary components to disk.
    """
    print("--- Starting Model Training ---")
    
    # 1. Load and Preprocess Data
    print("Step 1/4: Loading and preprocessing data...")
    try:
        df = pd.read_csv(data_filepath)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The data file '{data_filepath}' was not found.")
        return

    def clean_reason(reason_str):
        try:
            return ' '.join(ast.literal_eval(reason_str))
        except (ValueError, SyntaxError):
            return ""

    df['reason_cleaned'] = df['reason'].apply(clean_reason)
    df['soup'] = df['name'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['reason_cleaned']
    
    # Fill NaN values in substitute columns
    sub_cols = ['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']
    for col in sub_cols:
        df[col] = df[col].fillna('')
    print("Data loaded successfully.")

    # 2. Train TF-IDF Vectorizer
    print("Step 2/4: Training the NLP model (TfidfVectorizer)...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    print("NLP model trained.")

    # 3. Save the trained vectorizer
    print(f"Step 3/4: Saving the vectorizer to {vectorizer_path}...")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf, f)
    print("Vectorizer saved.")

    # 4. Save the TF-IDF matrix and the processed DataFrame
    print(f"Step 4/4: Saving the TF-IDF matrix to {matrix_path} and data to {df_path}...")
    save_npz(matrix_path, tfidf_matrix)
    df.to_pickle(df_path)
    print("Matrix and data saved.")
    
    print("\n--- Training Complete! ---")
    print("You can now run 'recommender.py'.")


if __name__ == '__main__':
    DATA_FILE = '../Datasets/merged_medicines_updated.csv'
    VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
    MATRIX_FILE = 'tfidf_matrix.npz'
    DATAFRAME_FILE = 'processed_data.pkl'
    
    train_and_save_model(DATA_FILE, VECTORIZER_FILE, MATRIX_FILE, DATAFRAME_FILE)


