from flask import Flask, render_template, request
from recommender import load_model_components, get_recommendations_from_query, get_substitutes

app = Flask(__name__)

# Load the model components when the application starts
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
MATRIX_FILE = 'tfidf_matrix.npz'
DATAFRAME_FILE = 'processed_data.pkl'

vectorizer, matrix, df = load_model_components(VECTORIZER_FILE, MATRIX_FILE, DATAFRAME_FILE)

@app.route('/')
def home():
    """Renders the new homepage."""
    return render_template('home.html')

@app.route('/recommender', methods=['GET', 'POST'])
def recommender_page():
    """Handles the recommendation logic and page."""
    if request.method == 'POST':
        user_query = request.form['query']
        
        if df is not None:
            recommended_medicines = get_recommendations_from_query(user_query, df, vectorizer, matrix)
            
            if not recommended_medicines.empty:
                top_recommendation = recommended_medicines.iloc[0]
                substitutes = get_substitutes(top_recommendation['name'], df)
                other_recommendations = recommended_medicines.iloc[1:]
                
                return render_template('index.html', 
                                       recommendation=top_recommendation, 
                                       substitutes=substitutes,
                                       other_recommendations=other_recommendations,
                                       query=user_query)
                                       
        return render_template('index.html', error="Sorry, we couldn't find any matching medicines for your query.", query=user_query)
    
    # FIX: For GET requests, pass None to the template for the expected variables
    return render_template('index.html', recommendation=None, error=None, query=None)

@app.route('/login')
def login_page():
    """Renders the login/signup page."""
    return render_template('login.html')

@app.route('/medicines')
def medicines_page():
    """Renders the medicine listing page."""
    # Placeholder for the top 20 medicines
    medicines = [
        "Medicine A", "Medicine B", "Medicine C", "Medicine D", "Medicine E",
        "Medicine F", "Medicine G", "Medicine H", "Medicine I", "Medicine J",
        "Medicine K", "Medicine L", "Medicine M", "Medicine N", "Medicine O",
        "Medicine P", "Medicine Q", "Medicine R", "Medicine S", "Medicine T"
    ]
    return render_template('medicines.html', medicines=medicines)

@app.route('/contact')
def contact_page():
    """Renders the contact page."""
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)