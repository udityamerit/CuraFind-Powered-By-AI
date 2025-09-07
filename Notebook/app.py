from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from recommender import load_model_components, get_recommendations_from_query, get_substitutes
import pandas as pd

# --- App and Login Configuration ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_change_this'  # Important for sessions

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'  # Redirect to login page if user is not authenticated
login_manager.login_message_category = 'info' # Flash message category

# --- User Model and Database (In-memory for demonstration) ---
# In a real application, you would use a database (e.g., SQLAlchemy)
users = {}

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# --- Load Model Components ---
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
MATRIX_FILE = 'tfidf_matrix.npz'
DATAFRAME_FILE = 'processed_data.pkl'

vectorizer, matrix, df = load_model_components(VECTORIZER_FILE, MATRIX_FILE, DATAFRAME_FILE)

# --- Route Definitions ---

@app.route('/')
def home():
    """Renders the homepage. Accessible to everyone."""
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Handles user login and registration."""
    if current_user.is_authenticated:
        return redirect(url_for('recommender_page'))

    if request.method == 'POST':
        # --- Registration Logic ---
        if 'signup_submit' in request.form:
            username = request.form['username']
            password = request.form['password']
            if username in [u.username for u in users.values()]:
                flash('Username already exists.', 'danger')
            else:
                new_id = str(len(users) + 1)
                new_user = User(new_id, username, password)
                users[new_id] = new_user
                login_user(new_user)
                flash('Account created successfully!', 'success')
                return redirect(url_for('recommender_page'))
        
        # --- Login Logic ---
        elif 'login_submit' in request.form:
            username = request.form['username_login']
            password = request.form['password_login']
            user = next((u for u in users.values() if u.username == username), None)
            
            if user and user.password == password:
                login_user(user)
                next_page = request.args.get('next')
                return redirect(next_page or url_for('recommender_page'))
            else:
                flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/recommender', methods=['GET', 'POST'])
@login_required
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
    
    return render_template('index.html', recommendation=None, error=None, query=None)


@app.route('/medicines')
@login_required
def medicines_page():
    """Renders the medicine listing page with 20 random medicines."""
    if df is not None and not df.empty:
        # Get 20 random medicines if the dataset has enough samples
        sample_size = min(20, len(df))
        medicines = df.sample(n=sample_size).to_dict('records')
    else:
        medicines = [] # Handle case where data isn't loaded
        
    return render_template('medicines.html', medicines=medicines)


@app.route('/contact')
@login_required
def contact_page():
    """Renders the contact page."""
    return render_template('contact.html')

if __name__ == '__main__':
    # Create a dummy user for testing
    users['1'] = User('1', 'testuser', 'password123')
    app.run(debug=True)