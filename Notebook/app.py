from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from recommender import load_model_components, get_recommendations, get_substitutes
import pandas as pd
import json
import os

# --- App and Login Configuration ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_change_this'  # Important for sessions

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'
login_manager.login_message_category = 'info'

# --- User Model and Persistent Storage ---
USERS_FILE = 'users.json'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

def load_users():
    """Loads users from a JSON file."""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        users_data = json.load(f)
        # Recreate User objects from the loaded data
        return {id: User(id, data['username'], data['password']) for id, data in users_data.items()}

def save_users(users_dict):
    """Saves users to a JSON file."""
    # Convert User objects to a serializable dictionary
    users_data = {id: {'username': user.username, 'password': user.password} for id, user in users_dict.items()}
    with open(USERS_FILE, 'w') as f:
        json.dump(users_data, f)

# Load users at startup
users = load_users()


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
    """Renders the homepage."""
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
                save_users(users)  # Save the updated user list to the file
                login_user(new_user)
                flash('Account created successfully! You are now logged in.', 'success')
                return redirect(url_for('recommender_page')) # Redirect to recommender after signup
        
        # --- Login Logic ---
        elif 'login_submit' in request.form:
            username = request.form['username_login']
            password = request.form['password_login']
            
            user = next((u for u in users.values() if u.username == username), None)
            
            if user and user.password == password:
                login_user(user)
                # Redirect to the home page after a successful login
                return redirect(url_for('home'))
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
    if request.method == 'POST':
        user_query = request.form.get('query')
        if df is not None:
            recommended_medicines = get_recommendations(user_query, df, vectorizer, matrix)
            if not recommended_medicines.empty:
                top_recommendation = recommended_medicines.iloc[0].to_dict()
                substitutes = get_substitutes(top_recommendation['name'], df)
                other_recommendations = recommended_medicines.iloc[1:]
                return render_template('index.html', recommendation=top_recommendation, substitutes=substitutes, other_recommendations=other_recommendations, query=user_query)
        return render_template('index.html', error="Sorry, we couldn't find any matching medicines.", query=user_query)
    return render_template('index.html', recommendation=None, error=None, query=None)


@app.route('/medicines')
@login_required
def medicines_page():
    """Renders the medicine listing page with 20 random medicines."""
    if df is not None and not df.empty:
        sample_size = min(20, len(df))
        medicines = df.sample(n=sample_size).to_dict('records')
    else:
        medicines = []
    return render_template('medicines.html', medicines=medicines)

@app.route('/contact')
@login_required
def contact_page():
    """Renders the contact page."""
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)