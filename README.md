# CuraFind AI - Intelligent Medicine Recommender

CuraFind AI is a web-based application leveraging **Natural Language Processing (NLP)** to intelligently recommend medicines. Users can search using symptoms, medicine names, or free-text descriptions, and receive suggestions along with brand substitutes for drugs.[1]

***

## ✨ Core Features

- **Symptom-Based Search**: Describe symptoms naturally (e.g., "runny nose and sore throat") to receive smart medicine suggestions.
- **Medicine Name Search**: Find detailed info and brand alternatives for specific medicines (e.g., "Crocin").
- **Intelligent Matching**: Utilizes TF-IDF Vectorization and Cosine Similarity to provide the most contextually relevant matches from a curated medicine dataset.
- **Brand Substitutes**: Instantly see a list of brands for any medicine queried.
- **Interactive UI**: Modern, fully responsive interface with animated particle backgrounds for enhanced user engagement.

***

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **UI Animation**: particles.js

***

## 🚀 Getting Started

### Prerequisites

- Python 3.7+  
- pip

### Installation

```bash
git clone https://github.com/udityamerit/CuraFind-Powered-By-AI.git
cd CuraFind-Powered-By-AI
```

Create and activate a virtual environment:
- **Windows**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS/Linux**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

Install requirements:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```
Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

***

## 📂 Project Structure

```
.
├── app.py                  # Main Flask application
├── recommender.py          # Recommender class and logic
├── train_model.py          # Data pre-processing/model training (optional)
├── requirements.txt        # Python dependencies
├── processed_data.pkl      # Pickled DataFrame (medicine data)
├── tfidf_vectorizer.pkl    # Pickled TF-IDF Vectorizer
├── tfidf_matrix.npz        # TF-IDF matrix file
├── static/
│   ├── style.css
│   └── (particles.json, script.js)  # Embedded config
└── templates/
    ├── home.html           # Landing page
    └── index.html          # Recommender main page
```

***

## 📄 File Descriptions

- **app.py**: Flask app routes (`/` and `/recommender`)
- **recommender.py**: Loads models, generates recommendations
- **train_model.py**: Utility for re-training (optional)
- **requirements.txt**: Lists dependencies
- **\*.pkl, \*.npz**: Trained machine learning models/data

***

## 🤝 Contributing

- Submit issues or bugs
- Open pull requests for improvements or new features

***
