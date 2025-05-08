from flask import Flask, request, render_template , jsonify, send_from_directory
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

nltk.download('stopwords')

# Load trained model and vectorizer
model = joblib.load('phishing_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

app = Flask(__name__)

# Add this route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static'), filename)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/PMD')
def PMD():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    sender = data.get('sender', '')
    subject = data.get('subject', '')
    body = data.get('body', '')
    
    # Combined text for analysis
    email_text = f"{sender} {subject} {body}".lower()

    # email_text = request.form['email']
    cleaned = preprocess(email_text)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]
    # label = "Phishing Email" if result == 1 else "Safe Email"

    
    response = {
        'prediction': "Phishing Email" if result == 1 else "Safe Email"
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
