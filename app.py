from flask import Flask, request, render_template
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    cleaned = preprocess(email_text)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]
    label = "Phishing Email" if result == 1 else "Safe Email"

    # Send prediction and email content back to the template
    return render_template('index.html', prediction=label, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)
