import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from datetime import datetime

# Download required NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

# Load fake and true news data
fake_news = pd.read_csv('C:/Users/Divya/Downloads/Fake.csv')
true_news = pd.read_csv('C:/Users/Divya/Downloads/True.csv')

# Extract headlines (assuming 'text' column contains headlines)
fake_headlines = fake_news['text'].tolist()
true_headlines = true_news['text'].tolist()

# Load saved model and vectorizer
with open('fake_news_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# Initialize history list
history = []

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Function to classify news
def classify_news(news_text):
    news_text = preprocess_text(news_text)
    news_vectorized = loaded_vectorizer.transform([news_text])
    prediction = loaded_model.predict(news_vectorized)
    return prediction[0]

# Example data for output page
accuracy = "0.93184855233853"
classification_report = """
               precision    recall  f1-score   support

        fake       0.93      0.94      0.93      4650
        true       0.93      0.93      0.93      4330

    accuracy                           0.93      8980
   macro avg       0.93      0.93      0.93      8980
weighted avg       0.93      0.93      0.93      8980
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        news_text = request.form.get('news_text')
        prediction = classify_news(news_text)
        # Store the classification result in history
        history.append({
            'news_text': news_text,
            'prediction': prediction,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        return render_template('index.html', prediction=prediction, history=history)
    return render_template('index.html', history=history)

@app.route('/output')
def output():
    return render_template('output.html', accuracy=accuracy, classification_report=classification_report)

@app.route('/fake')
def fake():
    return render_template('fake.html', headlines=fake_headlines)

@app.route('/true')
def true():
    return render_template('true.html', headlines=true_headlines)

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Removed the /classify route as it's redundant with the POST method in the index route

if __name__ == '__main__':
    app.run(debug=True)
