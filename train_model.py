import pandas as pd
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load datasets
true_news = pd.read_csv('C:/Users/Divya/Downloads/True.csv')[['text']]
fake_news = pd.read_csv('C:/Users/Divya/Downloads/Fake.csv')[['text']]


# Combine datasets with labels
all_news = pd.concat([
    true_news.assign(label='true'),
    fake_news.assign(label='fake'),
])

# Preprocess text data
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

all_news['text'] = all_news['text'].apply(preprocess_text)

# Split dataset into training and testing sets
X = all_news['text']
y = all_news['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions and evaluate the model
y_pred = classifier.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model for later use
import pickle
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Load saved model and vectorizer
with open('fake_news_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# Function to classify news
def classify_news(news_text):
    news_text = preprocess_text(news_text)
    news_vectorized = loaded_vectorizer.transform([news_text])
    prediction = loaded_model.predict(news_vectorized)
    return prediction[0]

# Example usage
news_to_check = "Example news headline or text"
print(classify_news(news_to_check))