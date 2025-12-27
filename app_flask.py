from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load the combined pipeline
try:
    pipeline = joblib.load('expense_categorization_pipeline.joblib')
    loaded_model = pipeline['model']
    loaded_tfidf_vectorizer = pipeline['tfidf_vectorizer']
    loaded_label_encoder = pipeline['label_encoder']
except FileNotFoundError:
    print("Error: 'expense_categorization_pipeline.joblib' not found. Make sure to run the model export step.")
    exit()

# Ensure NLTK data is downloaded for preprocessing (if not already present in environment)
try:
    stopwords.words('english')
    WordNetLemmatizer()
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Re-define preprocessing functions from previous steps
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers, keep only letters and spaces
    return text

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    tokens = text.split()  # Simple tokenization by splitting on space
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

@app.route('/')
def home():
    return "Expense Categorization Flask API. Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    description = data.get('description', '')

    if not description:
        return jsonify({'error': 'No description provided'}), 400

    # Preprocess the input text
    cleaned_desc = clean_text(description.lower().strip())
    processed_desc = preprocess_text(cleaned_desc)

    # Transform the text using the loaded TF-IDF vectorizer
    text_vector = loaded_tfidf_vectorizer.transform([processed_desc])

    # Predict the numerical category
    numerical_category = loaded_model.predict(text_vector)

    # Decode the numerical category back to the original category name
    category_name = loaded_label_encoder.inverse_transform(numerical_category)

    return jsonify({'predicted_category': category_name[0]})

if __name__ == '__main__':
    app.run()
