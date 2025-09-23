"""Training script for TrustLink AI job scam detection model."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import nltk
from model_utils import preprocess_text

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_and_prepare_data(data_path='data/synthetic_jobs.csv'):
    """Load and preprocess the job postings data."""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Convert labels to binary (scam=1, legitimate=0)
    df['label'] = df['label'].map({'scam': 1, 'legitimate': 0})
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    print(f"Data loaded: {len(df)} samples")
    print(f"Distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def train_model(df):
    """Train the TF-IDF vectorizer and logistic regression model."""
    X = df['processed_text']
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Create TF-IDF vectorizer
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Fit vectorizer and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF features created: {X_train_tfidf.shape[1]} features")
    
    # Train logistic regression model
    print("Training logistic regression model...")
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    print("\n=== Model Evaluation ===")
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Scam']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, vectorizer, X_test, y_test

def save_model_artifacts(model, vectorizer):
    """Save trained model and vectorizer to disk."""
    print("\nSaving model artifacts...")
    
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    
    print("Model saved to models/model.pkl")
    print("Vectorizer saved to models/vectorizer.pkl")

def main():
    """Main training pipeline."""
    print("=== TrustLink AI Training Pipeline ===")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Train model
    model, vectorizer, X_test, y_test = train_model(df)
    
    # Save artifacts
    save_model_artifacts(model, vectorizer)
    
    print("\n=== Training Complete ===")
    print("Run 'streamlit run src/app.py' to test the model!")

if __name__ == "__main__":
    main()