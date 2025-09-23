"""Evaluation script for TrustLink AI model on holdout data."""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from model_utils import preprocess_text

def load_model_artifacts():
    """Load trained model and vectorizer."""
    try:
        model = joblib.load('models/model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'python src/train.py' first.")
        return None, None

def evaluate_model():
    """Evaluate the model on holdout test data."""
    print("=== TrustLink AI Model Evaluation ===")
    
    # Load model artifacts
    model, vectorizer = load_model_artifacts()
    if model is None or vectorizer is None:
        return
    
    # Load and prepare data (same split as training)
    print("Loading evaluation data...")
    df = pd.read_csv('data/synthetic_jobs.csv')
    df['label'] = df['label'].map({'scam': 1, 'legitimate': 0})
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Use same train-test split as training
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n=== Model Performance Metrics ===")
    print(f"Test Set Size: {len(y_test)} samples")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    
    print(f"\n=== Detailed Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Scam']))
    
    # Show some example predictions
    print(f"\n=== Sample Predictions ===")
    test_df = pd.DataFrame({
        'text': X_test.iloc[:5],
        'true_label': y_test.iloc[:5],
        'predicted_label': y_pred[:5],
        'scam_probability': y_pred_proba[:5]
    })
    
    for idx, row in test_df.iterrows():
        label_map = {0: 'Legitimate', 1: 'Scam'}
        true_label = label_map[row['true_label']]
        pred_label = label_map[row['predicted_label']]
        prob = row['scam_probability']
        
        print(f"\nText: {row['text'][:100]}...")
        print(f"True: {true_label} | Predicted: {pred_label} | Scam Prob: {prob:.3f}")

if __name__ == "__main__":
    evaluate_model()