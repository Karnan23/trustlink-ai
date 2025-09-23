"""Utility functions for text preprocessing and heuristic analysis."""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """Clean and preprocess text for model input."""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_heuristic_analysis(text, url=""):
    """Analyze text for suspicious patterns using rule-based heuristics."""
    if not text:
        return {"score": 0.0, "reasons": []}
    
    text_lower = text.lower()
    reasons = []
    score_components = []
    
    # 1. Payment/Money requests
    payment_patterns = [
        r'send.*\$?\d+',
        r'pay.*\$?\d+',
        r'processing fee',
        r'registration fee',
        r'transfer.*money',
        r'wire.*transfer',
        r'send money',
        r'deposit.*fee'
    ]
    
    for pattern in payment_patterns:
        if re.search(pattern, text_lower):
            reasons.append("Requests payment/processing fees")
            score_components.append(0.4)
            break
    
    # 2. Suspicious contact methods
    contact_patterns = [
        r'whatsapp',
        r'telegram',
        r'call.*only',
        r'contact.*only',
        r'text.*only'
    ]
    
    for pattern in contact_patterns:
        if re.search(pattern, text_lower):
            reasons.append("Uses non-standard contact methods")
            score_components.append(0.3)
            break
    
    # 3. Unrealistic salary promises
    salary_patterns = [
        r'earn.*\$?\d{4,}.*month',
        r'make.*\$?\d{4,}.*week',
        r'\$?\d{5,}.*month',
        r'guaranteed.*income',
        r'easy money',
        r'massive income'
    ]
    
    for pattern in salary_patterns:
        if re.search(pattern, text_lower):
            reasons.append("Promises unrealistic earnings")
            score_components.append(0.3)
            break
    
    # 4. Urgency and pressure tactics
    urgency_patterns = [
        r'urgent',
        r'act now',
        r'limited time',
        r'hurry',
        r'immediate',
        r'start today'
    ]
    
    for pattern in urgency_patterns:
        if re.search(pattern, text_lower):
            reasons.append("Uses high-pressure tactics")
            score_components.append(0.2)
            break
    
    # 5. Vague job descriptions
    if len(text.split()) < 20:  # Very short postings
        if not re.search(r'apply.*@.*\.com', text_lower):  # No professional email
            reasons.append("Suspiciously vague job description")
            score_components.append(0.2)
    
    # 6. Suspicious domains (if URL provided)
    if url:
        url_lower = url.lower()
        suspicious_domains = ['.xyz', '.tk', '.ml', '.ga', '.cf']
        if any(domain in url_lower for domain in suspicious_domains):
            reasons.append("Suspicious website domain")
            score_components.append(0.3)
        
        # Very long domain names (often suspicious)
        domain_match = re.search(r'//([^/]+)', url_lower)
        if domain_match:
            domain = domain_match.group(1)
            if len(domain) > 30:
                reasons.append("Unusually long domain name")
                score_components.append(0.2)
    
    # Calculate final heuristic score (cap at 1.0)
    heuristic_score = min(sum(score_components), 1.0)
    
    return {
        "score": heuristic_score,
        "reasons": reasons[:3]  # Return top 3 reasons
    }

def get_model_explanation(text, model, vectorizer, top_n=5):
    """Get top contributing words from the model for explanation."""
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Transform text to TF-IDF
        text_tfidf = vectorizer.transform([processed_text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get model coefficients for the scam class (class 1)
        scam_coef = model.coef_[0]
        
        # Get TF-IDF values for this text
        tfidf_values = text_tfidf.toarray()[0]
        
        # Calculate contribution scores (coefficient * tfidf_value)
        contributions = []
        for i, (coef, tfidf_val) in enumerate(zip(scam_coef, tfidf_values)):
            if tfidf_val > 0:  # Only consider words present in the text
                contribution = coef * tfidf_val
                contributions.append((feature_names[i], contribution))
        
        # Sort by absolute contribution and return top N
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return contributions[:top_n]
    
    except Exception as e:
        print(f"Error in model explanation: {e}")
        return []

def calculate_final_score(model_prob, heuristic_score):
    """Calculate final risk score combining model probability and heuristics."""
    # Weighted combination: 60% model + 40% heuristics
    final_score = (0.6 * model_prob + 0.4 * heuristic_score) * 100
    return round(final_score)

def get_risk_label(score):
    """Convert numerical score to risk label."""
    if score <= 39:
        return "Safe ✅"
    elif score <= 69:
        return "Suspicious ⚠️"
    else:
        return "High-risk ❌"

def get_recommendation(score):
    """Get recommendation based on risk score."""
    if score <= 39:
        return "This posting appears legitimate. Proceed with normal caution."
    elif score <= 69:
        return "Exercise caution. Verify company details and avoid upfront payments."
    else:
        return "High scam probability. Strongly recommend avoiding this posting."