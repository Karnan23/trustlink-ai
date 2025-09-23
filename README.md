# TrustLink AI - Job Scam Detection Prototype

🎯 **Elevator Pitch**: AI-powered tool that combines machine learning with rule-based heuristics to identify suspicious job postings and protect job seekers from scams.

## Quick Start

1. **Setup Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download NLTK Data**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

3. **Train the Model**
```bash
python src/train.py
```

4. **Run the App**
```bash
streamlit run src/app.py
```

5. **Evaluate Model**
```bash
python src/sample_eval.py
```

## How It Works

- **ML Component**: Logistic regression on TF-IDF features (4000 features, 1-2 grams)
- **Heuristics**: Rule-based detection for payment requests, suspicious contacts, salary anomalies
- **Scoring**: Combined score (60% ML + 40% heuristics) with interpretable risk levels
- **Explainability**: Shows top contributing words and triggered heuristic rules

## Risk Levels
- **0-39**: Safe ✅
- **40-69**: Suspicious ⚠️  
- **70-100**: High-risk ❌

## Repository Structure
- `data/`: Synthetic training data (300 balanced samples)
- `src/`: Core application code
- `tests/`: Unit tests for utility functions

## Next Steps
1. Integrate real job board APIs
2. Add more sophisticated NLP features
3. Implement user feedback loop
4. Deploy as web service