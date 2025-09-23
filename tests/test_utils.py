"""Unit tests for model utility functions."""

import unittest
import sys
import os

# Add src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_utils import preprocess_text, get_heuristic_analysis, calculate_final_score

class TestModelUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing functionality."""
        # Test basic cleaning
        text = "Software Engineer position at TechCorp. Apply to jobs@tech.com"
        processed = preprocess_text(text)
        
        self.assertEqual(processed, "software engineer position at techcorp. apply to")
        self.assertNotIn("@", processed)  # Email should be removed
    
    def test_heuristic_analysis_payment_detection(self):
        """Test heuristic detection of payment requests."""
        scam_text = "Great opportunity! Just send $500 processing fee via WhatsApp"
        result = get_heuristic_analysis(scam_text)
        
        # Should detect payment request and WhatsApp
        self.assertGreater(result["score"], 0.5)
        self.assertGreater(len(result["reasons"]), 0)
        self.assertTrue(any("payment" in reason.lower() for reason in result["reasons"]))
    
    def test_calculate_final_score_combination(self):
        """Test final score calculation combining model and heuristics."""
        model_prob = 0.8  # High model probability
        heuristic_score = 0.6  # Moderate heuristic score
        
        # Should be weighted combination: 60% model + 40% heuristics
        expected = round((0.6 * 0.8 + 0.4 * 0.6) * 100)  # = 72
        actual = calculate_final_score(model_prob, heuristic_score)
        
        self.assertEqual(actual, expected)
        self.assertGreaterEqual(actual, 0)
        self.assertLessEqual(actual, 100)

if __name__ == "__main__":
    unittest.main()