import os
import re
import pickle
import numpy as np
from typing import Dict, Any

class IntentClassifier:
    def __init__(self, model_path: str = None):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "intent_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Core regex rules for high-confidence immediate matching
        self.exact_rules = {
            "IDENTIFY_PERSON": [
                r"\bwho is\b", r"\bwho are you\b", r"\bwho was\b", 
                r"\bdo i know\b", r"\bidentify\b", r"\bwho am i looking at\b"
            ],
            "GET_MEMORY": [
                r"\btell me about\b", r"\bremind me\b", r"\bwhat do you remember\b",
                r"\bmemories about\b", r"\bgive me info\b", r"\bdetails about\b"
            ],
            "CHECK_MEDICATION": [
                r"\bmedicin(e|es)\b", r"\bpill(s)?\b", r"\bmedication\b", 
                r"\bdose\b", r"\btablet(s)?\b", r"\binsulin\b"
            ],
            "HELP": [
                r"\bhelp\b", r"\bassist\b", r"\bemergency\b", r"\bfell down\b"
            ]
        }

    def classify(self, text: str) -> Dict[str, Any]:
        cleaned_text = text.lower().strip()
        if not cleaned_text:
            return {
                "text": text,
                "intent": "UNKNOWN",
                "confidence": 0.0
            }

        # 1. Fast Rule Matching for guaranteed project scenarios
        for intent, patterns in self.exact_rules.items():
            for pattern in patterns:
                if re.search(pattern, cleaned_text):
                    return {
                        "text": text,
                        "intent": intent,
                        "confidence": 0.95
                    }

        # 2. Fallback to Trained ML Model
        probabilities = self.model.predict_proba([cleaned_text])[0]
        max_index = np.argmax(probabilities)
        confidence = float(probabilities[max_index])
        predicted_intent = self.model.classes_[max_index]

        if confidence < 0.35:
            predicted_intent = "UNKNOWN"

        return {
            "text": text,
            "intent": predicted_intent,
            "confidence": round(confidence, 2)
        }


if __name__ == "__main__":
    classifier = IntentClassifier()
    test_queries = [
        "Who is this person standing in front of me?",
        "Tell me about him",
        "Did I take my morning pills today?",
        "Please help me I need assistance"
    ]

    for q in test_queries:
        print(classifier.classify(q))