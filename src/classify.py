"""Intent prediction using a lightweight brand-specific nearest-neighbor model."""
from __future__ import annotations
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class IntentClassifier:
    def __init__(self, frame=None):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.fallback = "other"
        if frame is not None and len(frame): self.fit(frame)
    def fit(self, frame):
        labels = frame["intent"].fillna("other").astype(str) if "intent" in frame else frame.get("true_intent", "other")
        self.fallback = labels.mode().iloc[0]
        self.model.fit(self.vectorizer.fit_transform(frame["message_text"].fillna("")), labels)
        return self
    def predict(self, text: str) -> dict:
        if not hasattr(self.model, "classes_"): return {"intent": self.fallback, "confidence": 0.0}
        probs = self.model.predict_proba(self.vectorizer.transform([text]))[0]
        idx = int(np.argmax(probs))
        return {"intent": str(self.model.classes_[idx]), "confidence": float(probs[idx])}

def normalize_intent(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "other"
