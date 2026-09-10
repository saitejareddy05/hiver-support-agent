"""Reproducible majority and TF-IDF baselines."""
from __future__ import annotations
from .classify import IntentClassifier
class MajorityBaseline:
    def __init__(self, frame): self.intent = frame["intent"].mode().iloc[0] if len(frame) else "other"
    def predict(self, text): return {"intent": self.intent, "confidence": 1.0}
class TfidfBaseline(IntentClassifier):
    def reply(self, text): return "Thanks for contacting us. Please check your order details and contact support if the issue continues."
