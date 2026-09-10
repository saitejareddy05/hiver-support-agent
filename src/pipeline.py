"""End-to-end support agent."""
from __future__ import annotations
import os
import yaml
import pandas as pd
from .classify import IntentClassifier
from .retrieve import HistoricalRetriever
from .escalation_policy import decide
from .draft_reply import draft_reply

def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as stream: return yaml.safe_load(stream)
class SupportAgent:
    def __init__(self, frame: pd.DataFrame, config=None, llm_client=None):
        self.config = config or {}; self.frame = frame
        self.classifier = IntentClassifier(frame); self.retriever = HistoricalRetriever(frame); self.llm_client = llm_client
    def run(self, message: str) -> dict:
        prediction = self.classifier.predict(message)
        examples = self.retriever.search(message, self.config.get("retrieval_top_k", 3))
        decision = decide(message, prediction["confidence"], self.config.get("escalation_confidence_threshold", .6))
        reply = draft_reply(message, prediction["intent"], examples, self.config.get("brand", "AmazonHelp"), self.config.get("max_reply_chars", 500), self.llm_client, self.config.get("model_name", "gpt-4o-mini"))
        return {"message": message, **prediction, "retrieved_examples": examples, "reply": reply, **decision}
