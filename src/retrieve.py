"""TF-IDF retrieval over resolved historical replies."""
from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HistoricalRetriever:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True).copy()
        text = (self.frame["message_text"].fillna("") + " " + self.frame.get("thread_context", "").fillna(""))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(text)
    def search(self, message: str, top_k: int = 3) -> list[dict]:
        scores = cosine_similarity(self.vectorizer.transform([message]), self.matrix)[0]
        order = scores.argsort()[::-1][:top_k]
        return [{**self.frame.iloc[i].to_dict(), "similarity": float(scores[i])} for i in order]
