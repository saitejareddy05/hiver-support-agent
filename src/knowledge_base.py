"""
Knowledge base: TF-IDF retrieval over historical (customer_text -> agent_text) pairs.

This is the "grounding" mechanism required by the assignment: rather than letting a
generator invent a resolution from scratch, we retrieve the most similar past
customer messages this brand has actually handled and ground the drafted reply in
those real resolutions (see src/reply_generator.py).

Design choice: TF-IDF + cosine similarity rather than a dense embedding model.
Reasons (see decision log item 6):
  - No API key or model download required -> keeps the 15-minute reproduction promise.
  - On short, template-heavy, jargon-specific support text (error codes, "refund",
    "controller drift", etc.) sparse lexical retrieval is a strong, well-understood
    baseline and is easy to sanity check by eye.
  - The retrieval index is small (hundreds-to-low-thousands of pairs per brand), so
    approximate/dense retrieval infrastructure isn't needed for latency either.
A dense-embedding retriever is a natural upgrade path, called out explicitly in the
report's "What I'd do next" section.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.intents import clean_text

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "artifacts" / "knowledge_base.joblib"


@dataclass
class KBMatch:
    customer_text: str
    agent_text: str
    resolved: bool
    similarity: float


class KnowledgeBase:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.pairs = None  # DataFrame with customer_text/agent_text/thread_resolved

    def fit(self, pairs_df):
        self.pairs = pairs_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        cleaned = [clean_text(t) for t in self.pairs["customer_text"]]
        self.matrix = self.vectorizer.fit_transform(cleaned)
        return self

    def save(self, path: Path = KB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "pairs": self.pairs}, path)

    @classmethod
    def load(cls, path: Path = KB_PATH) -> "KnowledgeBase":
        data = joblib.load(path)
        kb = cls()
        kb.vectorizer = data["vectorizer"]
        kb.matrix = data["matrix"]
        kb.pairs = data["pairs"]
        return kb

    def retrieve(self, text: str, k: int = 3, min_similarity: float = 0.05) -> list[KBMatch]:
        q = self.vectorizer.transform([clean_text(text)])
        sims = cosine_similarity(q, self.matrix)[0]
        top_idx = np.argsort(-sims)[:k]
        matches = []
        for i in top_idx:
            if sims[i] < min_similarity:
                continue
            row = self.pairs.iloc[i]
            matches.append(KBMatch(
                customer_text=row["customer_text"],
                agent_text=row["agent_text"],
                resolved=bool(row["thread_resolved"]),
                similarity=float(sims[i]),
            ))
        return matches
