"""
Intent taxonomy + classifier for the @AskPlayStation support agent.

Taxonomy (defined by reading ~300 real-style threads for this brand and grouping
recurring issues — see reports/REPORT.md "Problem framing" for the reasoning):

  account_login_access          - can't sign in, 2FA, locked/compromised account
  refund_billing                - refunds, duplicate/incorrect charges, subscriptions
  purchase_missing_content      - paid for something, entitlement not showing up
  technical_bug_error           - crashes, freezes, error codes, in-game bugs
  network_connectivity          - PSN outages, disconnects, NAT/port issues
  hardware_malfunction          - controller drift, console won't turn on, disc drive, overheating
  how_to_general_inquiry        - "how do I..." questions, not a problem report
  complaint_repeat_escalation   - explicit frustration / repeat contact / demands a human

We deliberately do NOT build a 77-way Banking77-style taxonomy: for a single brand's
support inbox, 8 intents already separate the routing/reply-strategy space cleanly,
and a finer taxonomy would mostly split hairs within categories that get the same
downstream handling (see decision log, item 3).

Classifier: TF-IDF (word 1-2 grams, char 3-5 grams) + Logistic Regression, one-vs-rest.
Chosen over calling an LLM for classification because:
  (a) it's free, deterministic, and reproducible without any API key — required for the
      "reproduce in <15 minutes" constraint,
  (b) 8-way intent classification on domain-specific short text is exactly the regime
      where a small supervised linear model is competitive with zero-shot LLM prompting,
      while being ~1000x cheaper and easier to audit,
  (c) it gives calibrated-enough confidence scores (via predict_proba) that we use
      directly in the escalation decision (see src/escalation.py).
An LLM classifier is still available as a drop-in via --classifier llm (see pipeline.py)
if ANTHROPIC_API_KEY is set, for comparison purposes in the eval harness.
"""
from __future__ import annotations
import re
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

INTENTS = [
    "account_login_access",
    "refund_billing",
    "purchase_missing_content",
    "technical_bug_error",
    "network_connectivity",
    "hardware_malfunction",
    "how_to_general_inquiry",
    "complaint_repeat_escalation",
]

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "intent_classifier.joblib"


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"@\w+", " ", text)           # strip @mentions (brand + user handles)
    text = re.sub(r"http\S+", " ", text)          # strip urls
    text = re.sub(r"#(\w+)", r"\1", text)          # keep hashtag words, drop the #
    text = re.sub(r"\s+", " ", text).strip()
    return text


class _TextSelector(BaseEstimator, TransformerMixin):
    """Applies clean_text before vectorizing. Lets us keep raw text in the DataFrame."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(x) for x in X]


def build_pipeline() -> Pipeline:
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    return Pipeline([
        ("clean", _TextSelector()),
        ("features", features),
        ("clf", clf),
    ])


class IntentClassifier:
    """Thin wrapper around the sklearn pipeline with save/load and a predict_one API
    that returns (intent, confidence, full distribution)."""

    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline

    def fit(self, texts, labels):
        self.pipeline = build_pipeline()
        self.pipeline.fit(texts, labels)
        return self

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "IntentClassifier":
        pipeline = joblib.load(path)
        return cls(pipeline)

    def predict_one(self, text: str):
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        top_idx = int(np.argmax(proba))
        dist = {c: float(p) for c, p in zip(classes, proba)}
        return classes[top_idx], float(proba[top_idx]), dist

    def predict_many(self, texts):
        return [self.predict_one(t) for t in texts]
