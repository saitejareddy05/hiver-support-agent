from __future__ import annotations
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def intent_metrics(gold: pd.DataFrame, predictions: pd.Series) -> dict:
    p, r, f, _ = precision_recall_fscore_support(gold.true_intent, predictions, average="macro", zero_division=0)
    return {"intent_accuracy": float(accuracy_score(gold.true_intent, predictions)), "intent_macro_precision": float(p), "intent_macro_recall": float(r), "intent_macro_f1": float(f)}
def escalation_metrics(gold: pd.DataFrame, predictions: pd.Series) -> dict:
    p, r, f, _ = precision_recall_fscore_support(gold.true_should_escalate.astype(bool), predictions.astype(bool), average="binary", zero_division=0)
    return {"escalation_precision": float(p), "escalation_recall": float(r), "escalation_f1": float(f)}
def retrieval_hit_rate(outputs: list[dict], threshold=.05) -> float:
    if not outputs: return 0.0
    return sum(bool(x.get("retrieved_examples")) and x["retrieved_examples"][0].get("similarity", 0) >= threshold for x in outputs) / len(outputs)
