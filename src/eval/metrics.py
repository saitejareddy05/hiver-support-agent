"""
Automated metrics computed against the golden evaluation set.

Deliberately kept separate from llm_judge.py: these are cheap, deterministic, and
don't require any API key, so they always run as part of `python -m src.eval.run_eval`.
"""
from __future__ import annotations
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def intent_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
    }


def escalation_metrics(y_true_escalate: list[bool], y_pred_escalate: list[bool]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true_escalate, y_pred_escalate), 4),
        "precision_escalate": round(precision_score(y_true_escalate, y_pred_escalate, zero_division=0), 4),
        "recall_escalate": round(recall_score(y_true_escalate, y_pred_escalate, zero_division=0), 4),
        "f1_escalate": round(f1_score(y_true_escalate, y_pred_escalate, zero_division=0), 4),
        # "unsafe auto-handle rate": of the cases that SHOULD have been escalated,
        # what fraction did we wrongly auto-handle? This is the number that matters
        # most for trust -- see reports/REPORT.md "what's misleading about the
        # headline number".
        "unsafe_auto_handle_rate": _unsafe_auto_handle_rate(y_true_escalate, y_pred_escalate),
    }


def _unsafe_auto_handle_rate(y_true_escalate, y_pred_escalate) -> float:
    should_escalate = [i for i, v in enumerate(y_true_escalate) if v]
    if not should_escalate:
        return 0.0
    wrongly_auto_handled = sum(1 for i in should_escalate if not y_pred_escalate[i])
    return round(wrongly_auto_handled / len(should_escalate), 4)
