"""Conservative human-escalation rules."""
from __future__ import annotations
import re
RISK_RULES = {
    "refund_or_financial": r"refund|chargeback|charged|money back|payment|billing|fraud",
    "legal_or_safety": r"lawyer|legal|court|police|unsafe|danger|injur|threat",
    "abuse_or_privacy": r"abuse|harass|scam|password|ssn|social security|credit card",
    "urgent_or_vulnerable": r"urgent|emergency|stolen|account hacked|deadline",
}
def decide(message: str, confidence: float, threshold: float = 0.60) -> dict:
    text = message.casefold()
    for reason, pattern in RISK_RULES.items():
        if re.search(pattern, text): return {"should_escalate": True, "reason": reason, "policy_confidence": 1.0}
    if confidence < threshold: return {"should_escalate": True, "reason": "low_intent_confidence", "policy_confidence": confidence}
    return {"should_escalate": False, "reason": "within_policy_and_confident", "policy_confidence": confidence}
