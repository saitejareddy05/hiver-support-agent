"""
Two baselines required by the assignment:

TRIVIAL baseline: always predicts the majority-class intent from the training
  distribution, and always escalates (the "safest possible" policy — never
  auto-handles anything). This lower-bounds what "doing nothing clever" gets you.

SIMPLE baseline: keyword/regex rules (literally the weak-labeling rules from
  data_gen/weak_labeling.py, reused here as a standalone classifier) + a single
  fixed reply template per intent (no retrieval/grounding) + escalate only on the
  explicit safety pattern, auto-handle everything else. This is "the thing you'd
  ship in an afternoon without any ML," and is the bar the real system needs to
  clear to justify its added complexity.
"""
from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data_gen.weak_labeling import RULES
from src.escalation import SAFETY_PATTERN

FIXED_REPLIES = {
    "account_login_access": "Please try resetting your password from account.sonyentertainmentnetwork.com and check spam for the reset email.",
    "refund_billing": "Please DM your order ID and we'll look into the charge.",
    "purchase_missing_content": "Please try restarting your console and checking Library > Purchased. DM your order ID if it's still missing.",
    "technical_bug_error": "Please make sure your system software and the game are fully updated, then try again.",
    "network_connectivity": "Please check status.playstation.com for known outages and try a wired connection if possible.",
    "hardware_malfunction": "Please start a repair request at playstation.com/repair.",
    "how_to_general_inquiry": "Thanks for the question — please check our support site for the latest guidance on this.",
    "complaint_repeat_escalation": "We're sorry to hear that. Please DM your ticket number.",
}
MAJORITY_INTENT_FALLBACK = "how_to_general_inquiry"  # set from training distribution at eval time


@dataclass
class BaselinePrediction:
    intent: str
    reply: str
    action: str
    reason: str


def trivial_baseline(message: str, majority_intent: str = MAJORITY_INTENT_FALLBACK) -> BaselinePrediction:
    return BaselinePrediction(
        intent=majority_intent,
        reply="Thanks for reaching out, a member of our team will follow up with you shortly.",
        action="escalate_to_human",
        reason="Trivial baseline: always escalates, never attempts to auto-handle anything.",
    )


def simple_baseline(message: str, majority_intent: str = MAJORITY_INTENT_FALLBACK) -> BaselinePrediction:
    best_intent, best_hits = None, 0
    for intent, pattern in RULES:
        hits = len(pattern.findall(message))
        if hits > best_hits:
            best_hits = hits
            best_intent = intent
    intent = best_intent or majority_intent

    if SAFETY_PATTERN.search(message):
        action, reason = "escalate_to_human", "Simple baseline: safety keyword match."
    else:
        action, reason = "auto_handle", "Simple baseline: auto-handles everything except safety keyword matches."

    return BaselinePrediction(
        intent=intent,
        reply=FIXED_REPLIES.get(intent, "Thanks for reaching out, please DM us more details."),
        action=action,
        reason=reason,
    )
