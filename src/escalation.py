"""
Escalation decision: auto_handle vs escalate_to_human, with a stated reason.

This is intentionally a transparent rule layer on top of the classifier's confidence
and the retrieval quality, NOT another opaque model — for a support agent, "why did
you decide to auto-send this?" needs to be answerable in one sentence, and rules are
auditable/tunable by a human reviewer without retraining anything (see decision log
item 8: we prioritized explainability over squeezing out extra recall here).

Escalation triggers, checked in order (first match wins, reasons are stack-ranked by
severity):
  1. SAFETY          - self-harm / threats / illegal-activity language -> always escalate.
  2. HIGH_RISK_INTENT - refund_billing above a $ threshold, or hardware_malfunction
                        (potential warranty/legal/cost exposure) -> always escalate.
  3. LOW_CONFIDENCE   - classifier top-1 probability below threshold -> escalate
                        (we don't trust the intent routing enough to auto-reply).
  4. WEAK_GROUNDING   - no knowledge-base match above similarity threshold -> escalate
                        (we have nothing real to ground a reply in, so we'd be
                        generating from nothing / guessing).
  5. ANGER_REPEAT     - complaint_repeat_escalation intent, or anger keywords, or an
                        unresolved follow-up in the thread -> escalate.
  6. otherwise: AUTO_HANDLE.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

CONFIDENCE_THRESHOLD = 0.45
GROUNDING_THRESHOLD = 0.12
REFUND_AMOUNT_THRESHOLD = 50.0

SAFETY_PATTERN = re.compile(
    r"\b(kill myself|suicide|self[- ]harm|end my life|hurt myself|going to hurt|"
    r"i have a weapon|bomb threat)\b", re.IGNORECASE,
)
ANGER_PATTERN = re.compile(
    r"\b(furious|unacceptable|ridiculous|scam|lawsuit|lawyer|sue you|"
    r"never again|worst (support|service)|disgusting|fraud)\b", re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")

HIGH_RISK_INTENTS = {"hardware_malfunction"}


@dataclass
class EscalationDecision:
    action: str          # "auto_handle" | "escalate_to_human"
    reason: str           # human-readable, one sentence
    trigger: str           # machine-readable trigger code


def decide(message: str, intent: str, confidence: float, grounding_score: float,
           has_unresolved_followup: bool = False) -> EscalationDecision:

    if SAFETY_PATTERN.search(message):
        return EscalationDecision(
            action="escalate_to_human",
            reason="Message contains language suggesting a safety risk; always routed to a human regardless of intent or confidence.",
            trigger="SAFETY",
        )

    if intent in HIGH_RISK_INTENTS:
        return EscalationDecision(
            action="escalate_to_human",
            reason=f"Intent '{intent}' carries potential cost/warranty exposure; these are always human-reviewed even when the classifier is confident.",
            trigger="HIGH_RISK_INTENT",
        )

    if intent == "refund_billing":
        amounts = [float(a) for a in AMOUNT_PATTERN.findall(message)]
        if any(a >= REFUND_AMOUNT_THRESHOLD for a in amounts):
            return EscalationDecision(
                action="escalate_to_human",
                reason=f"Refund/billing request involves an amount at or above the ${REFUND_AMOUNT_THRESHOLD:.0f} auto-handle limit.",
                trigger="HIGH_RISK_INTENT",
            )

    if confidence < CONFIDENCE_THRESHOLD:
        return EscalationDecision(
            action="escalate_to_human",
            reason=f"Intent classifier confidence ({confidence:.2f}) is below the {CONFIDENCE_THRESHOLD:.2f} threshold; routing is not reliable enough to auto-reply.",
            trigger="LOW_CONFIDENCE",
        )

    if grounding_score < GROUNDING_THRESHOLD:
        return EscalationDecision(
            action="escalate_to_human",
            reason=f"No sufficiently similar historical resolution was found (best match similarity {grounding_score:.2f}); auto-replying would mean generating an ungrounded answer.",
            trigger="WEAK_GROUNDING",
        )

    if intent == "complaint_repeat_escalation" or ANGER_PATTERN.search(message) or has_unresolved_followup:
        return EscalationDecision(
            action="escalate_to_human",
            reason="Message shows signs of repeated contact or escalated frustration; these are routed to a human to protect the relationship even if a templated answer exists.",
            trigger="ANGER_REPEAT",
        )

    return EscalationDecision(
        action="auto_handle",
        reason=f"High classifier confidence ({confidence:.2f}), strong grounding (similarity {grounding_score:.2f}), and low-risk intent '{intent}' with no anger/repeat signals.",
        trigger="AUTO_OK",
    )
