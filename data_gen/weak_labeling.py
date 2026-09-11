"""
Weak labeling: assigns a provisional intent label to a customer message using
keyword/regex rules per intent. This is NOT the ground truth (that's the hand-labeled
golden set in data/golden/) — it exists purely to give the supervised classifier
enough labeled training volume from the mostly-unlabeled dataset, at low cost and
with full transparency about what each rule is keying off.

This is a standard "weak supervision" pattern: cheap, noisy labels for training data
volume, an independent hand-labeled set for evaluation. See decision log item 4 for
why we didn't hand-label thousands of training examples instead (not worth the time
budget vs. the eval set, where hand-labels actually matter for trustworthiness).
"""
from __future__ import annotations
import re
import pandas as pd

RULES = [
    ("account_login_access", re.compile(
        r"\b(login|log in|password|2fa|locked|lock(ed)?|signed? out|"
        r"can'?t sign in|hacked|compromised|suspend)\b", re.I)),
    ("refund_billing", re.compile(
        r"\b(refund|charged|charge|billing|subscription|renew(ed|al)?|"
        r"money back|double (charged|billed)|\$\d)\b", re.I)),
    ("purchase_missing_content", re.compile(
        r"\b(not showing|missing|library|entitlement|pre-?order bonus|"
        r"paid but|download(ed)? but|dlc)\b", re.I)),
    ("technical_bug_error", re.compile(
        r"\b(crash(es|ing)?|freeze|freezing|error [a-z]{2}-?\d|bug|glitch|"
        r"black screen|won'?t (save|load))\b", re.I)),
    ("network_connectivity", re.compile(
        r"\b(psn|network|disconnect(ing|ed)?|nat type|outage|down for me|"
        r"can'?t connect|lag(gy)?)\b", re.I)),
    ("hardware_malfunction", re.compile(
        r"\b(controller drift|drift(ing)?|won'?t turn on|blinking (blue|light)|"
        r"disc drive|grinding|overheat(ing)?|fan(s)? (are )?loud)\b", re.I)),
    ("how_to_general_inquiry", re.compile(
        r"\b(how do i|how (do|can) you|quick q|is .* cross-?platform|"
        r"how much storage|how to)\b", re.I)),
    ("complaint_repeat_escalation", re.compile(
        r"\b(third time|3rd time|again|furious|unacceptable|actual human|"
        r"speak to a human|keep(s)? (closing|ignoring)|done with this)\b", re.I)),
]


def weak_label(texts: list[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        best_intent = None
        best_hits = 0
        for intent, pattern in RULES:
            hits = len(pattern.findall(t))
            if hits > best_hits:
                best_hits = hits
                best_intent = intent
        if best_intent is not None:
            rows.append({"text": t, "intent": best_intent})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_prep import load_brand_pairs

    pairs = load_brand_pairs()
    labeled = weak_label(pairs["customer_text"].tolist())
    print(f"Weak-labeled {len(labeled)}/{len(pairs)} messages ({len(labeled)/len(pairs):.1%} coverage)")
    print(labeled["intent"].value_counts())
