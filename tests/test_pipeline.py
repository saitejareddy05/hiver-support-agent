"""
Basic sanity tests. Run with `pytest tests/` after `python -m src.pipeline train`
(the tests assume trained artifacts exist at artifacts/).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.intents import clean_text, INTENTS
from src.escalation import decide, SAFETY_PATTERN
from src.data_prep import load_brand_pairs


def test_clean_text_strips_mentions_and_urls():
    out = clean_text("@AskPlayStation check http://example.com please #urgent")
    assert "@" not in out
    assert "http" not in out
    assert "urgent" in out


def test_load_brand_pairs_returns_rows():
    pairs = load_brand_pairs()
    assert len(pairs) > 50
    assert {"customer_text", "agent_text", "thread_resolved"}.issubset(pairs.columns)


def test_safety_pattern_triggers_escalation():
    decision = decide(
        message="i want to kill myself over this",
        intent="technical_bug_error",
        confidence=0.9,
        grounding_score=0.9,
    )
    assert decision.action == "escalate_to_human"
    assert decision.trigger == "SAFETY"


def test_hardware_always_escalates():
    decision = decide(
        message="controller drift is annoying",
        intent="hardware_malfunction",
        confidence=0.99,
        grounding_score=0.99,
    )
    assert decision.action == "escalate_to_human"
    assert decision.trigger == "HIGH_RISK_INTENT"


def test_low_confidence_escalates():
    decision = decide(
        message="help",
        intent="how_to_general_inquiry",
        confidence=0.1,
        grounding_score=0.9,
    )
    assert decision.action == "escalate_to_human"
    assert decision.trigger == "LOW_CONFIDENCE"


def test_high_confidence_grounded_low_risk_auto_handles():
    decision = decide(
        message="how do i change my display name",
        intent="how_to_general_inquiry",
        confidence=0.9,
        grounding_score=0.9,
    )
    assert decision.action == "auto_handle"


def test_large_refund_escalates_on_amount():
    decision = decide(
        message="refund me $250 please",
        intent="refund_billing",
        confidence=0.9,
        grounding_score=0.9,
    )
    assert decision.action == "escalate_to_human"


@pytest.mark.skipif(
    not Path("artifacts/intent_classifier.joblib").exists(),
    reason="run `python -m src.pipeline train` first",
)
def test_full_pipeline_runs():
    from src.pipeline import SupportAgent
    agent = SupportAgent.load()
    resp = agent.handle("my controller drifts so bad I can't aim in any shooter anymore")
    assert resp.intent in INTENTS
    assert resp.action in ("auto_handle", "escalate_to_human")
    assert isinstance(resp.reply, str) and len(resp.reply) > 0
