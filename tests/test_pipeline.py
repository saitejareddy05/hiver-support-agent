from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.pipeline import SupportAgent

def agent():
    frame=pd.read_csv(Path(__file__).parents[1]/"data/processed/sample.csv")
    return SupportAgent(frame, {"brand":"AmazonHelp","retrieval_top_k":2,"escalation_confidence_threshold":.6,"max_reply_chars":500})
def test_classification_and_retrieval():
    result=agent().run("Where is my package?")
    assert result["intent"] and result["retrieved_examples"] and result["reply"]
def test_risk_escalates_without_api():
    result=agent().run("I was charged twice and need a refund")
    assert result["should_escalate"] is True
    assert result["reason"] == "refund_or_financial"
