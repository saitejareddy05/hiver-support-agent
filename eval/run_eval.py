from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from src.pipeline import SupportAgent, load_config
from src.baselines import MajorityBaseline
from .metrics import intent_metrics, escalation_metrics, retrieval_hit_rate

def run(golden_path, history_path, config_path="config.yaml"):
    gold=pd.read_csv(golden_path); history=pd.read_csv(history_path); agent=SupportAgent(history, load_config(config_path))
    outputs=[agent.run(x) for x in gold.message_text]; intents=pd.Series([x["intent"] for x in outputs]); escalations=pd.Series([x["should_escalate"] for x in outputs])
    result={**intent_metrics(gold,intents), **escalation_metrics(gold,escalations), "retrieval_hit_rate": retrieval_hit_rate(outputs)}
    result["baseline_majority_intent"] = MajorityBaseline(history).intent
    return result
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--golden", default="data/golden_set/golden_eval.csv"); p.add_argument("--history", default="data/processed/sample.csv"); p.add_argument("--config", default="config.yaml"); p.add_argument("--output", default="results/eval.json"); a=p.parse_args()
    result=run(a.golden,a.history,a.config); Path(a.output).parent.mkdir(exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
