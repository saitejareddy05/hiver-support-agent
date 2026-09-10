"""Create a human-labeling worksheet using intent, length, and risk strata.

Sampling is deterministic and stratified: each inferred intent is crossed with short/
long threads and neutral/negative/urgent text flags. The output deliberately leaves
truth columns blank; a human must label them before evaluation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def build(frame, n=200, seed=42):
    frame = frame.copy(); text = frame.message_text.fillna("").str.casefold()
    frame["sentiment_flag"] = text.map(lambda x: "urgent_or_negative" if any(k in x for k in ["urgent", "late", "missing", "hate", "broken", "angry", "refund"]) else "neutral")
    frame["length_flag"] = frame.thread_length.map(lambda x: "long" if x >= 4 else "short")
    frame["stratum"] = frame.intent.fillna("other") + "|" + frame.length_flag + "|" + frame.sentiment_flag
    pieces = []
    for _, group in frame.groupby("stratum", dropna=False): pieces.append(group.sample(min(len(group), max(1, n // max(1, frame.stratum.nunique()))), random_state=seed))
    result = pd.concat(pieces).drop_duplicates("tweet_id").head(n)
    result = result.rename(columns={"text": "message_text"})
    result["true_intent"] = ""; result["true_should_escalate"] = ""; result["escalate_reason"] = ""; result["gold_reply_summary"] = ""; result["notes"] = ""
    print("Sampling report:\n", result.groupby("stratum").size().to_string())
    return result[["tweet_id", "brand", "message_text", "thread_context", "true_intent", "true_should_escalate", "escalate_reason", "gold_reply_summary", "notes"]]
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input", default="data/processed/amazonhelp_intents.csv"); p.add_argument("--output", default="data/golden_set/golden_eval.csv"); p.add_argument("--n", type=int, default=200); a=p.parse_args()
    data=pd.read_csv(a.input); Path(a.output).parent.mkdir(parents=True, exist_ok=True); build(data, a.n).to_csv(a.output,index=False)
