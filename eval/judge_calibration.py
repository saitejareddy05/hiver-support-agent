"""Compare human and LLM quality scores on a reviewed subsample."""
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def calibrate(path):
    data = pd.read_csv(path).dropna(subset=["human_score", "llm_score"])
    if data.empty: raise ValueError("Add human_score and llm_score columns after hand-scoring the subsample.")
    return {"n": len(data), "weighted_kappa": float(cohen_kappa_score(data.human_score, data.llm_score, weights="quadratic"))}
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("path"); print(calibrate(p.parse_args().path))
