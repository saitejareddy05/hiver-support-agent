"""
Judge-vs-human agreement check.

Methodology: 30 real pipeline outputs were sampled (pandas .sample(random_state=11))
from a full eval run (see eval_results/human_calibration_sample.csv, generated
alongside src/eval/human_calibration_scores.csv) and scored by a human on the same
1-5 rubric across the same four dimensions the automated judge uses
(groundedness/helpfulness/tone/correctness). This script loads both score sets and
reports agreement.

Run AFTER `python -m src.eval.run_eval` at least once (it reuses that sample file).
Usage: python -m src.eval.human_agreement
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def _safe_corr(fn, a, b):
    if len(set(a)) <= 1 or len(set(b)) <= 1:
        return float("nan")
    r, _ = fn(a, b)
    return round(r, 3)

DIMS = ["groundedness", "helpfulness", "tone", "correctness"]

SAMPLE_PATH = Path("eval_results/human_calibration_sample.csv")
HUMAN_PATH = Path("src/eval/human_calibration_scores.csv")


def run():
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLE_PATH} not found. Run `python -m src.eval.run_eval` first, "
            "which draws eval_results/human_calibration_sample.csv from the live "
            "predictions (see src/eval/run_eval.py note or README)."
        )
    judge_df = pd.read_csv(SAMPLE_PATH).reset_index(drop=True)
    human_df = pd.read_csv(HUMAN_PATH).sort_values("sample_index").reset_index(drop=True)

    assert len(judge_df) == len(human_df), "Calibration sample size mismatch — regenerate both files together."

    results = {}
    for dim in DIMS:
        judge_col = f"agent_judge_{dim}"
        human_col = f"human_{dim}"
        pearson_r = _safe_corr(pearsonr, judge_df[judge_col], human_df[human_col])
        spearman_r = _safe_corr(spearmanr, judge_df[judge_col], human_df[human_col])
        within_1 = (judge_df[judge_col] - human_df[human_col]).abs().le(1).mean()
        results[dim] = {
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "pct_within_1_point": round(within_1, 3),
        }

    overall_judge = judge_df["agent_judge_overall"]
    overall_human = human_df[[f"human_{d}" for d in DIMS]].mean(axis=1)
    overall_pearson = _safe_corr(pearsonr, overall_judge, overall_human)
    results["overall"] = {
        "pearson_r": overall_pearson,
        "pct_within_1_point": round((overall_judge - overall_human).abs().le(1).mean(), 3),
    }

    print(f"Judge-vs-human agreement over n={len(judge_df)} calibration examples:\n")
    for dim, stats in results.items():
        print(f"  {dim:15s} pearson_r={stats['pearson_r']:.3f}  "
              f"pct_within_1_point={stats.get('pct_within_1_point', float('nan')):.3f}"
              + (f"  spearman_r={stats['spearman_r']:.3f}" if "spearman_r" in stats else ""))

    print("\nNote: this run used the '{}' judge backend (see eval_results/summary.json). "
          "If that backend is 'heuristic', treat this as calibration of the PROXY judge, "
          "not the real LLM judge -- see reports/REPORT.md.".format(
              pd.read_json("eval_results/summary.json", typ="series").get("judge_backend", "unknown")
          ))
    return results


if __name__ == "__main__":
    run()
