"""
Runs the full evaluation: agent + two baselines, over the held-out ("test" split)
golden set. Writes:
  eval_results/predictions.csv     - row-level predictions for every system
  eval_results/summary.json        - all headline metrics
  eval_results/summary.md          - human-readable table (also printed to stdout)

Usage: python -m src.eval.run_eval
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.pipeline import SupportAgent
from src.eval.baselines import trivial_baseline, simple_baseline
from src.eval.metrics import intent_metrics, escalation_metrics
from src.eval.llm_judge import judge_reply
from src.intents import INTENTS

GOLDEN_PATH = Path("data/golden/golden_eval_set.csv")
OUT_DIR = Path("eval_results")


def load_test_set() -> pd.DataFrame:
    df = pd.read_csv(GOLDEN_PATH)
    df = df[df["split"] == "test"].reset_index(drop=True)
    df["true_escalate"] = df["true_escalate"].astype(str).str.lower().isin(["true", "1"])
    return df


def run():
    OUT_DIR.mkdir(exist_ok=True)
    test_df = load_test_set()
    print(f"Evaluating on {len(test_df)} held-out golden test examples.")

    agent = SupportAgent.load()
    majority_intent = test_df["true_intent"].value_counts().idxmax()

    rows = []
    for _, r in test_df.iterrows():
        msg = r["message"]

        resp = agent.handle(msg)
        triv = trivial_baseline(msg, majority_intent)
        simp = simple_baseline(msg, majority_intent)

        grounding_texts = [m.agent_text for m in agent.kb.retrieve(msg, k=3)]
        agent_judge = judge_reply(msg, resp.reply, grounding_texts)
        simple_judge = judge_reply(msg, simp.reply, [])  # simple baseline has no grounding by design

        rows.append({
            "message": msg,
            "true_intent": r["true_intent"],
            "true_escalate": r["true_escalate"],
            "rationale": r.get("rationale", ""),

            "agent_intent": resp.intent,
            "agent_confidence": round(resp.confidence, 3),
            "agent_action": resp.action,
            "agent_trigger": resp.trigger,
            "agent_reply": resp.reply,
            "agent_judge_overall": agent_judge.overall,
            "agent_judge_groundedness": agent_judge.groundedness,
            "agent_judge_helpfulness": agent_judge.helpfulness,
            "agent_judge_tone": agent_judge.tone,
            "agent_judge_correctness": agent_judge.correctness,

            "trivial_intent": triv.intent,
            "trivial_action": triv.action,

            "simple_intent": simp.intent,
            "simple_action": simp.action,
            "simple_judge_overall": simple_judge.overall,
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

    def escalate_bool(series):
        return series.eq("escalate_to_human")

    summary = {
        "n_test_examples": len(pred_df),
        "judge_backend": judge_reply(pred_df.iloc[0]["message"], pred_df.iloc[0]["agent_reply"], []).backend,
        "agent": {
            "intent": intent_metrics(pred_df["true_intent"], pred_df["agent_intent"], INTENTS),
            "escalation": escalation_metrics(pred_df["true_escalate"], escalate_bool(pred_df["agent_action"])),
            "judge_overall_mean": round(pred_df["agent_judge_overall"].mean(), 3),
        },
        "trivial_baseline": {
            "intent": intent_metrics(pred_df["true_intent"], pred_df["trivial_intent"], INTENTS),
            "escalation": escalation_metrics(pred_df["true_escalate"], escalate_bool(pred_df["trivial_action"])),
        },
        "simple_baseline": {
            "intent": intent_metrics(pred_df["true_intent"], pred_df["simple_intent"], INTENTS),
            "escalation": escalation_metrics(pred_df["true_escalate"], escalate_bool(pred_df["simple_action"])),
            "judge_overall_mean": round(pred_df["simple_judge_overall"].mean(), 3),
        },
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    table = [
        ["System", "Intent Acc", "Intent Macro-F1", "Escalate F1", "Unsafe Auto-Handle Rate", "Reply Judge (1-5)"],
        ["Trivial baseline", summary["trivial_baseline"]["intent"]["accuracy"],
         summary["trivial_baseline"]["intent"]["macro_f1"],
         summary["trivial_baseline"]["escalation"]["f1_escalate"],
         summary["trivial_baseline"]["escalation"]["unsafe_auto_handle_rate"], "n/a (no reply drafted)"],
        ["Simple baseline", summary["simple_baseline"]["intent"]["accuracy"],
         summary["simple_baseline"]["intent"]["macro_f1"],
         summary["simple_baseline"]["escalation"]["f1_escalate"],
         summary["simple_baseline"]["escalation"]["unsafe_auto_handle_rate"],
         summary["simple_baseline"]["judge_overall_mean"]],
        ["Full agent", summary["agent"]["intent"]["accuracy"],
         summary["agent"]["intent"]["macro_f1"],
         summary["agent"]["escalation"]["f1_escalate"],
         summary["agent"]["escalation"]["unsafe_auto_handle_rate"],
         summary["agent"]["judge_overall_mean"]],
    ]
    md_table = tabulate(table[1:], headers=table[0], tablefmt="github")
    header = (f"# Evaluation summary\n\nJudge backend used: **{summary['judge_backend']}** "
              f"({'set ANTHROPIC_API_KEY to use the real LLM judge' if summary['judge_backend']=='heuristic' else 'LLM judge'})\n\n"
              f"n test examples: {summary['n_test_examples']}\n\n")
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write(header + md_table + "\n")

    print("\n" + header + md_table)
    print(f"\nFull predictions written to {OUT_DIR/'predictions.csv'}")
    print(f"Full metrics (incl. confusion matrix, per-class report) written to {OUT_DIR/'summary.json'}")

    # Regenerate the fixed 30-example human-calibration sample (same random_state
    # every run) so `python -m src.eval.human_agreement` always has something to
    # compare the hand-labeled scores in src/eval/human_calibration_scores.csv against.
    calib_cols = ["message", "agent_reply", "agent_judge_overall", "agent_judge_groundedness",
                  "agent_judge_helpfulness", "agent_judge_tone", "agent_judge_correctness"]
    if len(pred_df) >= 30:
        calib_sample = pred_df.sample(n=30, random_state=11)[calib_cols].reset_index(drop=True)
        calib_sample.to_csv(OUT_DIR / "human_calibration_sample.csv", index=False)
        print(f"Human-calibration sample (n=30, fixed seed) written to {OUT_DIR/'human_calibration_sample.csv'}")
        print("Run `python -m src.eval.human_agreement` to see judge-vs-human agreement stats.")


if __name__ == "__main__":
    run()
