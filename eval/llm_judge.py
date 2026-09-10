from __future__ import annotations
import json
from pathlib import Path

def rubric_prompt(message, reply, evidence):
    rubric = Path(__file__).parents[1].joinpath("src/prompts/judge_rubric.txt").read_text(encoding="utf-8")
    return rubric + f"\nMESSAGE:\n{message}\nREPLY:\n{reply}\nEVIDENCE:\n{evidence}"
def parse_judge_response(text):
    try: return json.loads(text)
    except json.JSONDecodeError as exc: raise ValueError("Judge did not return strict JSON") from exc
def average_score(score): return sum(score[k] for k in ["grounded_in_history","factual_consistency","tone","actionable","concise"]) / 5
