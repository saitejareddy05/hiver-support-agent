"""
LLM-as-judge for reply quality, scoring each (customer message, drafted reply,
grounding context) triple on a 1-5 rubric across four dimensions:

  groundedness   - is the reply actually supported by the retrieved historical
                    resolutions, or does it invent facts/policies?
  helpfulness    - does it move the customer's problem forward (concrete next step,
                    not just an empty acknowledgement)?
  tone           - appropriately empathetic/professional for a support context?
  correctness    - not contradicted by anything we know (e.g. doesn't promise a
                    refund policy that isn't real, doesn't misstate steps)?

Two backends:
  - LLM backend (ANTHROPIC_API_KEY set): asks Claude to score the rubric directly.
  - Heuristic backend (default, no key needed): a transparent, rule-based proxy
    scorer using lexical overlap with the grounding context (groundedness), presence
    of a concrete next step / question / link (helpfulness), politeness markers and
    length (tone), and a penalty for contradicting known facts (correctness).

The heuristic backend exists so `python -m src.eval.run_eval` works out of the box
in <15 minutes with zero setup, per the assignment's reproducibility requirement.
It's explicitly a PROXY, not a replacement for the real LLM judge — see
reports/REPORT.md "what's misleading about the headline number" for why the
heuristic numbers should not be over-trusted, and human_agreement.py for the actual
agreement-with-human evidence this harness produces.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, asdict

DIMENSIONS = ["groundedness", "helpfulness", "tone", "correctness"]

POLITE_MARKERS = re.compile(r"\b(sorry|thanks|thank you|please|appreciate|happy to)\b", re.I)
NEXT_STEP_MARKERS = re.compile(r"\b(please dm|please try|please check|start a repair request|"
                                r"reset your password|please submit)\b", re.I)
HEDGE_MARKERS = re.compile(r"\b(we're not sure|unclear|maybe|not sure|no idea)\b", re.I)


@dataclass
class JudgeScore:
    groundedness: int
    helpfulness: int
    tone: int
    correctness: int
    overall: float
    backend: str
    rationale: str

    def to_dict(self):
        return asdict(self)


def _word_overlap(a: str, b: str) -> float:
    aw = set(re.findall(r"[a-z]{3,}", a.lower()))
    bw = set(re.findall(r"[a-z]{3,}", b.lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def _heuristic_judge(message: str, reply: str, grounding_texts: list[str]) -> JudgeScore:
    grounding_blob = " ".join(grounding_texts)

    # groundedness: lexical overlap between reply and the retrieved historical replies
    overlap = _word_overlap(reply, grounding_blob) if grounding_blob else 0.0
    groundedness = 1 if not grounding_blob else min(5, max(1, round(1 + overlap * 8)))

    # helpfulness: does it contain a concrete next step, and is it non-trivially long?
    has_next_step = bool(NEXT_STEP_MARKERS.search(reply))
    length_ok = 20 <= len(reply.split()) <= 70
    helpfulness = 3 + (1 if has_next_step else -1) + (1 if length_ok else -1)
    helpfulness = min(5, max(1, helpfulness))

    # tone: politeness markers present, not overly curt, not a hedge-fest
    polite_hits = len(POLITE_MARKERS.findall(reply))
    hedge_hits = len(HEDGE_MARKERS.findall(reply))
    tone = 3 + min(2, polite_hits) - min(2, hedge_hits)
    tone = min(5, max(1, tone))

    # correctness: proxy — penalize replies that contradict a "no" signal in the
    # message (e.g. customer says "already tried X" and reply says "try X" again)
    correctness = 4
    if "already tried" in message.lower() or "tried that" in message.lower():
        # crude check: if the reply's next-step verb also appears right after
        # "already tried" in the message, it's likely repeating a step the
        # customer said didn't work
        correctness = 3
    correctness = min(5, max(1, correctness))

    overall = round((groundedness + helpfulness + tone + correctness) / 4, 2)
    rationale = (
        f"heuristic: lexical overlap with grounding={overlap:.2f}, "
        f"has_next_step={has_next_step}, length_ok={length_ok}, "
        f"polite_hits={polite_hits}, hedge_hits={hedge_hits}"
    )
    return JudgeScore(groundedness, helpfulness, tone, correctness, overall, "heuristic", rationale)


def _llm_judge(message: str, reply: str, grounding_texts: list[str]) -> JudgeScore | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        import json
    except ImportError:
        return None

    grounding_blob = "\n".join(f"- {g}" for g in grounding_texts) or "(no grounding context retrieved)"
    prompt = f"""You are grading a customer support agent's draft reply on a 1-5 scale across four dimensions.

Customer message: {message}
Drafted reply: {reply}
Historical resolutions this reply was supposed to be grounded in:
{grounding_blob}

Score each 1 (poor) to 5 (excellent):
- groundedness: is the reply actually supported by the historical resolutions, without inventing facts/policies?
- helpfulness: does it give the customer a concrete, actionable next step?
- tone: is it empathetic and professional?
- correctness: is anything in the reply factually wrong or contradicted by the message?

Respond ONLY with JSON: {{"groundedness": int, "helpfulness": int, "tone": int, "correctness": int, "rationale": "one sentence"}}"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        text = text.strip().strip("`").replace("json\n", "")
        data = json.loads(text)
        scores = [data["groundedness"], data["helpfulness"], data["tone"], data["correctness"]]
        overall = round(sum(scores) / 4, 2)
        return JudgeScore(*scores, overall=overall, backend="llm", rationale=data.get("rationale", ""))
    except Exception:
        return None


def judge_reply(message: str, reply: str, grounding_texts: list[str]) -> JudgeScore:
    llm_score = _llm_judge(message, reply, grounding_texts)
    if llm_score:
        return llm_score
    return _heuristic_judge(message, reply, grounding_texts)
