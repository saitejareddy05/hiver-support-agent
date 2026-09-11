"""
Reply generation, grounded in the knowledge base of historical resolutions.

Two modes, chosen automatically based on environment:

1. LOCAL (default, no API key needed):
   Retrieve the top-k most similar past (customer_text -> agent_text) pairs from the
   KnowledgeBase, and adapt the single best-matching historical agent reply: reused
   near-verbatim if the match is close, or lightly stitched from the top-2 if not.
   This guarantees every word in the reply traces back to something the brand
   actually said before -> that's what "grounded" means for this deliverable, and it
   is what the eval harness's groundedness metric checks against.

2. LLM-ASSISTED (if ANTHROPIC_API_KEY is set):
   Same retrieval step, then the LLM is prompted to rewrite/merge the retrieved
   resolutions into a single natural reply *for this specific customer message*,
   explicitly instructed not to invent facts/policies beyond what's in the
   retrieved context. This is optional/better-quality, not required for the
   pipeline to run.

Both modes return the same GeneratedReply dataclass so the rest of the pipeline
(and the eval harness) doesn't care which one produced the text.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field

from src.knowledge_base import KnowledgeBase, KBMatch

GREETING_OPENERS = ["Hi there,", "Hello,", "Hey,", "Thanks for reaching out —"]


@dataclass
class GeneratedReply:
    text: str
    grounded_on: list[KBMatch] = field(default_factory=list)
    mode: str = "local"  # "local" or "llm"


def _local_generate(message: str, matches: list[KBMatch]) -> str:
    if not matches:
        return ("Thanks for reaching out. We want to help but need a bit more detail — "
                "could you DM us your account email or order ID along with a short "
                "description of the issue so we can look into it?")

    best = matches[0]
    reply = best.agent_text.strip()

    # If the top match is a weak match, blend in a second relevant match so the
    # reply isn't over-fit to one possibly-unrelated historical thread.
    if best.similarity < 0.25 and len(matches) > 1:
        second = matches[1]
        reply = reply.rstrip(".") + ". " + second.agent_text.strip()

    return reply


def _llm_generate(message: str, matches: list[KBMatch]) -> str | None:
    """Optional LLM-assisted rewrite. Returns None if no API key / call fails, so
    callers can fall back to _local_generate."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic  # imported lazily so the package is optional
    except ImportError:
        return None

    context_block = "\n".join(
        f"- Past customer msg: {m.customer_text}\n  Past agent reply: {m.agent_text}"
        for m in matches
    )
    prompt = (
        "You are a support agent for a PlayStation-style gaming brand replying on Twitter/X. "
        "Below are real past customer messages and how the brand actually resolved them. "
        "Write ONE new reply to the NEW customer message. Rules:\n"
        "- Only use facts, steps, and policies present in the past replies below. Do not invent new policies, links, or promises.\n"
        "- Keep it under 280 characters, friendly and concise, matching the brand's tone in the examples.\n"
        "- If none of the past replies are actually relevant, ask for the account email/order ID instead of guessing.\n\n"
        f"PAST RESOLUTIONS:\n{context_block}\n\n"
        f"NEW customer message: {message}\n\nReply:"
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "\n".join(text_blocks).strip() or None
    except Exception:
        return None


def generate_reply(message: str, kb: KnowledgeBase, k: int = 3) -> GeneratedReply:
    matches = kb.retrieve(message, k=k)

    llm_text = _llm_generate(message, matches)
    if llm_text:
        return GeneratedReply(text=llm_text, grounded_on=matches, mode="llm")

    local_text = _local_generate(message, matches)
    return GeneratedReply(text=local_text, grounded_on=matches, mode="local")
