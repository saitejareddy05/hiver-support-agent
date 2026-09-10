"""Grounded reply drafting with optional OpenAI-compatible LLM and offline fallback."""
from __future__ import annotations
import os
from pathlib import Path

def _offline(message, intent, examples, max_chars):
    evidence = examples[0].get("resolution", "") if examples else ""
    if evidence:
        reply = f"I’m sorry this has been frustrating. Based on similar {intent.replace('_', ' ')} cases, {evidence}"
    else: reply = "I’m sorry you’re dealing with this. I’ll route it to a specialist who can review the details safely."
    return reply[:max_chars]

def draft_reply(message, intent, examples, brand="AmazonHelp", max_chars=500, client=None, model="gpt-4o-mini"):
    if client is None or not os.getenv("LLM_API_KEY"):
        return _offline(message, intent, examples, max_chars)
    system = Path(__file__).parent.joinpath("prompts/reply_system.txt").read_text(encoding="utf-8").format(brand=brand, max_reply_chars=max_chars)
    evidence = "\n".join(f"Customer: {x.get('message_text','')}\nResolution: {x.get('resolution','')}" for x in examples)
    user = Path(__file__).parent.joinpath("prompts/reply_user.txt").read_text(encoding="utf-8").format(message=message, intent=intent, examples=evidence)
    response = client.chat.completions.create(model=model, temperature=0.1, messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return response.choices[0].message.content.strip()[:max_chars]
