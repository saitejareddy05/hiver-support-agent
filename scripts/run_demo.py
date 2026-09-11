"""Quick end-to-end demo. Run after `python -m src.pipeline train`.

    python scripts/run_demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import SupportAgent

EXAMPLES = [
    "@AskPlayStation my controller drifts so bad I can't aim, is this covered under warranty?",
    "@AskPlayStation how do i transfer my saves from ps4 to my new ps5",
    "@AskPlayStation this is the third time I'm messaging about my refund, absolutely furious",
    "@AskPlayStation is the store down right now or is it just me",
    "@AskPlayStation charged me $85 twice for the same game, need this fixed",
]


def main():
    print("Loading trained agent...")
    agent = SupportAgent.load()

    for msg in EXAMPLES:
        resp = agent.handle(msg)
        print("\n" + "=" * 80)
        print(f"CUSTOMER: {msg}")
        print(f"  intent:     {resp.intent}  (confidence {resp.confidence:.2f})")
        print(f"  action:     {resp.action}  [{resp.trigger}]")
        print(f"  reason:     {resp.reason}")
        print(f"  reply:      {resp.reply}")


if __name__ == "__main__":
    main()
