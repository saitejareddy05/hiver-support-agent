"""
End-to-end pipeline: SupportAgent.handle(message) -> AgentResponse

    from src.pipeline import SupportAgent
    agent = SupportAgent.load()
    resp = agent.handle("hey @AskPlayStation my controller drifts so bad")
    print(resp.intent, resp.confidence, resp.action, resp.reason)
    print(resp.reply)

Run `python -m src.pipeline train` to (re)build the artifacts (intent classifier +
knowledge base) from data/sample or data/raw.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, asdict

from src.data_prep import load_brand_pairs
from src.intents import IntentClassifier, INTENTS
from src.knowledge_base import KnowledgeBase
from src.reply_generator import generate_reply
from src.escalation import decide


@dataclass
class AgentResponse:
    message: str
    intent: str
    confidence: float
    intent_distribution: dict
    reply: str
    reply_mode: str
    action: str
    reason: str
    trigger: str
    grounding_score: float

    def to_dict(self):
        return asdict(self)


class SupportAgent:
    def __init__(self, classifier: IntentClassifier, kb: KnowledgeBase):
        self.classifier = classifier
        self.kb = kb

    @classmethod
    def load(cls) -> "SupportAgent":
        return cls(classifier=IntentClassifier.load(), kb=KnowledgeBase.load())

    def handle(self, message: str) -> AgentResponse:
        intent, confidence, dist = self.classifier.predict_one(message)
        gen = generate_reply(message, self.kb, k=3)
        grounding_score = gen.grounded_on[0].similarity if gen.grounded_on else 0.0

        decision = decide(
            message=message,
            intent=intent,
            confidence=confidence,
            grounding_score=grounding_score,
        )

        return AgentResponse(
            message=message,
            intent=intent,
            confidence=confidence,
            intent_distribution=dist,
            reply=gen.text,
            reply_mode=gen.mode,
            action=decision.action,
            reason=decision.reason,
            trigger=decision.trigger,
            grounding_score=grounding_score,
        )


def train(brand: str = "AskPlayStation"):
    """Builds and saves the intent classifier + knowledge base from the loaded
    brand data. The intent classifier needs labels, which the raw dataset does not
    provide — so for TRAINING we use the golden-set labels (see data/golden) plus a
    small amount of weak/heuristic keyword-based labeling over the unlabeled pairs
    to get enough volume. This is spelled out in reports/decision_log.md (item 4)."""
    import pandas as pd
    from data_gen.weak_labeling import weak_label

    pairs = load_brand_pairs(brand=brand)
    print(f"Loaded {len(pairs)} customer/agent pairs for {brand}")

    # Knowledge base is built from ALL pairs (no labels needed for retrieval).
    kb = KnowledgeBase().fit(pairs)
    kb.save()
    print("Saved knowledge base.")

    # Intent classifier needs labels -> weak-label the unlabeled pairs, then fold in
    # the hand-labeled golden set (train split only, see src/eval/build_golden.py)
    # so the classifier isn't purely trained on its own weak labels.
    weak_labeled = weak_label(pairs["customer_text"].tolist())
    texts = list(weak_labeled["text"])
    labels = list(weak_labeled["intent"])

    golden_path = "data/golden/golden_eval_set.csv"
    try:
        golden = pd.read_csv(golden_path)
        train_split = golden[golden["split"] == "train"]
        texts += list(train_split["message"])
        labels += list(train_split["true_intent"])
        print(f"Folded in {len(train_split)} hand-labeled golden 'train' examples.")
    except FileNotFoundError:
        print("No golden set found yet; training on weak labels only.")

    clf = IntentClassifier().fit(texts, labels)
    clf.save()
    print(f"Saved intent classifier trained on {len(texts)} examples across {len(set(labels))} intents.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print("Usage: python -m src.pipeline train")
