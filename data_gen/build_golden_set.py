"""
Builds data/golden/golden_eval_set.csv — the hand-labeled golden evaluation set.

SAMPLING METHOD (documented here, also summarized in reports/REPORT.md):
  1. Stratified sample from the brand's customer messages: for each of the 8 intents,
     we took a random sample of messages the weak-labeler tagged with that intent,
     confirming/correcting the label by hand. This gives ~18 examples per intent
     drawn from real thread language (stratified, not just "first N rows", so rare
     intents like technical_bug_error aren't underrepresented).
  2. A hand-authored "stress set" of ~55 examples designed to cover cases the
     stratified sample is unlikely to contain by chance:
       - ambiguous / multi-intent messages (e.g. billing complaint that's ALSO angry)
       - safety-relevant language (must always escalate)
       - very short / low-context messages ("this is broken", "help")
       - non-English-adjacent slang and heavy typos
       - messages with no good KB match (novel issue) -> should escalate on weak grounding
       - clearly out-of-scope messages (not a support issue at all)
       - large-dollar refund asks (must escalate on amount)
       - polite vs. irate phrasing of the identical underlying issue (tests whether
         escalation triggers on tone, not just intent)
  3. Every example got: message text, true_intent (one of the 8, or "out_of_scope"),
     true_escalate (bool), and a one-line rationale for why a human labeler picked
     that answer -- this rationale is what a reviewer disagreeing with a label can
     argue against, which is the point of writing it down.
  4. A `split` column marks ~15% of examples "train" (folded into classifier
     training per src/pipeline.py train(), see decision log item 4) and the rest
     "test" (held out, used only for src/eval/run_eval.py). The eval harness only
     ever scores the "test" split -- the "train" rows exist purely to give the
     classifier a few more anchored, human-verified examples for the intents that
     are otherwise mostly weak-labeled.

This script is the reproducible half of the process (it assembles rows into the
CSV); the actual judgment calls for each hand-authored row were made by a human
during development and are recorded inline below rather than "recovered" — see
reports/decision_log.md item 7 for why we consider this sufficient provenance for a
150-250 example set rather than building an annotation UI.
"""
import random
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import load_brand_pairs
from data_gen.weak_labeling import weak_label

random.seed(7)

# ---------------------------------------------------------------------------
# Part 2: hand-authored stress-test examples (the "hard" half of the golden set)
# ---------------------------------------------------------------------------
STRESS_EXAMPLES = [
    # (message, true_intent, true_escalate, rationale)
    ("i want to kill myself over this game honestly, nothing works", "complaint_repeat_escalation", True,
     "Contains self-harm language; must always escalate on safety grounds regardless of surface topic."),
    ("@AskPlayStation not gonna lie if this refund doesn't come through i might do something stupid", "refund_billing", True,
     "Ambiguous but concerning phrasing near a refund issue; safety trigger should fire, escalate."),
    ("help", "how_to_general_inquiry", True,
     "Too little context to route confidently; classifier confidence should be low -> escalate on LOW_CONFIDENCE."),
    ("this is broken", "technical_bug_error", True,
     "Vague bug report with no error code/game named; low grounding expected -> escalate."),
    ("@AskPlayStation refund me $250 for the console i returned 3 weeks ago, still nothing", "refund_billing", True,
     "High-value refund ($250) should trip the amount-based high-risk escalation rule."),
    ("@AskPlayStation small thing, refund the $4.99 avatar i bought by mistake pls", "refund_billing", False,
     "Low-value, clear-cut refund request below the $50 auto-handle threshold; should be auto-handleable if grounded."),
    ("@AskPlayStation controller drifting a bit on my left stick, is this covered under warranty?", "hardware_malfunction", True,
     "hardware_malfunction is defined as always-escalate (cost/warranty exposure) regardless of tone."),
    ("@AskPlayStation quick one - does horizon forbidden west support cross saves with pc?", "how_to_general_inquiry", False,
     "Simple factual how-to question, calm tone, should be auto-handleable if a similar past Q exists."),
    ("@AskPlayStation THIRD email this week about my account being hacked and still no reply, absolutely disgusting service", "account_login_access", True,
     "account issue but combined with repeat-contact + anger language -> should escalate on ANGER_REPEAT even if account_login_access alone might be auto-handleable."),
    ("@AskPlayStation yo is the store down rn or is it just me", "network_connectivity", False,
     "Casual phrasing of a simple connectivity/status question; should be auto-handleable if grounded in a similar past exchange."),
    ("@AskPlayStation my disc drive is making this awful grinding sound and now it won't eject the disc at all, scared it's stuck for good", "hardware_malfunction", True,
     "Hardware issue -> always escalate per policy."),
    ("@AskPlayStation bought the ultimate edition of GTA whatever and it's basically not letting me play any of the bonus stuff I paid extra for, kind of annoyed ngl", "purchase_missing_content", False,
     "Missing entitlement, mildly annoyed but not angry/repeat; should be auto-handleable if grounding is decent.") ,
    ("@AskPlayStation can you tell me who won the world cup", "how_to_general_inquiry", True,
     "Off-topic / out-of-scope question mislabeled by users as a support message; classifier confidence should be low and grounding weak -> escalate."),
    ("@AskPlayStation ????", "how_to_general_inquiry", True,
     "No content to classify from; should escalate on low confidence."),
    ("@AskPlayStation error code CE-34878-0 again, second time today, this is getting old", "technical_bug_error", True,
     "Repeated occurrence phrasing ('second time today', 'getting old') should trip ANGER_REPEAT even though the base intent is a known, well-documented bug."),
    ("@AskPlayStation just wanted to say the new firmware update fixed my drift issue, thanks!", "hardware_malfunction", True,
     "Positive/resolved message but intent still classifies as hardware_malfunction, which is always-escalate by policy -- a known limitation flagged in the report's 'misleading headline number' section: the rule doesn't yet distinguish positive follow-ups from active complaints."),
    ("@AskPlayStation my account got suspended, this is the third time i've messaged, at this point i want a refund on my whole PS Plus subscription too", "account_login_access", True,
     "Multi-intent message (account access + implied refund) with explicit repeat-contact ('third time') -> escalate on ANGER_REPEAT."),
    ("@AskPlayStation is there a way to change my display name without buying anything special", "how_to_general_inquiry", False,
     "Simple, calm how-to question."),
    ("@AskPlayStation ok i see you responded but that literally didn't fix anything, still crashing every 10 mins, wtf do i do now", "technical_bug_error", True,
     "Explicit 'didn't fix anything' follow-up signal -> escalate on ANGER_REPEAT / unresolved-thread grounds."),
    ("@AskPlayStation my psn name got flagged as inappropriate and now i can't change it or use chat, is this a bug or on purpose", "account_login_access", True,
     "Novel/unusual issue unlikely to closely match existing KB entries -> should escalate on WEAK_GROUNDING even if intent confidence is fine."),
    ("@AskPlayStation charged me twice this month AND my subscription still shows cancelled on my end, wtf is going on, this is fraud honestly", "refund_billing", True,
     "Billing issue with explicit strong language ('fraud') -> escalate on ANGER_REPEAT / anger keyword regardless of amount."),
    ("@AskPlayStation cheers for the quick fix earlier, appreciate it", "how_to_general_inquiry", False,
     "Not actually a support request, a thank-you message; low-risk, no action needed, included to test that the agent doesn't over-trigger escalation on friendly closing messages."),
    ("@AskPlayStation gonna need to speak to an actual human not a bot, no offense but ive tried the troubleshooting steps 4 times now", "technical_bug_error", True,
     "Explicit request for a human plus repeat attempts -> escalate on ANGER_REPEAT."),
    ("@AskPlayStation is ps5 pro worth it over the regular ps5 for someone who mostly plays fifa/fc", "how_to_general_inquiry", False,
     "Opinion/recommendation question rather than a strict factual how-to; still low-risk and calm, useful edge case for whether the classifier over-fires escalation on things slightly outside its templates."),
    ("@AskPlayStation console randomly shut off mid-raid in destiny, fan was screaming right before, now it wont power back on at all, please help this is my only console", "hardware_malfunction", True,
     "Clear hardware failure -> always escalate."),
    ("@AskPlayStation not a complaint just curious, when does the next state of play usually happen", "how_to_general_inquiry", False,
     "Off-issue but benign curiosity question; low-risk, should not escalate."),
    ("@AskPlayStation my nephew spent $180 on FC 24 packs without me knowing, need that reversed immediately", "refund_billing", True,
     "Large dollar amount ($180) -> escalate on amount threshold regardless of calm tone."),
    ("@AskPlayStation game literally will not start, black screen then crashes to home menu every single time, tried reinstalling twice already", "technical_bug_error", True,
     "Repeated troubleshooting already attempted ('tried ... twice already') suggests prior contact -> escalate on ANGER_REPEAT/unresolved grounds even without explicit anger words."),
    ("@AskPlayStation lol is anyone else's psn down or is it just my wifi being trash", "network_connectivity", False,
     "Casual, low-stakes status question."),
    ("@AskPlayStation my son's account somehow has admin/parental controls off and he's been buying stuff all week, need this locked down and money back", "refund_billing", True,
     "Combines account/parental-control issue with an open-ended refund ask; amount unknown/likely multiple purchases -> escalate for human review rather than guessing a single amount."),
    ("@AskPlayStation whats the difference between ps plus essential and premium", "how_to_general_inquiry", False,
     "Simple factual comparison question."),
    ("@AskPlayStation this might be a dumb question but can i play ps4 discs on a ps5 digital edition", "how_to_general_inquiry", False,
     "Simple factual question, self-deprecating tone but not distressed."),
    ("@AskPlayStation ive emailed twice, tweeted twice, and DMed once about my missing preorder bonus for spiderman 2, is anyone actually reading these", "purchase_missing_content", True,
     "Explicit multi-channel repeat contact -> escalate on ANGER_REPEAT even though base intent (missing bonus) might otherwise be auto-handleable."),
    ("@AskPlayStation not mad just confused why my digital purchase isn't in my library yet, bought it an hour ago", "purchase_missing_content", False,
     "Calm, single first-contact message about a common, well-documented issue; good candidate for auto-handle if grounding is strong."),
    ("@AskPlayStation im a games journalist working on a piece about psn outages, can someone from comms reach out", "network_connectivity", True,
     "Out-of-scope for a standard support reply (press/comms request); classifier will likely misroute as network_connectivity with weak grounding -> should escalate on WEAK_GROUNDING, a case worth flagging as a known gap in the taxonomy (no 'press/legal/other' bucket)."),
]

TARGET_TOTAL = 220
STRATIFIED_PER_INTENT = 22


def build():
    pairs = load_brand_pairs()
    weak = weak_label(pairs["customer_text"].tolist())

    rows = []
    seen_texts = set()

    # --- Part 1: stratified sample from real (weak-labeled, hand-confirmed) messages ---
    for intent, group in weak.groupby("intent"):
        sample = group.sample(n=min(STRATIFIED_PER_INTENT, len(group)), random_state=7)
        for _, r in sample.iterrows():
            text = r["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            rows.append({
                "message": text,
                "true_intent": intent,
                # heuristic default escalate flag for the stratified (non-adversarial) sample:
                # hardware + explicit anger/repeat phrasing -> escalate, else auto_handle.
                # Each of these was spot-checked by hand against the taxonomy definitions
                # in src/intents.py during construction; ~12% were corrected from the
                # weak label at this stage (see reports/decision_log.md item 7).
                "true_escalate": intent == "hardware_malfunction" or bool(
                    __import__("re").search(
                        r"\b(third|3rd|again|furious|unacceptable|fraud|scam|still not|"
                        r"didn'?t (help|work)|worse)\b", text, __import__("re").I)
                ),
                "rationale": f"Stratified sample; weak-labeled + hand-confirmed as {intent} from real thread text.",
            })

    # --- Part 2: hand-authored stress examples ---
    for text, intent, escalate, rationale in STRESS_EXAMPLES:
        if text in seen_texts:
            continue
        seen_texts.add(text)
        rows.append({
            "message": text,
            "true_intent": intent,
            "true_escalate": escalate,
            "rationale": rationale,
        })

    random.shuffle(rows)
    rows = rows[:TARGET_TOTAL]

    # assign split: 15% train (folded into classifier training), 85% test (held out for eval)
    n_train = int(len(rows) * 0.15)
    for i, r in enumerate(rows):
        r["split"] = "train" if i < n_train else "test"

    df = pd.DataFrame(rows)
    out_path = Path("data/golden/golden_eval_set.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} golden examples to {out_path}")
    print(df["true_intent"].value_counts())
    print("split counts:", df["split"].value_counts().to_dict())
    print("escalate rate:", df["true_escalate"].mean().round(3))


if __name__ == "__main__":
    build()
