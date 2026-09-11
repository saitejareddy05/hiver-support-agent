# Hiver SDE Intern Take-Home — Report
**Brand:** `@AskPlayStation` (PlayStation platform support) · **Repo:** hiver-support-agent

All numbers below come from an actual run of `python -m src.eval.run_eval` and
`python -m src.eval.human_agreement` on the held-out golden test split (n=133).
Raw outputs are in `eval_results/predictions.csv` and `eval_results/summary.json`
if you want to check anything against source.

---

## 1. Problem framing

**What "good" means for this brand.** A gaming-console support account handles a mix
of (a) routine, well-documented issues with a known fix — reset your password, check
your library, enable UPnP — and (b) issues where the *cost of a wrong autonomous
action* is high: refunds, hardware failures, anything involving a customer who's
already unhappy. I treated the system's job as **triage + drafting**, not
**autonomous resolution**: the win condition is "route correctly, draft a reply
that's actually grounded in what this brand has said before, and know when to get
out of the way," not "resolve 100% of tickets with no human in the loop." A support
agent that confidently and wrongly closes a $250 refund dispute is worse than no
agent at all — so the escalation decision is treated as at least as important as
reply quality, and is optimized to fail *safe* (over-escalate) rather than fail
cheap (auto-handle everything).

**What I chose not to build:**
- **No fully autonomous send.** Every "auto_handle" output is still a *draft* a
  human/automation layer sends; the system never claims to have taken an action
  (issued a refund, unlocked an account) — only to have classified, drafted, and
  routed.
- **No 77-way Banking77-style intent taxonomy.** I used 8 intents (see
  `src/intents.py`) instead of a fine-grained one. Banking77 is designed for a
  banking product surface where sub-intents genuinely need different handling
  (dozens of specific card/transfer flows); for a single console-support inbox,
  most of that granularity would collapse to the same routing/reply behavior. I
  used Banking77-style intents only as a *reference point* for taxonomy granularity,
  not as the label set.
- **No dense/embedding retrieval, no fine-tuned generation model.** Both are
  reasonable v2 upgrades (see §5) but weren't needed to prove the core idea and
  would have cost the "reproducible in 15 minutes, no API key" property.
- **No conversation memory across multiple contacts** beyond a single thread's
  immediate follow-up. Real repeat-contact detection ("this is their 4th ticket
  this month across different threads") would need a customer-level ticket
  history I don't have in this dataset shape.

---

## 2. Results vs. two baselines

| System | Intent Acc | Intent Macro-F1 | Escalation F1 | Unsafe Auto-Handle Rate* | Reply Quality (judge, 1–5) |
|---|---|---|---|---|---|
| **Trivial** (majority-class intent, always escalates) | 0.173 | 0.037 | 0.398 | **0.000** | n/a — never drafts a reply |
| **Simple** (keyword rules, fixed template reply, never escalates except safety words) | 0.887 | 0.889 | **0.000** | **1.000** | 2.93 |
| **Full agent** (TF-IDF+LogReg classifier, retrieval-grounded reply, rule-based escalation) | **0.895** | 0.878 | **0.718** | **0.152** | **4.58** |

\*Unsafe auto-handle rate = of the examples that *should* have been escalated
(per the golden label), what fraction did the system wrongly auto-handle? This is
the number I care about most and is explained further in §4.

**Reading this table honestly:**
- The full agent's *intent accuracy* (89.5%) is barely better than the simple
  keyword-rule baseline (88.7%) — on its own, that would look like classification
  barely matters. It does matter, but not for the reason a leaderboard number
  suggests (see §4).
- The real separation between "full agent" and "simple baseline" is the
  **escalation column**: the simple baseline auto-handles *everything* except
  literal self-harm language, which means it would have auto-sent a reply to
  100% of the cases a human reviewer said needed escalation (refunds over $50,
  hardware complaints, angry repeat contacts, safety-adjacent language, weak-
  grounding cases). The full agent still gets 15.2% of those wrong, but that's a
  6.6x improvement in the number that actually protects customers and the brand.
- The trivial baseline's 0% unsafe-auto-handle rate is trivially achieved by
  escalating everything — it's the "safe but useless" corner of the tradeoff space,
  included specifically so the full agent's number means something relative to
  both corners (useless-but-safe vs. useful-but-unsafe).

---

## 3. Failure analysis — top 5 failure modes

All examples below are real rows from `eval_results/predictions.csv`.

**1. Meta-signal (repeat contact / anger) buried inside an on-topic message gets
missed by both the classifier and the escalation rule.**
> *"my PSN account got locked for no reason, this is the 3rd time this month, ticket
> 718213"* → classified `account_login_access` (correct topic), confidence high
> enough, grounding strong → **auto_handle**. Should have escalated.

The intent classifier correctly identifies the *topic* (account access) but the
*meta-signal* ("3rd time this month") that should trigger `ANGER_REPEAT` escalation
lives in `src/escalation.py`'s `ANGER_PATTERN` regex, which doesn't include ordinal
repeat-contact phrasing ("3rd time," "second time today") — only explicit anger
words ("furious," "unacceptable"). The weak-labeler used for *training* the
classifier does catch "3rd time" as a `complaint_repeat_escalation` signal, but
that pattern was never copied into the *escalation* rule itself. **Root cause: the
repeat-contact signal is encoded in two different places (classifier training
data vs. escalation regex) and they drifted out of sync.** This produced 5/133
(3.8%) of the unsafe auto-handles in this eval run.

**2. Duplicated sentences in ~11% of grounded replies (a real generation bug).**
> Message: *"is the store down rn or is it just me"* → Reply: *"Nothing showing on
> our status page currently, but please DM your region so we can check for
> localized issues. Nothing showing on our status page currently, but please DM
> your region so we can check for localized issues."*

15/133 (11.3%) of test replies contain a literal duplicated sentence. Root cause:
`reply_generator._local_generate` blends the top-2 retrieved matches when the best
match's similarity is below 0.25 — but when two retrieved historical pairs happen
to have *identical* agent replies (common here because several near-duplicate
customer messages in the training data got the same canned agent response), the
"blend" concatenates the same sentence with itself. This is a data-generation
artifact of the bundled sample dataset as much as a code bug, but the underlying
code bug (no dedup check before concatenating) would also bite on the real Kaggle
data, where agents do reuse near-identical phrasing across tickets.

**3. Off-topic / out-of-taxonomy messages get force-fit into the nearest intent,
then confidently mishandled.**
> *"can you tell me who won the world cup"* → classified `network_connectivity`
> (0.22 confidence, correctly low, correctly escalated) — but a similar case,
> *"im a games journalist working on a piece about psn outages, can someone from
> comms reach out"*, got force-fit into `refund_billing`/`network_connectivity`
> across different runs with a reply about "intermittent sign-in issues," entirely
> missing that this is a press inquiry, not a support ticket at all.

There is no "out of scope / other" bucket in the 8-intent taxonomy, so every
message — including ones that aren't actually support requests — gets mapped to
the closest of 8 buckets. The low-confidence escalation rule catches most of these,
but not reliably (see failure mode 4).

**4. Confidence threshold catches some but not all low-signal messages.**
> *"help"* → `how_to_general_inquiry`, confidence 0.335 → escalated (below the
> 0.45 threshold) ✓. But *"this might be a dumb question but can i play ps4 discs
> on a ps5 digital edition"* → misclassified as `technical_bug_error`, confidence
> 0.23, still escalated ✓ — correctly escalated but for the wrong intent, so if a
> human reviewer trusted the routing label to triage their own queue, they'd start
> in the wrong place.

Confidence-based escalation protects against acting on a bad classification, but
doesn't fix the classification itself — a human picking up an escalated ticket
still sees a wrong `agent_intent` label in the UI today. This is a real UX gap:
low-confidence escalations should probably surface the full probability
distribution (already computed, just not surfaced) rather than a single
best-guess label.

**5. "Positive/resolved" follow-ups are indistinguishable from active complaints
under the current taxonomy.**
> *"just wanted to say the new firmware update fixed my drift issue, thanks!"* →
> classified `hardware_malfunction` (or occasionally misclassified into an
> unrelated intent) → **always escalates**, because `hardware_malfunction` is a
> blanket always-escalate intent for cost/warranty-exposure reasons.

This is a case where the escalation *policy*, not the classifier, is the problem:
a thank-you message about a resolved hardware issue still trips the "hardware
issues always escalate" rule, wasting a human reviewer's time on a message that
needs no action. Fixing this needs a lightweight resolved-vs-active detector, which
doesn't currently exist (flagged as a next step, §5).

---

## 4. What is misleading about my headline number

The number most likely to get quoted from this report is **"full agent: 89.5%
intent accuracy, 0.72 escalation F1."** Here's why that's not the whole story:

1. **The simple keyword-rule baseline gets 88.7% intent accuracy — almost the same
   number — for a completely different reason: my golden set's "stratified" half
   was built by sampling messages the *same style of keyword rule* (the
   weak-labeler in `data_gen/weak_labeling.py`) had already labeled, then
   hand-confirming. That means roughly half the test set is, by construction,
   easy for a keyword classifier to get right, because the labels themselves
   were seeded from keyword matches.** The 89.5% vs 88.7% gap looks small because
   the benchmark is partly measuring "how well do you match your own labeling
   heuristic," not "how well do you handle language a keyword list wouldn't
   catch." The hand-authored stress examples (the other ~28 of 133 test rows) are
   the more honest signal, and that's exactly where the full agent's advantage is
   real but the absolute numbers are much less flattering (several of the failure
   modes in §3 are drawn from that subset).

2. **The reply-quality score of 4.58/5 is from the heuristic judge, not a real
   LLM judge, and this run's own calibration check shows the heuristic judge is a
   weak proxy.** Running `python -m src.eval.human_agreement` against 30
   hand-scored real outputs gives:

   | Dimension | Pearson r (judge vs. human) | % within 1 point |
   |---|---|---|
   | groundedness | undefined (heuristic score nearly constant) | 0.87 |
   | helpfulness | 0.32 | 0.70 |
   | tone | -0.01 | 0.90 |
   | correctness | undefined (heuristic score nearly constant) | 0.90 |
   | **overall** | **0.18** | 0.87 |

   An overall correlation of 0.18 between the heuristic judge and a human is weak
   — high "% within 1 point" mostly reflects that both the heuristic and my human
   scores cluster in the 4–5 range for a system that's mostly working, not that
   they agree on *which* examples are good vs. bad. **The honest conclusion is:
   the 4.58 headline reply-quality number should not be trusted as a reply-quality
   metric on its own** — it mainly reflects that the heuristic's "does this
   contain a DM-us next step and polite words" proxy fires on most replies,
   whether or not they're actually the *right* reply. The clearest real evidence
   of reply quality in this repo is the qualitative failure analysis in §3
   (duplicated sentences, off-topic replies), not the 4.58 number. If you set
   `ANTHROPIC_API_KEY`, the harness automatically switches to the real LLM judge
   (see `src/eval/llm_judge.py`), which is the version worth trusting — I did not
   have a key available in the environment this was built in, so I'm reporting
   the heuristic's honest limitations rather than a number I can't stand behind.

3. **The escalation F1 of 0.718 is computed on a test set where only 25% of
   examples should escalate.** A system that escalates too *much* is penalized
   softly by F1 (precision drops, but the business cost of over-escalation — more
   human review load — isn't captured by F1 at all). The metric that actually
   matters operationally, `unsafe_auto_handle_rate` (15.2%), is reported
   separately for exactly this reason, and it's arguably the more important
   number even though it's not the "prettiest" one at the top of the table.

4. **89.5% intent accuracy includes intents the classifier never has to
   distinguish from each other in a way that matters.** `refund_billing` and
   `complaint_repeat_escalation` are semantically close (both often involve money
   and frustration), and several genuine "the classifier got the exact intent
   wrong" cases (see §3, failure mode 1) still landed on a *correct escalation
   decision* by different means, so the headline accuracy number understates how
   often the wrong intent is silently recovered from downstream, and
   simultaneously the escalation number doesn't reveal how often that recovery
   happens by accident (a confidence threshold catching a low-confidence *wrong*
   guess) versus by design.

---

## 5. What I'd do next with one more week

Roughly in priority order:

1. **Fix the escalation/classifier signal drift (§3 failure mode 1).** Share a
   single source of truth for "repeat contact" and "anger" phrasing between
   `data_gen/weak_labeling.py` and `src/escalation.py` instead of two regex lists
   that can silently diverge. This is the highest-value, lowest-effort fix in the
   repo.
2. **Dedup the knowledge base before retrieval blending (§3 failure mode 2).**
   Collapse near-duplicate `(customer_text, agent_text)` pairs before building the
   TF-IDF index, and add a check in `reply_generator._local_generate` that never
   concatenates two retrieved texts that are identical or near-identical.
3. **Get a real LLM judge run with a proper agreement study.** Score the same 30
   calibration examples with `ANTHROPIC_API_KEY` set, and expand the calibration
   set to 50-75 examples stratified by intent and by auto-handle vs. escalate, so
   the correlation numbers in §4 are estimated more precisely (n=30 is barely
   enough to detect anything except a very strong or very weak correlation).
4. **Add an "out of scope / other" intent** (§3 failure mode 3) so
   off-topic messages get a real bucket instead of being force-fit into one of 8
   support intents, and route that bucket to a distinct "not a support ticket"
   handling path instead of the generic escalation flow.
5. **Resolved-vs-active detection** (§3 failure mode 5) — a lightweight
   classifier or rule (e.g., presence of "thanks"/"fixed"/"appreciate" plus
   absence of a new ask) to stop `hardware_malfunction`'s blanket always-escalate
   rule from firing on thank-you messages.
6. **Swap TF-IDF retrieval for a small sentence-embedding model** for the
   knowledge base, and validate whether it meaningfully changes the grounding
   quality on the harder stress-test examples specifically (not just the
   stratified sample, per the §4 caveat about what that sample can and can't
   tell you).
7. **Run everything against the real ~3M-row Kaggle dataset**, not just the
   bundled synthetic sample, and re-derive the golden set's stratified half from
   real customer language instead of templated text — the templated sample data
   is good enough to prove the pipeline works end-to-end, but the real dataset's
   noise (actual typos, actual multi-brand crosstalk, actual thread structure
   edge cases) will surface failure modes this report can't.
