# Decision Log

Non-obvious decisions made while building this, and the reasoning. Most cross-reference
comments in the code itself (search the same keywords) — this is the condensed version.

1. **Chose `@AskPlayStation` over a more commonly-picked brand (AmazonHelp, AppleSupport,
   comcastcares).** Gaming-console support has a good spread of intent types (account,
   billing, technical, hardware, how-to) without the ambiguity of e-commerce order-tracking
   flows, and hardware issues give a clean example of a "should always escalate regardless
   of confidence" policy to demonstrate.

2. **Built a synthetic-but-schema-faithful sample dataset instead of only supporting the
   real Kaggle file.** The real dataset needs a Kaggle login this environment can't reach.
   Rather than leave the repo non-runnable out of the box, `data_gen/generate_sample_data.py`
   produces a smaller dataset with the *exact same columns* as the real `twcs.csv`, so
   `src/data_prep.py` works unmodified against either. If you drop the real file at
   `data/raw/twcs.csv`, everything downstream (classifier training, KB, eval) uses it
   automatically with zero code changes — see README "Using the real dataset."

3. **8 intents, not a Banking77-style 77-way taxonomy.** For one brand's inbox, a much
   finer taxonomy would mostly split intents that get identical downstream handling
   (routing + reply strategy). 8 was chosen by grouping recurring issue types by
   "does this need a different reply strategy or escalation policy," not by surface
   topic alone — e.g., "controller drift" and "console won't turn on" are both
   `hardware_malfunction` because both get the same always-escalate treatment, even
   though they're different symptoms.

4. **Classifier trained on weak (keyword-rule) labels + a small folded-in slice of the
   hand-labeled golden set, not thousands of hand-labeled examples.** Hand-labeling
   scales linearly with time spent; I judged that time was better spent making the
   150-250 *evaluation* labels trustworthy (with rationale, stratified + adversarial
   sampling) than hand-labeling a much larger, less scrutinized training set. This is
   standard weak supervision, and it's flagged explicitly in the report's "misleading
   headline number" section because it does create a benchmark-leakage risk I don't
   want to hide.

5. **No "thread resolved" ground truth exists in the raw data, so I used a heuristic
   (presence and content of a customer follow-up) rather than treating it as reliable
   labels.** It's used only as a weak feature for the knowledge base's match metadata,
   never as ground truth anywhere in the eval harness — the golden set's `true_escalate`
   labels are the actual ground truth, and those were hand-set.

6. **TF-IDF retrieval instead of dense embeddings for the knowledge base.** Sparse
   lexical retrieval is a strong, auditable baseline for short, jargon-heavy support
   text (error codes, product names) and needs no model download or API key — directly
   serves the 15-minute reproducibility requirement. Flagged as the first retrieval
   upgrade to try next.

7. **Golden set provenance is "one experienced-enough labeler with written rationale
   per example," not a multi-annotator adjudication process.** For 150-250 examples,
   I judged that an annotation-UI + multiple-annotator setup would cost more time than
   it returned in label quality at this scale; instead every example carries a
   `rationale` column so a reviewer can audit *why* a label was assigned and disagree
   with a specific one, rather than trusting an aggregate inter-annotator-agreement
   number computed once and never revisited.

8. **Escalation logic is explicit rules on top of model outputs, not a second learned
   model.** A support-escalation decision needs a one-sentence, auditable "why" a human
   reviewer (or Hiver's own reviewers) can immediately sanity-check and tune without
   retraining anything. This trades a small amount of possible recall for a large amount
   of transparency and easy tunability (see `CONFIDENCE_THRESHOLD`, `GROUNDING_THRESHOLD`,
   `REFUND_AMOUNT_THRESHOLD` — all single constants at the top of `src/escalation.py`).

9. **`hardware_malfunction` and refunds above $50 always escalate, regardless of
   classifier confidence or grounding quality.** This is a deliberate business-policy
   choice, not a model limitation: even a highly-confident, well-grounded hardware/large-
   refund reply carries cost/warranty/legal exposure that I judged shouldn't be
   auto-sent by this version of the system. It does mean the escalation rate is higher
   than a pure confidence-based policy would produce — treated as a feature, not a bug,
   for this report.

10. **The reply generator supports an optional LLM rewrite step but defaults to fully
    local generation.** Every reply's grounding is checked against retrieved historical
    replies either way; the LLM step (`ANTHROPIC_API_KEY`-gated) only changes phrasing
    quality, not what facts are allowed into the reply — both modes return the same
    `GeneratedReply` object so the eval harness treats them identically.

11. **The LLM-as-judge deliverable ships with a heuristic fallback, and the report
    explicitly does NOT trust the heuristic's headline number.** I judged it more
    honest to ship a working, low-cost proxy judge plus a real (if weak, n=30)
    agreement-with-human study showing its limits, than to either (a) require an API
    key to run the eval harness at all, or (b) present heuristic scores as if they were
    LLM-judge-quality without caveat.

12. **`unsafe_auto_handle_rate` is reported as a first-class metric, not folded into F1.**
    Escalation F1 alone doesn't distinguish "we escalate too much" (costly but safe) from
    "we escalate too little" (cheap but risky) in a way that matches how a support org
    would actually care about the tradeoff. This custom metric — "of the cases that
    should escalate, how many did we wrongly auto-handle" — is the one I'd want a
    reviewer to look at first.

13. **The trivial baseline always escalates (rather than, say, always auto-handling).**
    "Always escalate" is the trivial baseline a real support team could actually run
    today without any AI system at all (every message reviewed by a human) — it's the
    more meaningful zero-effort comparison point than "always auto-handle everything,"
    which no real support org would accept as a starting policy.

14. **The golden set has an explicit `train`/`test` split column (15%/85%) rather than
    being purely held-out.** A small slice is folded into classifier training specifically
    to anchor the intents that are otherwise mostly weak-labeled (see decision 4);
    everything the eval harness reports is computed only on the `test` split, and this
    is called out directly in the report's misleading-number section rather than left
    implicit.

15. **Chose to write out real, run-derived numbers in the report rather than
    representative/illustrative ones.** Every number in `reports/REPORT.md` comes from
    an actual `eval_results/summary.json` / `predictions.csv` produced by running the
    pipeline in this repo — including the less flattering ones (0.18 judge-human
    correlation, 11.3% duplicated-sentence rate). The report is written to be
    re-verifiable by re-running `python -m src.eval.run_eval`, not just descriptive.
