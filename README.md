# Hiver SDE Intern Take-Home — AI Support Agent for `@AskPlayStation`

An AI support agent built from the *Customer Support on Twitter* dataset shape,
for a single brand (`@AskPlayStation`), that:

1. Classifies each incoming customer message into one of 8 intents.
2. Drafts a reply **grounded in retrieved historical resolutions** the brand has
   actually used before (not generated from scratch).
3. Decides **auto-handle vs. escalate to a human**, with a one-sentence stated reason.

See **[reports/REPORT.md](reports/REPORT.md)** for the full write-up (problem
framing, baseline comparison, failure analysis, and — most importantly — a section
on what's misleading about the headline numbers), and
**[reports/decision_log.md](reports/decision_log.md)** for 15 non-obvious decisions
and why.

---

## TL;DR — reproduce the headline results in under 15 minutes

```bash
git clone <this-repo>
cd hiver-support-agent
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# 1. Generate the bundled sample dataset (schema-identical to the real Kaggle file,
#    see "Using the real dataset" below to swap in the real ~3M-row file instead)
python data_gen/generate_sample_data.py

# 2. Build the golden evaluation set (156 hand-labeled examples; see data/golden/)
python data_gen/build_golden_set.py

# 3. Train the intent classifier + build the retrieval knowledge base (no API key needed)
python -m src.pipeline train

# 4. Run the full eval harness: agent + 2 baselines + automated metrics + judge scores
python -m src.eval.run_eval

# 5. (optional) Check how well the reply-quality judge agrees with a human
python -m src.eval.human_agreement

# 6. Try it interactively on a few example messages
python scripts/run_demo.py
```

All of the above is CPU-only, needs no API key, and takes well under 15 minutes on
a laptop (steps 1–4 together take under a minute in practice; most of the 15-minute
budget is `pip install`).

Headline numbers from the last run of step 4 (also in `reports/REPORT.md` and
`eval_results/summary.json`):

| System | Intent Acc | Escalation F1 | Unsafe Auto-Handle Rate |
|---|---|---|---|
| Trivial baseline | 0.173 | 0.398 | 0.000 |
| Simple baseline | 0.887 | 0.000 | 1.000 |
| **Full agent** | **0.895** | **0.718** | **0.152** |

*(Read §2 and especially §4 of the report before quoting these — the intent-accuracy
gap between the full agent and the simple baseline looks small for a reason that's
explained there, and it's not "classification barely matters.")*

---

## Repo layout

```
data_gen/                  # scripts that GENERATE data (not the data itself, though
                            # outputs are checked in for reproducibility)
  generate_sample_data.py  # synthetic-but-schema-faithful sample twcs-style dataset
  weak_labeling.py          # keyword-rule weak labeler, used for classifier training data
  build_golden_set.py       # builds the 156-example hand-labeled golden eval set

data/
  raw/                     # put the real Kaggle twcs.csv here if you have it (gitignored)
  sample/                  # bundled synthetic sample dataset (generated, checked in)
  golden/                  # golden_eval_set.csv — the hand-labeled eval set

src/
  data_prep.py              # loads raw data, filters to brand, reconstructs threads
  intents.py                # intent taxonomy + TF-IDF/LogReg classifier
  knowledge_base.py         # TF-IDF retrieval over historical resolutions
  reply_generator.py        # grounded reply drafting (local + optional LLM rewrite)
  escalation.py             # auto-handle vs. escalate rules, with stated reasons
  pipeline.py                # SupportAgent glue class + `train` CLI
  eval/
    baselines.py             # trivial + simple baselines
    metrics.py                # intent + escalation automated metrics
    llm_judge.py               # LLM-as-judge rubric (+ heuristic fallback)
    human_agreement.py         # judge-vs-human agreement calibration check
    run_eval.py                 # main eval entry point

reports/
  REPORT.md                 # the required report (problem framing, results, failure
                             # analysis, "misleading number" section, next steps)
  decision_log.md           # 15 non-obvious decisions and why

scripts/run_demo.py         # quick interactive demo
tests/test_pipeline.py      # pytest sanity tests
```

---

## Using the real Kaggle dataset instead of the bundled sample

This environment couldn't reach Kaggle to download the real
`thoughtvector/customer-support-on-twitter` dataset (it's behind a login), so the
repo ships with `data_gen/generate_sample_data.py`, which generates a smaller
dataset with the **exact same columns** as the real file
(`tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id`). If you have the real file:

```bash
# download twcs.csv from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
mkdir -p data/raw
cp /path/to/twcs.csv data/raw/twcs.csv

# everything else is unchanged — src/data_prep.py automatically prefers the real
# file over the sample if it's present:
python -m src.pipeline train
python -m src.eval.run_eval
```

You may also want to rebuild the golden set from the real data
(`python data_gen/build_golden_set.py`) so the stratified half reflects real
customer language rather than the bundled templates — see
`reports/decision_log.md` item 7 for the labeling methodology this script follows.

## Using a real LLM instead of the local/heuristic defaults

Everything above runs with **zero API keys**, by design (see decision log item 2
and 11). If you set:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

then:
- `src/reply_generator.py` will use the LLM to rewrite the retrieval-grounded
  draft more naturally (still constrained to only use retrieved facts).
- `src/eval/llm_judge.py` will use the real LLM-as-judge rubric instead of the
  heuristic proxy scorer, and `src/eval/run_eval.py` / `human_agreement.py` will
  report which backend was used.

`pip install anthropic` if you want this path (not in `requirements.txt` by
default, again to keep the zero-key path dependency-light).

## Running tests

```bash
python -m src.pipeline train   # tests need trained artifacts
pytest tests/ -q
```
