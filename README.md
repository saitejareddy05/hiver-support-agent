# Hiver Support Agent

A runnable, brand-specific customer-support prototype for the Customer Support on Twitter dataset, defaulting to `AmazonHelp`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Download the Kaggle dataset `thoughtvector/customer-support-on-twitter` and place its CSV at `data/raw/twcs.csv`. The expected columns are documented in `data/raw/README.md`. Edit `config.yaml` to change the brand, model, threshold, or paths. `LLM_API_KEY` is optional; without it, the demo uses deterministic local drafting.

## Reproduce the headline workflow

```powershell
python -m src.data_prep --input data/raw/twcs.csv --brand AmazonHelp
python -m src.intent_taxonomy --input data/processed/amazonhelp_threads.csv --output data/processed/amazonhelp_intents.csv
python eval/build_golden_set.py --input data/processed/amazonhelp_intents.csv --output data/golden_set/golden_eval.csv --n 200
# Human-label the blank truth columns, then:
python -m eval.run_eval
```

On a laptop this CPU workflow is generally under 15 minutes after the dataset is downloaded. An optional OpenAI-compatible call adds API cost and latency per draft; set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` in `.env`.

## Offline demo and tests

```powershell
python scripts/run_demo.py
pytest -q
```

The demo uses `data/processed/sample.csv` and needs no download or API key. `tests/test_pipeline.py` covers classification, retrieval, drafting, and mandatory escalation behavior.

## Evaluation protocol

`eval/build_golden_set.py` creates a deterministic worksheet stratified by inferred intent, thread length, and negative/urgent language. Replace the blank truth fields through human review; inferred clusters are only a starting point. `eval/metrics.py` measures intent accuracy/macro-F1, retrieval hit rate, and escalation precision/recall/F1. `eval/llm_judge.py` contains the five-dimension 1-5 rubric, and `eval/judge_calibration.py` compares a human-scored subsample with quadratic Cohen's kappa.

Banking77 can optionally sanity-check the mechanics of intent classification, but its labels are not used as AmazonHelp ground truth.
