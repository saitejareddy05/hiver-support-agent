# Report

## Problem framing
For AmazonHelp, good means identifying the customer's actual support need, grounding a concise reply in resolved AmazonHelp conversations, and routing risky or uncertain cases to a human. This prototype does not build multilingual support, voice, real refunds, order-system access, or production API integration.

## Baselines
The trivial baseline predicts the majority intent and emits a generic template. The simple baseline trains TF-IDF plus logistic regression and uses a fixed template reply. The full agent adds brand-specific clustering, historical retrieval, optional LLM drafting, and conservative escalation policy.

## Results table

Run `python -m eval.run_eval` after replacing the starter golden worksheet with 150-250 human labels. Paste its JSON here:

| System | Intent accuracy | Macro F1 | Retrieval hit rate | Escalation F1 |
|---|---:|---:|---:|---:|
| Majority + generic | TBD | TBD | 0.00 | TBD |
| TF-IDF + template | TBD | TBD | TBD | TBD |
| Full agent | TBD | TBD | TBD | TBD |

## Failure analysis
Record five real examples from `run_eval.py` outputs, including the tweet and system output. Expected modes include ambiguous multi-intent messages, sparse historical evidence, policy keywords used metaphorically, cluster names that need human renaming, and LLM replies that overstate what history proves.

## What is misleading about my headline number?
The golden set can be biased by stratification and human labeling choices. LLM-judge agreement can be inflated when the same model family drafts and judges. Few-shot or retrieval overlap can leak near-duplicates into evaluation. This is brand-specific overfitting, not general customer-support intelligence, and 150-250 examples leave substantial small-sample noise. Retrieval hit rate is not the same as resolution correctness.

## What I'd do with one more week
Deduplicate near-identical tweets, add temporal holdout evaluation, rename clusters with a human review loop, compare embedding retrieval with TF-IDF, calibrate escalation probabilities, and audit replies for privacy leakage with an adversarial test set.
