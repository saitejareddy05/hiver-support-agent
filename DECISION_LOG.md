# Decision log

- Chose AmazonHelp as the default brand because it has a large support footprint while remaining configurable.
- Filtered by support-account author rather than assuming a brand column exists in TWCS.
- Reconstructed threads using parent and response tweet IDs so retrieval sees resolution context.
- Excluded unresolved inbound tweets from the historical resolution index to avoid unsupported advice.
- Used TF-IDF clustering as an inspectable CPU-first intent discovery method.
- Kept cluster names derived from top terms and explicitly expects human renaming before submission.
- Used a lightweight logistic classifier for offline reproducibility and easy error analysis.
- Added optional LLM drafting rather than making an API mandatory for tests or demos.
- Stored prompts as text files so reviewers can inspect and edit them.
- Used retrieval evidence in the offline fallback so every demo reply has historical grounding.
- Escalated financial, legal, safety, abuse, privacy, and urgent keywords regardless of classifier confidence.
- Set the default confidence threshold to 0.60 as a conservative starting point, not a validated optimum.
- Added a deterministic stratified golden-set helper crossing intent, thread length, and risk language.
- Left truth labels to the human because inferred clusters are not ground truth.
- Reported LLM-judge calibration separately because judge scores are not an objective metric.
