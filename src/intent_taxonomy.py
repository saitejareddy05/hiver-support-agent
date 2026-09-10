"""Discover brand-specific intents with TF-IDF clustering and inspectable names."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

DEFAULT_NAMES = ["delivery_tracking", "account_access", "payment_or_charge", "returns_and_refunds", "product_or_order_help", "other"]

def derive_taxonomy(frame: pd.DataFrame, n_clusters: int = 6, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    if frame.empty: return frame.assign(intent="other"), {"clusters": []}
    texts = frame["message_text"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    k = max(1, min(n_clusters, len(frame)))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(matrix)
    terms = vectorizer.get_feature_names_out()
    names = {}
    for cluster in range(k):
        scores = matrix[labels == cluster].mean(axis=0).A1
        top = scores.argsort()[-5:][::-1]
        names[cluster] = " / ".join(terms[i] for i in top) or DEFAULT_NAMES[cluster % len(DEFAULT_NAMES)]
    result = frame.copy()
    result["cluster_id"] = labels
    result["intent"] = result["cluster_id"].map(names)
    report = {"clusters": [{"cluster_id": i, "label": names[i], "count": int((labels == i).sum()),
                             "examples": result.loc[result.cluster_id == i, "message_text"].head(3).tolist()} for i in range(k)]}
    if k > 1 and len(frame) > k: report["silhouette"] = float(silhouette_score(matrix, labels))
    return result, report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/amazonhelp_threads.csv")
    parser.add_argument("--output", default="data/processed/amazonhelp_intents.csv")
    parser.add_argument("--clusters", type=int, default=6)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    labeled, report = derive_taxonomy(frame, args.clusters)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); labeled.to_csv(args.output, index=False)
    print(report)

if __name__ == "__main__": main()
