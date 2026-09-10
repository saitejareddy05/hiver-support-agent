"""Load TWCS, filter one brand, and reconstruct conversation threads."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQUIRED = {"tweet_id", "author_id", "inbound", "text", "response_tweet_id", "in_response_to_tweet_id"}

def load_twcs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Download TWCS and place it there; see data/raw/README.md.")
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"TWCS CSV is missing columns: {sorted(missing)}")
    frame["text"] = frame["text"].fillna("").astype(str).str.replace(r"\\s+", " ", regex=True).str.strip()
    frame["inbound"] = frame["inbound"].astype(bool)
    return frame

def filter_brand(frame: pd.DataFrame, brand: str = "AmazonHelp") -> pd.DataFrame:
    # In TWCS, author_id identifies the support account and inbound=False identifies its replies.
    brand_rows = frame[frame["author_id"].astype(str).str.casefold() == brand.casefold()].copy()
    if brand_rows.empty and "author_id" in frame:
        brand_rows = frame[frame["author_id"].astype(str).str.contains(brand, case=False, na=False)].copy()
    return brand_rows

def reconstruct_threads(frame: pd.DataFrame, brand: str = "AmazonHelp") -> pd.DataFrame:
    rows = []
    by_id = frame.set_index("tweet_id", drop=False)
    replies = frame[(frame["inbound"]) & frame["in_response_to_tweet_id"].notna()].copy()
    for _, customer in replies.iterrows():
        chain = [str(customer["text"])]
        parent_id = customer["in_response_to_tweet_id"]
        seen = set()
        while pd.notna(parent_id) and parent_id not in seen and parent_id in by_id.index:
            seen.add(parent_id)
            parent = by_id.loc[parent_id]
            chain.append(str(parent["text"]))
            parent_id = parent["in_response_to_tweet_id"]
        response_id = customer["response_tweet_id"]
        if pd.isna(response_id) or response_id not in by_id.index:
            continue
        response = by_id.loc[response_id]
        if bool(response["inbound"]):
            continue
        rows.append({"tweet_id": customer["tweet_id"], "brand": brand, "message_text": customer["text"],
                     "thread_context": " || ".join(reversed(chain)), "resolution": response["text"],
                     "thread_length": len(chain) + 1, "resolved": True})
    return pd.DataFrame(rows).drop_duplicates("tweet_id")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/twcs.csv")
    parser.add_argument("--output", default="data/processed/amazonhelp_threads.csv")
    parser.add_argument("--brand", default="AmazonHelp")
    args = parser.parse_args()
    result = reconstruct_threads(filter_brand(load_twcs(args.input), args.brand), args.brand)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} resolved {args.brand} threads to {args.output}")

if __name__ == "__main__":
    main()
