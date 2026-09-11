"""
Loads the Customer Support on Twitter dataset (real Kaggle twcs.csv, if present, else
the bundled synthetic sample), filters to a single brand, and reconstructs
(customer_message -> agent_resolution) pairs by following the
in_response_to_tweet_id / response_tweet_id links.

Usage:
    from src.data_prep import load_brand_pairs
    df = load_brand_pairs(brand="AskPlayStation")
    # df columns: customer_text, agent_text, customer_tweet_id, agent_tweet_id,
    #             followup_text (or None), thread_resolved (heuristic bool)
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REAL_DATA_PATH = ROOT / "data" / "raw" / "twcs.csv"
SAMPLE_DATA_PATH = ROOT / "data" / "sample" / "askplaystation_tweets.csv"

# Heuristic phrases that suggest a thread was NOT resolved by the agent's reply
# (used only as a weak signal for the "resolved" flag on the knowledge base / for
# the escalation training features — never used as ground truth; ground truth
# resolution status is not labeled in the raw data, see decision log item 5).
UNRESOLVED_SIGNALS = re.compile(
    r"\b(still (not|doesn'?t|isn'?t)|didn'?t (help|work)|same (issue|problem)|"
    r"no (update|response)|not (working|fixed)|worse|unacceptable|useless)\b",
    re.IGNORECASE,
)


def _load_raw() -> pd.DataFrame:
    if REAL_DATA_PATH.exists():
        path = REAL_DATA_PATH
    elif SAMPLE_DATA_PATH.exists():
        path = SAMPLE_DATA_PATH
    else:
        raise FileNotFoundError(
            "No dataset found. Either place the real Kaggle file at "
            f"{REAL_DATA_PATH}, or run `python data_gen/generate_sample_data.py` "
            f"to create {SAMPLE_DATA_PATH}."
        )
    df = pd.read_csv(path, dtype=str)
    # normalize dtypes across real vs sample files
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    for col in ("response_tweet_id", "in_response_to_tweet_id"):
        df[col] = df[col].replace({"": None, "nan": None})
    return df


def load_brand_pairs(brand: str = "AskPlayStation") -> pd.DataFrame:
    """Returns one row per (customer message, brand's first reply to it), plus any
    immediate customer follow-up message and a heuristic 'resolved' flag."""
    df = _load_raw()

    # Index by tweet_id for fast lookup
    df = df.set_index("tweet_id", drop=False)

    # Agent (outbound) tweets from the brand, replying to a customer tweet
    agent_mask = (~df["inbound"]) & (df["author_id"] == brand) & (df["in_response_to_tweet_id"].notna())
    agent_rows = df[agent_mask]

    records = []
    for _, agent_row in agent_rows.iterrows():
        cust_id = agent_row["in_response_to_tweet_id"]
        if cust_id not in df.index:
            continue
        cust_row = df.loc[cust_id]
        if isinstance(cust_row, pd.DataFrame):  # duplicate ids, guard
            cust_row = cust_row.iloc[0]
        if not cust_row["inbound"]:
            continue  # only keep genuine customer->agent pairs

        # look for an immediate customer follow-up replying to this agent tweet
        followup_mask = (df["inbound"]) & (df["in_response_to_tweet_id"] == agent_row["tweet_id"])
        followups = df[followup_mask]
        followup_text = followups.iloc[0]["text"] if len(followups) else None

        resolved = True
        if followup_text and UNRESOLVED_SIGNALS.search(str(followup_text)):
            resolved = False
        elif followup_text is not None:
            resolved = False  # any follow-up at all is treated as a weak "not fully resolved" signal

        records.append({
            "customer_tweet_id": cust_row["tweet_id"],
            "agent_tweet_id": agent_row["tweet_id"],
            "customer_text": cust_row["text"],
            "agent_text": agent_row["text"],
            "followup_text": followup_text,
            "thread_resolved": resolved,
        })

    out = pd.DataFrame.from_records(records)
    out = out.drop_duplicates(subset=["customer_tweet_id"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    pairs = load_brand_pairs()
    print(f"Loaded {len(pairs)} customer->agent resolution pairs")
    print(pairs.head(3).to_string())
    print("Resolved rate (heuristic):", pairs["thread_resolved"].mean().round(3))
