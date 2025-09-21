# prep_dashboard.py
# Merge predictions + features and add dashboard columns.

import pandas as pd
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions.csv")
    ap.add_argument("--feat", default="features.csv")
    ap.add_argument("--out",  default="predictions_dashboard.csv")
    args = ap.parse_args()

    p = pd.read_csv(args.pred)
    f = pd.read_csv(args.feat)

    # Choose a join key: prefer 'path' if present in both, else 'image'
    if "path" in p.columns and "path" in f.columns:
        key = "path"
    else:
        key = "image"

    # Keep only one row per key in features (in case of duplicates)
    f = f.drop_duplicates(subset=[key])

    # Merge
    m = p.merge(f[[key, "area_pct"]], on=key, how="left")

    # Derived columns
    prob_cols = [c for c in m.columns if c.startswith("prob_")]
    m["confidence"] = m[prob_cols].max(axis=1)
    if "prob_Prime" in m.columns:
        m["prime_similarity"] = m["prob_Prime"]
    else:
        m["prime_similarity"] = np.nan

    if "area_pct" in m.columns:
        m["meat_pct"] = 100.0 - m["area_pct"]

    # Nice column order for the dashboard
    front = [c for c in ["image","path","label","predicted",
                         "prime_similarity","confidence",
                         "area_pct","meat_pct",
                         "prob_Select","prob_Choice","prob_Prime"] if c in m.columns]
    cols = front + [c for c in m.columns if c not in front]
    m = m[cols]

    # Optional: round for display
    for c in ["prime_similarity","confidence","area_pct","meat_pct","prob_Select","prob_Choice","prob_Prime"]:
        if c in m.columns:
            m[c] = m[c].astype(float).round(3)

    # Sort for convenience (Prime at top by similarity)
    if "prime_similarity" in m.columns:
        m = m.sort_values(["predicted","prime_similarity"], ascending=[True, False])

    m.to_csv(args.out, index=False)
    print(f"Saved {args.out} with {len(m)} rows. Join key = {key}")

if __name__ == "__main__":
    main()
