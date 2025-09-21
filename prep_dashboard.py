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
    if key in f.columns:
        f = f.drop_duplicates(subset=[key])
    else:
        # if features don't have the key (unusual), fall back to p as-is
        f = f.copy()

    # Merge (bring across area_pct for dashboard)
    merge_cols = [c for c in [key, "area_pct"] if c in f.columns]
    m = p.merge(f[merge_cols], on=key, how="left") if key in p.columns and key in f.columns else p.copy()

    # ----- Derived columns -----
    prob_cols = [c for c in m.columns if c.startswith("prob_")]
    if len(prob_cols) > 0:
        m["confidence"] = m[prob_cols].max(axis=1)
    else:
        m["confidence"] = np.nan

    # Similarities
    m["prime_similarity"] = m["prob_Prime"] if "prob_Prime" in m.columns else np.nan
    m["wagyu_similarity"] = m["prob_Wagyu"] if "prob_Wagyu" in m.columns else np.nan

    # Lean %
    if "area_pct" in m.columns:
        m["meat_pct"] = 100.0 - m["area_pct"]

    # ----- Nice column order for the dashboard -----
    front = [c for c in [
        "image", "path", "label", "predicted",
        "prime_similarity", "wagyu_similarity", "confidence",
        "area_pct", "meat_pct",
        # keep common prob_* up front if present
        "prob_Select", "prob_Choice", "prob_Prime", "prob_Wagyu"
    ] if c in m.columns]

    # Append any remaining columns (including any other prob_* dynamically)
    remaining = [c for c in m.columns if c not in front]
    # Optionally bring other prob_* (if any) next, in alphabetical order
    other_probs = sorted([c for c in remaining if c.startswith("prob_")])
    # Ensure we don't duplicate
    remaining = [c for c in remaining if c not in other_probs]
    cols = front + other_probs + remaining
    m = m[cols]

    # ----- Rounding for display -----
    round_cols = ["prime_similarity", "wagyu_similarity", "confidence", "area_pct", "meat_pct"] + [c for c in m.columns if c.startswith("prob_")]
    for c in round_cols:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").round(3)

    # Sort for convenience (Pred within class by prime similarity descending if present)
    if "prime_similarity" in m.columns and "predicted" in m.columns:
        m = m.sort_values(["predicted", "prime_similarity"], ascending=[True, False])

    m.to_csv(args.out, index=False)
    print(f"Saved {args.out} with {len(m)} rows. Join key = {key}")

if __name__ == "__main__":
    main()
