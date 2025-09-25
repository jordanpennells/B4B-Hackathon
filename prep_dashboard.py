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

    # ----- Join key: prefer 'path' if present in both, else 'image' -----
    if "path" in p.columns and "path" in f.columns:
        key = "path"
    elif "image" in p.columns and "image" in f.columns:
        key = "image"
    else:
        # Fallback: just attach what we can without merging (uncommon)
        key = None

    # Keep only one row per key in features (defensive)
    if key is not None and key in f.columns:
        f = f.drop_duplicates(subset=[key]).copy()

    # Bring across steak-aware and legacy metrics if present
    feat_cols_to_pull = [c for c in [
        key,
        "fat_within_steak_pct",
        "meat_within_steak_pct",
        "area_pct",
        "marbling_index",
        "mi_fat_component",
        "mi_align_component",
        "mi_lac_component",
    ] if c in f.columns]

    if key is not None and key in p.columns and key in f.columns:
        m = p.merge(f[feat_cols_to_pull], on=key, how="left")
    else:
        m = p.copy()

    # ----- Derived columns -----
    # Confidence = max prob_*
    prob_cols = [c for c in m.columns if c.startswith("prob_")]
    m["confidence"] = m[prob_cols].max(axis=1) if prob_cols else np.nan

    # Similarities
    m["prime_similarity"] = m["prob_Prime"] if "prob_Prime" in m.columns else np.nan
    m["wagyu_similarity"] = m["prob_Wagyu"] if "prob_Wagyu" in m.columns else np.nan

    # Steak-aware lean %
    if "fat_within_steak_pct" in m.columns and "meat_within_steak_pct" not in m.columns:
        m["meat_within_steak_pct"] = 100.0 - m["fat_within_steak_pct"]

    # Frame-based lean % (legacy)
    if "area_pct" in m.columns and "meat_pct" not in m.columns:
        m["meat_pct"] = 100.0 - m["area_pct"]

    # ----- Nice column order for the dashboard -----
    front = [c for c in [
        "image", "path", "label", "predicted",
        "confidence", "prime_similarity", "wagyu_similarity",
        "marbling_index",
        "fat_within_steak_pct", "meat_within_steak_pct",
        "area_pct", "meat_pct",
        "prob_Select", "prob_Choice", "prob_Prime", "prob_Wagyu",
    ] if c in m.columns]

    # Bring any other prob_* next in alphabetical order (without dupes)
    remaining = [c for c in m.columns if c not in front]
    other_probs = sorted([c for c in remaining if c.startswith("prob_")])
    remaining = [c for c in remaining if c not in other_probs]
    cols = front + other_probs + remaining
    m = m[cols]

    # ----- Rounding for display -----
    round_cols = [
        "confidence", "prime_similarity", "wagyu_similarity",
        "fat_within_steak_pct", "meat_within_steak_pct",
        "area_pct", "meat_pct",
        "marbling_index", "mi_fat_component", "mi_align_component", "mi_lac_component",
    ] + [c for c in m.columns if c.startswith("prob_")]

    for c in round_cols:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").round(3)

    # Sort for convenience: by predicted, then by prime_similarity (if present), else confidence
    if "predicted" in m.columns:
        if "prime_similarity" in m.columns:
            m = m.sort_values(["predicted", "prime_similarity"], ascending=[True, False])
        else:
            m = m.sort_values(["predicted", "confidence"], ascending=[True, False])

    m.to_csv(args.out, index=False)
    jk = key if key is not None else "(none — merge skipped)"
    print(f"Saved {args.out} with {len(m)} rows. Join key = {jk}")

if __name__ == "__main__":
    main()
