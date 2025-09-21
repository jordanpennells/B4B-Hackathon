#!/usr/bin/env python3
"""
Compute a Marbling Index (MI) for each image and visualise distributions by grade.

Inputs:
  - features.csv from marbling_mvp.py (must include: area_pct, orientation_dispersion, lacunarity_mean)
  - Optional columns (if present): label, label_source, relpath, image

Outputs:
  - mi_by_image.csv  (per-image MI + normalised components)
  - mi_by_grade.png  (box/violin plot of MI by grade)
  - mi_summary.json  (per-grade summary stats)

Usage examples:
  python compute_mi_and_plot.py --features features.csv
  python compute_mi_and_plot.py --features features.csv --w 0.6 0.3 0.1
"""

import argparse, os, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Defaults ----------
DEFAULT_WEIGHTS = (0.6, 0.3, 0.1)   # (area_pct, alignment, lacunarity)
MI_FEATURES = {
    "area": "area_pct",
    # alignment is (1 - orientation_dispersion) after normalisation
    "orientation_dispersion": "orientation_dispersion",
    "lacunarity": "lacunarity_mean"
}
VALID_GRADES = ["Select", "Choice", "Prime", "Wagyu"]

# ---------- Helpers ----------
def _norm01(x, lo=None, hi=None):
    x = np.array(x, dtype=np.float64)
    if lo is None: lo = np.nanmin(x)
    if hi is None: hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # fallback to safe bounds
        lo, hi = np.nanmin(x), np.nanmax(x) + 1e-9
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0), (float(lo), float(hi))

def infer_wagyu_row(row):
    """
    Decide if this row should be grouped as 'Wagyu'.
    Priority:
      - label_source contains 'alias:wagyu'
      - image/relpath/label contains 'wagyu' (case-insensitive)
    Otherwise, keep original label.
    """
    def has_wagyu(s):
        return isinstance(s, str) and ("wagyu" in s.lower())

    # label_source signal (best when using folder run)
    if has_wagyu(row.get("label_source", None)):
        return True
    # filename or relpath clues
    for key in ("image", "relpath", "path"):
        if has_wagyu(row.get(key, None)):
            return True
    # label text itself
    if has_wagyu(row.get("label", None)):
        return True
    return False

def make_grade_group(row):
    # Prefer explicit Wagyu override
    if infer_wagyu_row(row):
        return "Wagyu"
    # Else map known labels (title-case)
    lab = row.get("label", "")
    lab = str(lab).strip().title()
    if lab in {"Select","Choice","Prime"}:
        return lab
    return "Unknown"

def summarise_group(series):
    s = pd.Series(series.dropna().values, dtype=float)
    if s.empty:
        return {"n": 0}
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "median": float(s.median()),
        "p10": float(np.percentile(s, 10)),
        "p25": float(np.percentile(s, 25)),
        "p75": float(np.percentile(s, 75)),
        "p90": float(np.percentile(s, 90)),
    }

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Compute Marbling Index and plot distributions by grade.")
    ap.add_argument("--features", required=True, help="Path to features.csv from marbling_mvp.py")
    ap.add_argument("--weights", "--w", nargs=3, type=float, default=DEFAULT_WEIGHTS,
                    help="Weights for (area_pct, alignment(1-OD), lacunarity_mean). Default: 0.6 0.3 0.1")
    ap.add_argument("--out-prefix", default="", help="Prefix for output files (default none)")
    ap.add_argument("--order", nargs="*", default=["Select","Choice","Prime","Wagyu"],
                    help="Order of grades on plot (default: Select Choice Prime Wagyu)")
    ap.add_argument("--drop-unknown", action="store_true", help="Drop rows whose grade group is 'Unknown'")
    args = ap.parse_args()

    W_area, W_align, W_lac = args.weights
    if not np.isclose(W_area + W_align + W_lac, 1.0):
        raise SystemExit("Weights must sum to 1.0")

    df = pd.read_csv(args.features)
    # basic column checks
    needed = [MI_FEATURES["area"], MI_FEATURES["orientation_dispersion"], MI_FEATURES["lacunarity"]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required feature columns in {args.features}: {missing}")

    # Grade grouping
    df["grade_group"] = df.apply(make_grade_group, axis=1)
    if args.drop_unknown:
        df = df[df["grade_group"] != "Unknown"].copy()
    if df.empty:
        raise SystemExit("No rows to process after filtering.")

    # Build normalised components
    # 1) area_pct → larger is more marbling
    area_raw = df[MI_FEATURES["area"]].astype(float)
    area_n, area_bounds = _norm01(area_raw)

    # 2) alignment = 1 - orientation_dispersion (so higher means more aligned webbing)
    od_raw = df[MI_FEATURES["orientation_dispersion"]].astype(float)
    od_n, od_bounds = _norm01(od_raw)  # normalise OD
    align_n = 1.0 - od_n               # invert to make 'higher is better marbling'

    # 3) lacunarity_mean → normalise (directionality can be dataset dependent; we’ll keep 'higher is more marbled' for now)
    lac_raw = df[MI_FEATURES["lacunarity"]].astype(float)
    lac_n, lac_bounds = _norm01(lac_raw)

    # Marbling Index (MI)
    MI = W_area * area_n + W_align * align_n + W_lac * lac_n
    df_out = df.copy()
    df_out["mi"] = MI
    df_out["mi_area_n"] = area_n
    df_out["mi_align_n"] = align_n
    df_out["mi_lac_n"] = lac_n

    # Save per-image MI table
    pref = args.out_prefix
    out_csv = f"{pref}mi_by_image.csv"
    df_out.sort_values("mi", ascending=False).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  (n={len(df_out)})")

    # Summary by grade
    order = [g for g in args.order if g in set(df_out["grade_group"])]
    # ensure any missing grades aren’t plotted but are summarised if present
    groups = df_out.groupby("grade_group")
    summary = {g: summarise_group(groups.get_group(g)["mi"] if g in groups.groups else pd.Series(dtype=float))
               for g in sorted(df_out["grade_group"].unique())}
    with open(f"{pref}mi_summary.json","w") as f:
        json.dump({
            "weights": {"area": W_area, "alignment": W_align, "lacunarity": W_lac},
            "normalisation_bounds": {
                "area_pct": {"min": area_bounds[0], "max": area_bounds[1]},
                "orientation_dispersion": {"min": od_bounds[0], "max": od_bounds[1]},
                "lacunarity_mean": {"min": lac_bounds[0], "max": lac_bounds[1]},
            },
            "per_grade": summary
        }, f, indent=2)
    print(f"Wrote {pref}mi_summary.json")

    # Plot: violin + box overlay
    data = [df_out.loc[df_out["grade_group"]==g, "mi"].values for g in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vp = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    # Light fill for violins
    for pc in vp['bodies']:
        pc.set_alpha(0.35)

    # Boxplot overlay (medians + IQR)
    bp = ax.boxplot(data, widths=0.15, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_alpha(0.6)

    # Aesthetics
    ax.set_xticks(np.arange(1, len(order)+1))
    ax.set_xticklabels(order)
    ax.set_ylabel("Marbling Index (0–1)")
    ax.set_title("MI distribution by grade")
    ax.grid(True, axis="y", alpha=0.25)

    out_png = f"{pref}mi_by_grade.png"
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_png}")

    # Also print a short console summary
    print("\nPer-grade MI summary:")
    for g in order:
        s = summary.get(g, {"n":0})
        if s["n"] == 0:
            print(f"  {g}: n=0")
        else:
            print(f"  {g}: n={s['n']}, median={s['median']:.3f}, mean={s['mean']:.3f}, p25–p75=({s['p25']:.3f}–{s['p75']:.3f})")

if __name__ == "__main__":
    main()
