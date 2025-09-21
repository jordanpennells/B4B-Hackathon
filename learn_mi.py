# learn_mi.py
# Derive Marbling Index (MI) scores from features.csv in three ways:
# 1) Hand-crafted MI (your formula, using 3 features)
# 2) LDA-based MI (single discriminant axis, data-driven)
# 3) Logistic-derived MI (expected class rank from multinomial LR)
#
# Usage examples:
#   python learn_mi.py --features features.csv
#   python learn_mi.py --features features.csv --all-features
#   python learn_mi.py --features features.csv --w1 0.6 --w2 0.3 --w3 0.1
#
# Outputs:
#   reports/mi_formulas.txt                 <- readable formulas/coeffs
#   reports/mi_scores.csv                   <- per-image MI variants
#   reports/mi_violins.png                  <- quick violin plots (3 rows)
#   reports/mi_violin_logit_pretty.png      <- polished single-panel violin (recommended)
#   reports/mi_scatter_fatpct.png           <- scatter MI vs fat_within_steak_pct (if present)

import argparse, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Fixed label order (monotonic marbling)
LABELS_ORDER = ["Select", "Choice", "Prime", "Wagyu"]

DEFAULT_MI_FEATURES = ["area_pct", "orientation_dispersion", "lacunarity_mean"]

# ----- Pretty single-panel violin theme -----
CLASS_ORDER  = ["Select", "Choice", "Prime", "Wagyu"]
CLASS_COLORS = {
    "Select": (0.69, 0.87, 0.90),   # light cyan
    "Choice": (0.80, 0.86, 0.62),   # soft green
    "Prime":  (0.99, 0.80, 0.58),   # apricot
    "Wagyu":  (1.00, 0.69, 0.69),   # soft red
}

def plot_mi_violins_pretty(
    df, score_col, outfile,
    order=("Select","Choice","Prime","Wagyu"),
    colors=None,
    title=None,
    y_label="Marbling Index 0–1",
    show_thresholds=None,   # keep for completeness; pass None to hide
    annotate_counts="xtick" # "xtick" | "bottom" | None
):
    """
    df: DataFrame with columns ['label', score_col]
    score_col: which MI to plot (e.g., 'MI_logit')
    outfile: path to save PNG
    order: class order on x-axis
    colors: dict class->RGB triples (0..1). If None, a pleasant default will be used.
    show_thresholds: list of y values (0..1) for optional dashed lines (None = off)
    annotate_counts: where to show sample sizes: "xtick" appends (n=) to labels,
                     "bottom" draws text near baseline, None to disable.
    """
    # Default palette
    if colors is None:
        colors = {
            "Select": (0.72, 0.86, 0.91),
            "Choice": (0.82, 0.87, 0.64),
            "Prime":  (0.99, 0.80, 0.58),
            "Wagyu":  (1.00, 0.69, 0.69),
        }

    # Filter to present classes
    labels = [cls for cls in order if (df["label"] == cls).any()]
    if not labels:
        return

    # Data per class
    data, ns, means, medians = [], [], [], []
    for cls in labels:
        vals = pd.to_numeric(df.loc[df["label"] == cls, score_col], errors="coerce").dropna().values
        data.append(vals)
        ns.append(len(vals))
        means.append(float(np.mean(vals)) if len(vals) else np.nan)
        medians.append(float(np.median(vals)) if len(vals) else np.nan)

    # Figure
    fig, ax = plt.subplots(figsize=(14, 6))
    positions = np.arange(1, len(labels) + 1)

    # Draw each violin (individual colouring)
    for i, (vals, cls) in enumerate(zip(data, labels), start=1):
        v = ax.violinplot([vals], positions=[i], widths=0.8,
                          showmeans=False, showmedians=False, showextrema=False)
        for body in v["bodies"]:
            body.set_facecolor(colors.get(cls, (0.85, 0.85, 0.85)))
            body.set_edgecolor((0.25, 0.25, 0.25))
            body.set_linewidth(0.8)
            body.set_alpha(0.95)

        # Median bar + mean dot
        ax.hlines(medians[i-1], i - 0.33, i + 0.33, lw=3, color=(0.15, 0.15, 0.18), alpha=1.0)
        ax.plot(i, means[i-1], marker="o", ms=7, color=(0.12, 0.12, 0.12))

    # Optional horizontal reference lines
    if show_thresholds:
        for y in show_thresholds:
            ax.axhline(y, ls="--", lw=1.0, color=(0.45, 0.45, 0.45), alpha=0.4, zorder=0)

    # Axes cosmetics
    if annotate_counts == "xtick":
        xticklabels = [f"{lbl}\n(n={n})" for lbl, n in zip(labels, ns)]
    else:
        xticklabels = labels

    ax.set_xticks(list(positions))
    ax.set_xticklabels(xticklabels, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # Informative title by default
    if title is None:
        N = int(np.nansum(ns))
        title = f"Violin Plot for Marbling Index vs Steak Grade (N={N})"
    ax.set_title(title, fontsize=20, pad=14)

    ax.set_xlim(0.5, len(labels) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, ls=":", alpha=0.25)
    ax.set_axisbelow(True)

    # If you prefer counts near the baseline instead of in xticks
    if annotate_counts == "bottom":
        y0, y1 = ax.get_ylim()
        for i, n in zip(positions, ns):
            ax.text(i, y0 + 0.02*(y1 - y0), f"n={n}",
                    ha="center", va="bottom", fontsize=13, weight="bold", alpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    plt.savefig(outfile, dpi=180)
    plt.close(fig)


# ----- Args / IO -----
def parse_args():
    ap = argparse.ArgumentParser(description="Learn Marbling Index (MI) variants")
    ap.add_argument("--features", type=str, default="features.csv")
    ap.add_argument("--all-features", action="store_true",
                    help="Use all engineered features (except admin columns) for LDA/logistic. "
                         "Hand-crafted MI still uses the 3 canonical features unless you change --mi-features.")
    ap.add_argument("--mi-features", type=str, default=",".join(DEFAULT_MI_FEATURES),
                    help="Comma-separated list of features for the hand-crafted MI. Default: area_pct,orientation_dispersion,lacunarity_mean")
    ap.add_argument("--w1", type=float, default=0.6, help="Weight for fat feature in hand-crafted MI")
    ap.add_argument("--w2", type=float, default=0.3, help="Weight for (1 - orientation_dispersion) in hand-crafted MI")
    ap.add_argument("--w3", type=float, default=0.1, help="Weight for lacunarity_mean in hand-crafted MI")
    return ap.parse_args()

def load_data(path):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("features.csv must contain a 'label' column")
    df = df[df["label"].isin(LABELS_ORDER)].copy()
    admin_cols = [c for c in ["path","image","relpath","label"] if c in df.columns]
    return df, admin_cols

# ----- MI builders -----
def make_handcrafted_mi(df, mi_feats, w1, w2, w3):
    """
    Hand-crafted MI:
      MI = w1 * norm(fat) + w2 * (1 - norm(orientation_dispersion)) + w3 * norm(lacunarity_mean)
    Normalisation: min-max on the data present here (clip to [0,1]).
    """
    mi_feats = [f.strip() for f in mi_feats]
    needed = set(["orientation_dispersion"]) | set(mi_feats)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for MI: {missing}")

    # Prefer fat_within_steak_pct when available
    if "fat_within_steak_pct" in df.columns and "fat_within_steak_pct" in mi_feats:
        fat_col = "fat_within_steak_pct"
    elif "fat_within_steak_pct" in df.columns and "area_pct" in mi_feats:
        fat_col = "fat_within_steak_pct"
    elif "area_pct" in df.columns:
        fat_col = "area_pct"
    else:
        raise ValueError("Neither 'fat_within_steak_pct' nor 'area_pct' found for MI.")

    fat = df[fat_col].astype(float).to_numpy().reshape(-1,1)
    fat_norm = MinMaxScaler().fit_transform(fat).ravel()

    od = df["orientation_dispersion"].astype(float).to_numpy().reshape(-1,1)
    od_norm = MinMaxScaler().fit_transform(od).ravel()
    align_norm = 1.0 - od_norm

    if "lacunarity_mean" in mi_feats and "lacunarity_mean" in df.columns:
        lac = df["lacunarity_mean"].astype(float).to_numpy().reshape(-1,1)
        lac_norm = MinMaxScaler().fit_transform(lac).ravel()
    else:
        lac_norm = np.zeros(len(df), dtype=float)

    W = np.array([w1, w2, w3], dtype=float)
    if W.sum() <= 0:
        raise ValueError("w1+w2+w3 must be > 0")
    W = W / W.sum()

    mi = W[0]*fat_norm + W[1]*align_norm + W[2]*lac_norm
    return mi, {
        "fat_feature": fat_col,
        "weights": {"w1": float(W[0]), "w2": float(W[1]), "w3": float(W[2])},
        "normalisers": {
            fat_col: "min-max (fit on this dataset)",
            "orientation_dispersion": "min-max (fit on this dataset)",
            "lacunarity_mean": "min-max (fit on this dataset)" if "lacunarity_mean" in mi_feats else "not used"
        }
    }

def prepare_matrix(df, use_all_features, admin_cols):
    non_feature_cols = set(admin_cols) | {"label", "label_source"}
    if use_all_features:
        feat_cols = [c for c in df.columns if c not in non_feature_cols]
    else:
        keep_core = [
            "fat_within_steak_pct", "area_pct", "steak_coverage_pct",
            "n_flecks", "mean_fleck_area", "median_fleck_area", "fleck_solidity",
            "glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity",
            "orientation_dispersion", "orientation_anisotropy",
            "lacunarity_mean", "lacunarity_slope"
        ]
        feat_cols = [c for c in keep_core if c in df.columns]

    if len(feat_cols) == 0:
        raise ValueError("No feature columns found to train LDA/Logistic.")

    X = df[feat_cols].copy()
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    Xz = pipe.fit_transform(X)
    return X, Xz, feat_cols, pipe

def labels_to_index(y):
    mapping = {lbl:i for i,lbl in enumerate(LABELS_ORDER)}
    return np.array([mapping[v] for v in y], dtype=int), mapping

def learn_lda_mi(Xz, y):
    lda = LinearDiscriminantAnalysis(n_components=1, solver="svd")
    z = lda.fit_transform(Xz, y).ravel()  # pass y!
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    mi = (z - z_min) / (z_max - z_min + 1e-12)  # normalise 0..1
    w_vec = np.asarray(lda.scalings_[:, 0]).ravel()  # direction in feature space
    return mi, w_vec, 0.0, lda

def learn_logit_mi(Xz, y_idx):
    lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=5000, C=1.0)
    lr.fit(Xz, y_idx)
    proba = lr.predict_proba(Xz)  # N x K
    ranks = np.arange(proba.shape[1], dtype=float)  # [0..K-1]
    exp_rank = (proba * ranks[None, :]).sum(axis=1)   # expected class rank
    exp_rank_norm = exp_rank / (proba.shape[1] - 1.0) # 0..1
    return exp_rank_norm, lr

# ----- Quick 3-row violins (kept for convenience) -----
def plot_violins(df, cols, outfile):
    labels = [lbl for lbl in LABELS_ORDER if (df["label"]==lbl).any()]
    fig, ax = plt.subplots(len(cols), 1, figsize=(6, 3.5*len(cols)))
    if len(cols) == 1:
        ax = [ax]
    for i, c in enumerate(cols):
        data = [df.loc[df["label"]==lbl, c].dropna() for lbl in labels]
        parts = ax[i].violinplot(data, showmedians=True)
        ax[i].set_title(f"{c} by label")
        ax[i].set_xticks(range(1, len(labels)+1))
        ax[i].set_xticklabels(labels, rotation=0)
        ax[i].set_ylabel(c)
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_scatter(df, mi_col, outfile):
    if "fat_within_steak_pct" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(df["fat_within_steak_pct"], df[mi_col], s=12, alpha=0.7)
    ax.set_xlabel("fat_within_steak_pct")
    ax.set_ylabel(mi_col)
    ax.set_title(f"{mi_col} vs fat_within_steak_pct")
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()

# ----- Main -----
def main():
    args = parse_args()
    df, admin_cols = load_data(args.features)

    # Hand-crafted MI
    mi_feats = [s.strip() for s in args.mi_features.split(",") if s.strip()]
    mi_hand, mi_meta = make_handcrafted_mi(df, mi_feats, args.w1, args.w2, args.w3)
    df["MI_handcrafted"] = mi_hand

    # Data-driven MIs
    X, Xz, feat_cols, prep_pipe = prepare_matrix(df, args.all_features, admin_cols)
    y = df["label"].astype(str).to_numpy()
    y_idx, lbl_map = labels_to_index(y)

    mi_lda, lda_coef, lda_intercept, lda = learn_lda_mi(Xz, y)
    df["MI_lda"] = mi_lda

    mi_logit, lr = learn_logit_mi(Xz, y_idx)
    df["MI_logit"] = mi_logit

    # Save scores
    out_cols = []
    for c in ["path","image","relpath"]:
        if c in df.columns: out_cols.append(c)
    out_cols += ["label", "MI_handcrafted", "MI_lda", "MI_logit"]
    os.makedirs("reports", exist_ok=True)
    df[out_cols].to_csv("reports/mi_scores.csv", index=False)

    # Plots (quick 3-row + pretty single-panel)
    plot_violins(df, ["MI_handcrafted","MI_lda","MI_logit"], "reports/mi_violins.png")
    plot_mi_violins_pretty(
    df, "MI_logit", "reports/mi_violin_logit_pretty.png",
    title=None,                 # let the function build an informative title with N
    y_label="Marbling Index 0–1",
    show_thresholds=None,       # remove the dotted guide lines
    annotate_counts="xtick"     # show (n=…) under each class label
    )   

    plot_scatter(df, "MI_handcrafted", "reports/mi_scatter_fatpct.png")

    # Formulas & coefficients (human-readable)
    meta = {
        "handcrafted": {
            "formula": "MI = w1*norm(fat) + w2*(1 - norm(orientation_dispersion)) + w3*norm(lacunarity_mean)",
            "weights": mi_meta["weights"],
            "fat_feature": mi_meta["fat_feature"],
            "normalisers": mi_meta["normalisers"],
        },
        "features_used_for_models": feat_cols,
        "all_features_flag": bool(args.all_features),
    }

    lda_readable = {feat_cols[i]: float(lda_coef[i]) for i in range(len(feat_cols))}
    meta["lda"] = {
        "note": "Coefficients are in standardized feature space (after median-impute + StandardScaler). MI_lda is min-max normalised LDA score.",
        "coef_standardized": lda_readable
    }

    lr_coefs = {LABELS_ORDER[i]: {feat_cols[j]: float(lr.coef_[i, j]) for j in range(len(feat_cols))}
                for i in range(lr.coef_.shape[0])}
    meta["logistic"] = {
        "note": "Multinomial LR on standardized features. MI_logit = expected class rank under predicted probabilities, normalized to [0,1].",
        "coef_per_class_standardized": lr_coefs,
        "intercept_per_class": {LABELS_ORDER[i]: float(lr.intercept_[i]) for i in range(lr.intercept_.shape[0])}
    }

    with open("reports/mi_formulas.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2))

    print("\nSaved:")
    print("  reports/mi_scores.csv")
    print("  reports/mi_violins.png")
    print("  reports/mi_violin_logit_pretty.png")
    print("  reports/mi_scatter_fatpct.png")
    print("  reports/mi_formulas.txt")
    print("\nNotes:")
    print("- Hand-crafted MI uses:", mi_feats, "with weights (normalised):", mi_meta["weights"])
    print("- LDA and Logistic were fit on features:", feat_cols)
    print("- Coefficients are reported in standardized space (median-impute + StandardScaler).")

if __name__ == "__main__":
    main()
