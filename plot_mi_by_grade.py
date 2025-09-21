# plot_mi_by_grade.py
import argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt

LABELS_ORDER = ["Select","Choice","Prime","Wagyu"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="reports/mi_scores.csv")
    ap.add_argument("--out", default="reports/mi_by_grade.png")
    ap.add_argument("--metric", default="MI_logit", choices=["MI_handcrafted","MI_lda","MI_logit"])
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    if "label" not in df.columns or args.metric not in df.columns:
        raise SystemExit("scores CSV must contain columns: label and your chosen MI metric.")

    # Global 0–1 normalization across the whole dataset (avoid per-class collapse)
    s = df[args.metric].astype(float)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    df["MI_norm"] = s

    # Keep only labels that are present, preserve canonical order
    labels = [lbl for lbl in LABELS_ORDER if (df["label"] == lbl).any()]

    # Prepare data per class
    data = [df.loc[df["label"]==lbl, "MI_norm"].values for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Marbling Index (normalised 0–1)")
    ax.set_title(f"{args.metric} distribution by grade")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
