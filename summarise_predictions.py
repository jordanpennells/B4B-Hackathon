# summarise_predictions.py
# Usage: python summarise_predictions.py --pred predictions.csv

import argparse, pandas as pd, numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os
import warnings

LABELS_ORDER = ["Select", "Choice", "Prime", "Wagyu"]  # fixed display/metric order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.pred)

    # ---- Confidence & similarity columns for the dashboard ----
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if len(prob_cols) == 0:
        warnings.warn("No probability columns (prob_*) found. Confidence will be NaN.")
        df["confidence"] = np.nan
    else:
        df["confidence"] = df[prob_cols].max(axis=1)

    # Keep existing prime similarity; add wagyu similarity if present
    df["prime_similarity"] = df["prob_Prime"] if "prob_Prime" in df.columns else np.nan
    if "prob_Wagyu" in df.columns:
        df["wagyu_similarity"] = df["prob_Wagyu"]
    else:
        df["wagyu_similarity"] = np.nan

    # ---- Metrics & confusion matrix (only if ground truth present) ----
    if "label" in df.columns and "predicted" in df.columns:
        y_true = df["label"]
        y_pred = df["predicted"]

        # Use fixed order but keep only labels that actually appear, preserving order
        present = [lbl for lbl in LABELS_ORDER if ((y_true == lbl).any() or (y_pred == lbl).any())]
        if not present:
            warnings.warn("No known labels present in y_true/y_pred; skipping metrics.")
        else:
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro")
            print(f"Accuracy: {acc:.3f} | Macro-F1: {f1:.3f}\n")
            print(classification_report(y_true, y_pred, labels=present))

            cm = confusion_matrix(y_true, y_pred, labels=present)
            # Slightly larger figure to fit 4-class labels comfortably
            fig_size = (5, 5) if len(present) >= 4 else (4, 4)
            fig, ax = plt.subplots(figsize=fig_size)
            im = ax.imshow(cm, interpolation="nearest")
            ax.set_title("Predictions Confusion")
            ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
            ax.set_xticklabels(present); ax.set_yticklabels(present)
            for i in range(len(present)):
                for j in range(len(present)):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center")
            plt.tight_layout(); os.makedirs("reports", exist_ok=True)
            plt.savefig("reports/confusion_from_predictions.png"); plt.close(fig)
            print("Saved reports/confusion_from_predictions.png")
    else:
        warnings.warn("Ground-truth 'label' or 'predicted' column missing; skipping metrics and confusion matrix.")

    # ---- Helpful slices ----
    os.makedirs("reports", exist_ok=True)
    df.sort_values("confidence").head(20).to_csv("reports/lowest_confidence.csv", index=False)
    if "label" in df.columns and "predicted" in df.columns:
        df[df["label"] != df["predicted"]].sort_values("confidence").to_csv("reports/misclassified.csv", index=False)

    # ---- Confidence histogram ----
    fig, ax = plt.subplots(figsize=(5, 3))
    # Handle all-NaN confidence gracefully
    conf_series = df["confidence"].astype(float)
    if conf_series.notna().any():
        ax.hist(conf_series.dropna(), bins=10)
    else:
        ax.text(0.5, 0.5, "No confidence data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Prediction confidence")
    ax.set_xlabel("max class probability"); ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig("reports/confidence_hist.png"); plt.close(fig)
    print("Wrote: reports/lowest_confidence.csv, misclassified.csv*, confidence_hist.png")

if __name__ == "__main__":
    main()
