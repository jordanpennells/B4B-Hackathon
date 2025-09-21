# summarise_predictions.py
# Usage: python summarise_predictions.py --pred predictions.csv

import argparse, pandas as pd, numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import os
import warnings
from io import StringIO

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

    df["prime_similarity"] = df["prob_Prime"] if "prob_Prime" in df.columns else np.nan
    df["wagyu_similarity"] = df["prob_Wagyu"] if "prob_Wagyu" in df.columns else np.nan

    os.makedirs("reports", exist_ok=True)

    # ---- Metrics & confusion matrix (only if ground truth present) ----
    if "label" in df.columns and "predicted" in df.columns:
        y_true = df["label"]
        y_pred = df["predicted"]

        # Keep only labels actually present (but in fixed order)
        present = [lbl for lbl in LABELS_ORDER if ((y_true == lbl).any() or (y_pred == lbl).any())]
        if not present:
            warnings.warn("No known labels present in y_true/y_pred; skipping metrics.")
        else:
            # Class distribution
            counts_true = y_true.value_counts().reindex(LABELS_ORDER).fillna(0).astype(int)
            print("Ground-truth class counts:\n", counts_true.to_string())

            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro")
            print(f"\nAccuracy: {acc:.3f} | Macro-F1: {f1:.3f}\n")

            # Text report (also saved to file)
            report_txt = classification_report(y_true, y_pred, labels=present)
            print(report_txt)
            with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
                f.write("Accuracy: {:.3f}\nMacro-F1: {:.3f}\n\n".format(acc, f1))
                f.write(report_txt)
            print("Saved reports/classification_report.txt")

            # Per-class table (CSV)
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
            per_class = pd.DataFrame({
                "label": present,
                "precision": p, "recall": r, "f1": f, "support": s
            })
            per_class.to_csv("reports/metrics_per_class.csv", index=False)
            print("Saved reports/metrics_per_class.csv")

            # Confusion (PNG + CSV)
            cm = confusion_matrix(y_true, y_pred, labels=present)
            pd.DataFrame(cm, index=present, columns=present).to_csv("reports/confusion_matrix.csv")
            fig_size = (5, 5) if len(present) >= 4 else (4, 4)
            fig, ax = plt.subplots(figsize=fig_size)
            im = ax.imshow(cm, interpolation="nearest")
            ax.set_title("Predictions Confusion")
            ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
            ax.set_xticklabels(present); ax.set_yticklabels(present)
            for i in range(len(present)):
                for j in range(len(present)):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center")
            plt.tight_layout()
            plt.savefig("reports/confusion_from_predictions.png")
            plt.close(fig)
            print("Saved reports/confusion_from_predictions.png and reports/confusion_matrix.csv")
    else:
        warnings.warn("Ground-truth 'label' or 'predicted' column missing; skipping metrics and confusion matrix.")

    # ---- Helpful slices ----
    df.sort_values("confidence").head(20).to_csv("reports/lowest_confidence.csv", index=False)
    if "label" in df.columns and "predicted" in df.columns:
        df[df["label"] != df["predicted"]].sort_values("confidence").to_csv("reports/misclassified.csv", index=False)

    # ---- Confidence histogram ----
    fig, ax = plt.subplots(figsize=(5, 3))
    conf_series = df["confidence"].astype(float)
    if conf_series.notna().any():
        ax.hist(conf_series.dropna(), bins=10)
    else:
        ax.text(0.5, 0.5, "No confidence data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Prediction confidence")
    ax.set_xlabel("max class probability")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig("reports/confidence_hist.png")
    plt.close(fig)
    print("Wrote: reports/lowest_confidence.csv, misclassified.csv*, confidence_hist.png")

if __name__ == "__main__":
    main()
