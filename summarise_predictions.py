# summarise_predictions.py
# Usage: python summarise_predictions.py --pred predictions.csv

import argparse, pandas as pd, numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.pred)

    # Add confidence & similarity-to-prime for the dashboard
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    df["confidence"] = df[prob_cols].max(axis=1)
    if "prob_Prime" in df.columns:
        df["prime_similarity"] = df["prob_Prime"]
    else:
        df["prime_similarity"] = np.nan

    # Compute metrics if ground-truth labels are present
    if "label" in df.columns:
        y_true = df["label"]
        y_pred = df["predicted"]
        labels = sorted(y_true.unique(), key=lambda x: ["Select","Choice","Prime"].index(x) if x in ["Select","Choice","Prime"] else 99)

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")
        print(f"Accuracy: {acc:.3f} | Macro-F1: {f1:.3f}\n")
        print(classification_report(y_true, y_pred, labels=labels))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Predictions Confusion")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        plt.tight_layout(); os.makedirs("reports", exist_ok=True)
        plt.savefig("reports/confusion_from_predictions.png"); plt.close(fig)
        print("Saved reports/confusion_from_predictions.png")

    # Save helpful slices
    os.makedirs("reports", exist_ok=True)
    df.sort_values("confidence").head(20).to_csv("reports/lowest_confidence.csv", index=False)
    if "label" in df.columns:
        df[df["label"]!=df["predicted"]].sort_values("confidence").to_csv("reports/misclassified.csv", index=False)

    # Confidence histogram
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(df["confidence"], bins=10)
    ax.set_title("Prediction confidence")
    ax.set_xlabel("max class probability"); ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig("reports/confidence_hist.png"); plt.close(fig)
    print("Wrote: reports/lowest_confidence.csv, misclassified.csv*, confidence_hist.png")

if __name__ == "__main__":
    main()
