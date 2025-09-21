# score_grader.py
# Usage:
#   python score_grader.py --features features.csv --model grader_model.pkl --out predictions.csv

import argparse
import numpy as np
import pandas as pd
import joblib
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="features.csv")
    ap.add_argument("--model", type=str, default="grader_model.pkl")
    ap.add_argument("--out", type=str, default="predictions.csv")
    args = ap.parse_args()

    # Load model bundle (expects keys: model, features, labels)
    bundle = joblib.load(args.model)
    model     = bundle["model"]
    feat_cols = bundle["features"]
    labels    = bundle["labels"]  # e.g. ["Select","Choice","Prime","Wagyu"]

    # Load features
    df = pd.read_csv(args.features)

    # Ensure derived features the model expects (e.g., meat_pct) exist
    if "meat_pct" in feat_cols and "meat_pct" not in df.columns:
        if "area_pct" in df.columns:
            df["meat_pct"] = 100.0 - df["area_pct"]
        else:
            df["meat_pct"] = np.nan  # imputer in the pipeline will handle NaN

    # Reindex to exactly the model feature set (adds missing cols as NaN, ignores extras)
    X = df.reindex(columns=feat_cols).values

    # Predict labels and probabilities
    try:
        preds = model.predict(X)
    except Exception as e:
        print(f"ERROR: model.predict failed: {e}", file=sys.stderr)
        raise

    # Some pipelines might not expose predict_proba if calibration was skipped and base model lacks it.
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The loaded model does not support predict_proba; cannot produce probability columns.")

    probs = model.predict_proba(X)

    # Align probabilities to the saved label order (bundle['labels'])
    class_list = list(model.classes_)
    try:
        order = [class_list.index(c) for c in labels]
    except ValueError as e:
        raise RuntimeError(f"Model classes {class_list} do not match expected labels {labels}") from e
    probs_ordered = probs[:, order]

    # Build output
    out = df.copy()
    out["predicted"] = preds
    for i, c in enumerate(labels):
        out[f"prob_{c}"] = probs_ordered[:, i]

    out.to_csv(args.out, index=False)
    print(f"Saved {args.out} with columns: predicted + " +
          ", ".join([f'prob_{c}' for c in labels]))

if __name__ == "__main__":
    main()
