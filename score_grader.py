# score_grader.py
# Usage:
#   python score_grader.py --features features.csv --model grader_model.pkl

import argparse
import numpy as np
import pandas as pd
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="features.csv")
    ap.add_argument("--model", type=str, default="grader_model.pkl")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model   = bundle["model"]
    feat_cols = bundle["features"]
    labels  = bundle["labels"]  # expected USDA order

    df = pd.read_csv(args.features)

    # Auto-add derived features the model expects
    if "meat_pct" in feat_cols and "meat_pct" not in df.columns:
        if "area_pct" in df.columns:
            df["meat_pct"] = 100.0 - df["area_pct"]
        else:
            # create placeholder; imputer in the pipeline will handle NaN
            df["meat_pct"] = np.nan

    # Reindex to exactly the model feature set (adds missing cols as NaN, ignores extras)
    X = df.reindex(columns=feat_cols).values

    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)

    # Align probabilities to fixed USDA order
    class_list = list(model.classes_)
    try:
        order = [class_list.index(c) for c in labels]
    except ValueError as e:
        raise RuntimeError(f"Model classes {class_list} do not match expected labels {labels}") from e
    probs_ordered = probs[:, order]

    out = df.copy()
    out["predicted"] = preds
    for i, c in enumerate(labels):
        out[f"prob_{c}"] = probs_ordered[:, i]

    out.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")

if __name__ == "__main__":
    main()
