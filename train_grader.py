# train_grader.py
# Train a USDA (Select/Choice/Prime) grader from engineered features (CPU-only).
# Usage:
#   python train_grader.py --features features.csv

import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import joblib

NON_FEATURE_COLS = {"image","label","label_source","path","relpath"}
SEED = 42

def load_data(path):
    df = pd.read_csv(path)
    # index may be path; make sure label exists and unknowns removed already
    if "label" not in df.columns:
        raise ValueError("features.csv must contain a 'label' column")
    # drop unknown if present
    df = df[df["label"].isin(["Prime","Choice","Select"])].copy()
    # compute meat_pct
    if "area_pct" in df.columns and "meat_pct" not in df.columns:
        df["meat_pct"] = 100.0 - df["area_pct"]
    # feature matrix
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols].values
    y = df["label"].values
    return df, X, y, feat_cols

def model_candidates():
    # class_weight='balanced' to handle Select being smaller
    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=5000,class_weight="balanced",solver="saga"))
    ])
    rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=600, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED))
    ])
    return {"logreg": logreg, "rf": rf}

def evaluate_cv(models, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = {}
    for name, mdl in models.items():
        f1 = cross_val_score(mdl, X, y, cv=skf, scoring="f1_macro")
        acc = cross_val_score(mdl, X, y, cv=skf, scoring="accuracy")
        scores[name] = {"f1_macro_mean": float(f1.mean()), "f1_macro_std": float(f1.std()),
                        "acc_mean": float(acc.mean()), "acc_std": float(acc.std())}
    return scores

def plot_confusion(cm, labels, title, outfile):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); plt.savefig(outfile); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="features.csv")
    args = ap.parse_args()

    df, X, y, feat_cols = load_data(args.features)
    labels = ["Select","Choice","Prime"]  # fixed order for plots

    # Small, imbalanced dataset → stratified hold-out for final sanity check
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    # Compare candidates via 5-fold CV on train
    models = model_candidates()
    scores = evaluate_cv(models, X_tr, y_tr, n_splits=5)
    print("CV scores (train split):", scores)
    # pick best by macro-F1
    best_name = max(scores, key=lambda k: scores[k]["f1_macro_mean"])
    base_model = models[best_name]
    print(f"Selected model: {best_name}")

    # Fit base model on train, then calibrate on train via 3-fold (sigmoid)
    base_model.fit(X_tr, y_tr)
    
    # --- save feature importance from the fitted base model (before calibration) ---
    try:
        if "rf" in best_name:
            rf = base_model.named_steps["clf"]
            imp = rf.feature_importances_
        else:
            clf = base_model.named_steps["clf"]
            imp = np.mean(np.abs(clf.coef_), axis=0)
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values("importance", ascending=False)
        os.makedirs("reports", exist_ok=True)
        imp_df.to_csv("reports/feature_importance.csv", index=False)
        print("Saved reports/feature_importance.csv")
    except Exception as e:
        print("Feature importance not available:", e)

    # calibrate
    calibrated = CalibratedClassifierCV(base_model, cv=3, method="sigmoid")
    calibrated.fit(X_tr, y_tr)

    # Evaluate on hold-out
    y_pred = calibrated.predict(X_te)
    y_prob = calibrated.predict_proba(X_te)
    print("\nHold-out metrics:")
    print("Macro-F1:", f1_score(y_te, y_pred, average="macro"))
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print("\nClassification report:\n", classification_report(y_te, y_pred, labels=labels))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    os.makedirs("reports", exist_ok=True)
    plot_confusion(cm, labels, "Hold-out Confusion", "reports/confusion_matrix.png")

        
        
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(calibrated, X_te, y_te, n_repeats=10, random_state=SEED, scoring="f1_macro")
    pd.DataFrame({"feature": feat_cols, "importance": pi.importances_mean}) \
    .sort_values("importance", ascending=False) \
    .to_csv("reports/feature_importance_perm.csv", index=False)
    print("Saved reports/feature_importance_perm.csv")

    # Save calibrated model + feature list
    joblib.dump({"model": calibrated, "features": feat_cols, "labels": labels}, "grader_model.pkl")
    print("Saved grader_model.pkl")

if __name__ == "__main__":
    main()
