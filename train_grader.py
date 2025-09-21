# train_grader.py
# Train a 4-class marbling grader (Select / Choice / Prime / Wagyu) from engineered features (CPU-only).
# Usage:
#   python train_grader.py --features features.csv

import argparse, os, warnings
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
import matplotlib.pyplot as plt
import joblib

NON_FEATURE_COLS = {"image","label","label_source","path","relpath"}
SEED = 42
LABELS_ORDER = ["Select","Choice","Prime","Wagyu"]  # fixed order for reports/plots/saved bundle

def load_data(path):
    df = pd.read_csv(path)
    # ensure label exists
    if "label" not in df.columns:
        raise ValueError("features.csv must contain a 'label' column")

    # keep only our target classes
    df = df[df["label"].isin(LABELS_ORDER)].copy()

    # ----- steak-aware preference -----
    # Use fat% inside steak if present; else fall back to frame-based area_pct.
    used_fat_col = None
    if "fat_within_steak_pct" in df.columns:
        used_fat_col = "fat_within_steak_pct"
        # (Re)define meat_pct from steak-fat%
        df["meat_pct"] = 100.0 - df["fat_within_steak_pct"]
        # Optional: if you do NOT want frame-based area_pct to influence the model, drop it here:
        # df = df.drop(columns=[c for c in ["area_pct"] if c in df.columns])
    else:
        # legacy fallback
        if "area_pct" not in df.columns:
            raise ValueError("Expected 'fat_within_steak_pct' or 'area_pct' in features.csv")
        used_fat_col = "area_pct"
        if "meat_pct" not in df.columns:
            df["meat_pct"] = 100.0 - df["area_pct"]

    print(f"Training will use fat metric: {used_fat_col}")

    # feature matrix
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols].values
    y = df["label"].values
    return df, X, y, feat_cols

def model_candidates():
    # class_weight='balanced' helps with class imbalance
    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="saga", multi_class="auto"))
    ])
    rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=600, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=SEED))
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
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); os.makedirs("reports", exist_ok=True)
    plt.savefig(outfile); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="features.csv")
    args = ap.parse_args()

    df, X, y, feat_cols = load_data(args.features)

    # --- dataset checks & CV folds ---
    if len(df) == 0:
        raise ValueError("No rows to train on after filtering to Select/Choice/Prime/Wagyu.")

    class_counts = pd.Series(y).value_counts().reindex(LABELS_ORDER).fillna(0).astype(int)
    print("Class counts:\n", class_counts.to_string())

    min_class = int(class_counts.min())
    cv_folds = max(2, min(5, min_class)) if min_class >= 2 else None

    # --- hold-out split (only if each class has >=2 examples) ---
    can_holdout = min_class >= 2
    if can_holdout:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=SEED
        )
    else:
        warnings.warn(
            "At least one class has < 2 samples. Skipping stratified hold-out and cross-validation; "
            "the model will be fit on all data and only basic training metrics will be shown."
        )
        X_tr, y_tr = X, y
        X_te, y_te = None, None

    # --- model selection ---
    models = model_candidates()
    if cv_folds is not None:
        scores = evaluate_cv(models, X_tr, y_tr, n_splits=cv_folds)
        print(f"CV (n_splits={cv_folds}) scores on train split:", scores)
        best_name = max(scores, key=lambda k: scores[k]["f1_macro_mean"])
    else:
        scores = {k: {"note": "no CV due to class counts < 2"} for k in models}
        best_name = "rf"  # sensible default when data is tiny/imbalanced
    base_model = models[best_name]
    print(f"Selected model: {best_name}")

    # --- fit base model then calibrate (sigmoid) ---
    base_model.fit(X_tr, y_tr)

    # Save feature importance before calibration
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

    # Calibrate (if possible)
    if min_class >= 2:
        calibrated = CalibratedClassifierCV(base_model, cv=3, method="sigmoid")
        calibrated.fit(X_tr, y_tr)
    else:
        warnings.warn("Skipping probability calibration due to class counts; using base model for probabilities.")
        calibrated = base_model

    # --- evaluation ---
    if X_te is not None:
        y_pred = calibrated.predict(X_te)
        print("\nHold-out metrics:")
        print("Macro-F1:", f1_score(y_te, y_pred, average="macro"))
        print("Accuracy:", accuracy_score(y_te, y_pred))
        print("\nClassification report:\n",
              classification_report(y_te, y_pred, labels=LABELS_ORDER))
        cm = confusion_matrix(y_te, y_pred, labels=LABELS_ORDER)
        plot_confusion(cm, LABELS_ORDER, "Hold-out Confusion", "reports/confusion_matrix.png")
    else:
        y_pred_tr = calibrated.predict(X_tr)
        print("\nTrain-set (no hold-out) quick report — NOT a generalisation metric:")
        try:
            print("Macro-F1:", f1_score(y_tr, y_pred_tr, average="macro"))
            print("Accuracy:", accuracy_score(y_tr, y_pred_tr))
            print("\nClassification report:\n",
                  classification_report(y_tr, y_pred_tr, labels=LABELS_ORDER))
        except Exception as e:
            print("Train-set report not available:", e)

    # Permutation importance on hold-out (only if we have one)
    try:
        if X_te is not None:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(calibrated, X_te, y_te, n_repeats=10, random_state=SEED, scoring="f1_macro")
            pd.DataFrame({"feature": feat_cols, "importance": pi.importances_mean}) \
              .sort_values("importance", ascending=False) \
              .to_csv("reports/feature_importance_perm.csv", index=False)
            print("Saved reports/feature_importance_perm.csv")
        else:
            warnings.warn("Skipping permutation importance (no hold-out split).")
    except Exception as e:
        warnings.warn(f"Permutation importance skipped due to error: {e}")

    # --- save calibrated model + feature list + fixed label order ---
    joblib.dump({"model": calibrated, "features": feat_cols, "labels": LABELS_ORDER}, "grader_model.pkl")
    print("Saved grader_model.pkl with labels:", LABELS_ORDER)

if __name__ == "__main__":
    main()
