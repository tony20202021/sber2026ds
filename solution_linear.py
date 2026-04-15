"""
Linear model (LogisticRegression) baseline.
Uses top-100 features from feature importance analysis.
Compares metrics and speed vs LightGBM.
"""
import json
import os
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from solution import load_split, generate_features, add_target_encoding

RESULTS_DIR = "results/solution_2026_04_15_linear"
os.makedirs(RESULTS_DIR, exist_ok=True)
OVERALL_PATH = "results/overall.json"
BRANCH = "solution_2026_04_15_linear"

# Top-100 features from feature_importance.py
with open("results/overall.json") as f:
    overall = json.load(f)
fi_entry = next(e for e in overall if e["branch"] == "solution_2026_04_15_feature_importance")
TOP_FEATURES = fi_entry["top_features"]
print(f"Using {len(TOP_FEATURES)} features from feature importance analysis")

# ── Load & generate features ─────────────────────────────────────────────────
print("Loading splits...")
train_df = load_split("train")
val_df   = load_split("val")
test_df  = load_split("test")

print("Generating features...")
X_train, y_train, top_mcc, top_tr = generate_features(train_df)
X_val,   y_val,   _,       _      = generate_features(val_df,  top_mcc=top_mcc, top_tr=top_tr)
X_test,  y_test,  _,       _      = generate_features(test_df, top_mcc=top_mcc, top_tr=top_tr)

print("Adding target encoding...")
X_train, X_val, X_test = add_target_encoding(
    X_train, y_train, X_val, X_test, train_df, val_df, test_df)
del train_df, val_df, test_df

# Use top features only
X_train_top = X_train[TOP_FEATURES]
X_val_top   = X_val[TOP_FEATURES]
X_test_top  = X_test[TOP_FEATURES]

# ── Train on train+val ────────────────────────────────────────────────────────
X_full = pd.concat([X_train_top, X_val_top]).reset_index(drop=True)
y_full = pd.concat([y_train, y_val]).reset_index(drop=True)

print("Training LogisticRegression (train+val)...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                random_state=42, n_jobs=-1)),
])

t0 = time.perf_counter()
model.fit(X_full, y_full)
train_time = round(time.perf_counter() - t0, 2)

t0 = time.perf_counter()
y_pred_proba = model.predict_proba(X_test_top)[:, 1]
infer_time = round(time.perf_counter() - t0, 4)

y_pred = (y_pred_proba >= 0.5).astype(int)

auc = round(roc_auc_score(y_test, y_pred_proba), 4)
acc = round(accuracy_score(y_test, y_pred), 4)
prec = round(precision_score(y_test, y_pred), 4)
rec  = round(recall_score(y_test, y_pred), 4)

print(f"\n=== Test metrics ===")
print(f"  auc_score:       {auc}")
print(f"  accuracy_score:  {acc}")
print(f"  precision_score: {prec}")
print(f"  recall_score:    {rec}")
print(f"  train_time:      {train_time}s")
print(f"  infer_time:      {infer_time}s")

# ── Update overall.json ───────────────────────────────────────────────────────
with open(OVERALL_PATH) as f:
    overall = json.load(f)

# Remove existing entry for this branch if re-running
overall = [e for e in overall if e["branch"] != BRANCH]

overall.append({
    "branch": BRANCH,
    "date": "2026-04-15",
    "metrics": {
        "auc_score": auc,
        "accuracy_score": acc,
        "precision_score": prec,
        "recall_score": rec,
    },
    "timing": {
        "train_time_s": train_time,
        "infer_time_s": infer_time,
    },
    "n_features": len(TOP_FEATURES),
    "comment": (
        f"Logistic Regression (sklearn, lbfgs, C=1.0) + StandardScaler. "
        f"Top-{len(TOP_FEATURES)} признаков из LightGBM importance. "
        f"Обучение на train+val, threshold=0.5. "
        f"Цель — быстрая линейная модель для сравнения."
    ),
})

with open(OVERALL_PATH, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nUpdated: {OVERALL_PATH}")
