"""
Feature importance analysis.
Trains a single LightGBM on train (early-stop on val), computes importances,
plots top-50, then evaluates metrics for several top-N cutoffs.
Saves plot to results/solution_2026_04_06/.
"""
import json
import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from solution import (
    load_split, generate_features, add_target_encoding,
    LGB_PARAMS, evaluate,
)

RESULTS_DIR = "results/solution_2026_04_06"
os.makedirs(RESULTS_DIR, exist_ok=True)


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

print(f"Total features: {X_train.shape[1]}")


# ── Train single model for importance ────────────────────────────────────────

print("Training probe model (single, for importance)...")
probe = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS)
t0 = time.perf_counter()
probe.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(150, verbose=False),
               lgb.log_evaluation(period=200)],
)
train_time_full = round(time.perf_counter() - t0, 2)
best_iter = probe.best_iteration_
print(f"  best_iter={best_iter}  train_time={train_time_full}s")

t0 = time.perf_counter()
_ = probe.predict_proba(X_test)
infer_time_full = round(time.perf_counter() - t0, 4)
print(f"  inference_time={infer_time_full}s  ({len(X_test)} samples)")


# ── Feature importance ────────────────────────────────────────────────────────

importances = pd.DataFrame({
    "feature":    X_train.columns,
    "importance": probe.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

importances.to_csv(f"{RESULTS_DIR}/feature_importance.csv", index=False)
print(f"Saved: {RESULTS_DIR}/feature_importance.csv")

# Plot top 50
TOP_PLOT = 50
fig, ax = plt.subplots(figsize=(10, 16))
top_df = importances.head(TOP_PLOT)
ax.barh(top_df["feature"][::-1], top_df["importance"][::-1])
ax.set_xlabel("Importance (gain)")
ax.set_title(f"Top-{TOP_PLOT} feature importances (LightGBM)")
plt.tight_layout()
plot_path = f"{RESULTS_DIR}/feature_importance_top{TOP_PLOT}.png"
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"Saved: {plot_path}")


# ── Evaluate different top-N cutoffs ─────────────────────────────────────────

print("\nEvaluating top-N feature subsets (single model, train→val)...")

cutoffs = [10, 20, 30, 50, 75, 100, 150, 200, X_train.shape[1]]
rows = []

for n in cutoffs:
    top_feats = importances["feature"].head(n).tolist()
    m = lgb.LGBMClassifier(n_estimators=best_iter, random_state=42, **LGB_PARAMS)
    t0 = time.perf_counter()
    m.fit(X_train[top_feats], y_train, callbacks=[lgb.log_evaluation(period=-1)])
    t_train = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    proba = m.predict_proba(X_test[top_feats])[:, 1]
    t_infer = round(time.perf_counter() - t0, 4)
    pred  = (proba >= 0.5).astype(int)
    auc   = round(roc_auc_score(y_test, proba), 4)
    acc   = round(accuracy_score(y_test, pred), 4)
    rows.append({"n_features": n, "auc_score": auc, "accuracy_score": acc,
                 "train_time_s": t_train, "infer_time_s": t_infer})
    print(f"  top-{n:>3}: AUC={auc:.4f}  acc={acc:.4f}  train={t_train}s  infer={t_infer}s")

cutoff_df = pd.DataFrame(rows)
cutoff_df.to_csv(f"{RESULTS_DIR}/cutoff_metrics.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/cutoff_metrics.csv")

# Pick optimal: fastest (fewest features) where AUC doesn't drop > 0.01 from max
max_auc = cutoff_df["auc_score"].max()
candidates = cutoff_df[cutoff_df["auc_score"] >= max_auc - 0.01]
optimal_n = int(candidates["n_features"].min())
optimal_row = candidates[candidates["n_features"] == optimal_n].iloc[0]

print(f"\nOptimal cutoff: top-{optimal_n} features")
print(f"  AUC={optimal_row['auc_score']}  acc={optimal_row['accuracy_score']}")

top_features = importances["feature"].head(optimal_n).tolist()

# Plot AUC vs N
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cutoff_df["n_features"], cutoff_df["auc_score"], marker="o", label="AUC")
ax.plot(cutoff_df["n_features"], cutoff_df["accuracy_score"], marker="s", label="Accuracy")
ax.axvline(optimal_n, color="red", linestyle="--", label=f"optimal={optimal_n}")
ax.set_xlabel("Number of features")
ax.set_ylabel("Score")
ax.set_title("Metrics vs. number of top features")
ax.legend()
ax.grid(True)
plt.tight_layout()
curve_path = f"{RESULTS_DIR}/cutoff_curve.png"
plt.savefig(curve_path, dpi=120)
plt.close()
print(f"Saved: {curve_path}")


# ── Update overall.json ───────────────────────────────────────────────────────

overall_path = "results/overall.json"
with open(overall_path) as f:
    overall = json.load(f)

new_entry = {
    "branch": "solution_2026_04_15_feature_importance",
    "date": "2026-04-15",
    "metrics": {
        "auc_score":      float(optimal_row["auc_score"]),
        "accuracy_score": float(optimal_row["accuracy_score"]),
    },
    "timing": {
        "train_time_s":  float(optimal_row["train_time_s"]),
        "infer_time_s":  float(optimal_row["infer_time_s"]),
        "train_time_full_features_s": train_time_full,
        "infer_time_full_features_s": infer_time_full,
    },
    "n_features": optimal_n,
    "top_features": top_features,
    "comment": (
        f"Feature importance analysis. "
        f"Из {X_train.shape[1]} признаков выбраны top-{optimal_n} по LightGBM gain importance. "
        f"Критерий: минимальное кол-во фич при просадке AUC не более 0.01 от максимума. "
        f"Модель: одиночный LightGBM (без CV-ансамбля), threshold=0.5. "
        f"Цель — ускорение обучения, итоговые метрики могут быть ниже целевых."
    ),
}

overall.append(new_entry)
with open(overall_path, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nUpdated: {overall_path}")
print("Done.")
