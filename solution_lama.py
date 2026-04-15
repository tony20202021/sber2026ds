"""
solution_lama.py — LightAutoML (Sber AI Lab).
Автоматический ML на агрегированных признаках из solution.py.
"""
import json
import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from solution import load_split, generate_features, add_target_encoding

RESULTS_DIR = "results/solution_2026_04_15_lama"
os.makedirs(RESULTS_DIR, exist_ok=True)
OVERALL_PATH = "results/overall.json"
BRANCH = "solution_2026_04_15_lama"

# ── Генерация признаков ───────────────────────────────────────────────────────
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

# Ограничиваем top-50 фич для экономии памяти (только 485MB доступно)
with open(OVERALL_PATH) as f:
    _overall = json.load(f)
_fi = next(e for e in _overall if e.get("top_features"))
TOP_FEATURES = _fi["top_features"][:50]
X_train = X_train[TOP_FEATURES]
X_val   = X_val[TOP_FEATURES]
X_test  = X_test[TOP_FEATURES]

# Объединяем train+val для LightAutoML
train_lama = X_train.copy()
train_lama['target'] = y_train.values

# LAMA обучается на train, оценивает на val
print(f"\nFeatures: {X_train.shape[1]}  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

# ── LightAutoML ───────────────────────────────────────────────────────────────
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

task = Task("binary")

roles = {"target": "target"}

automl = TabularAutoML(
    task=task,
    timeout=300,           # 5 минут
    cpu_limit=4,
    general_params={"use_algos": [["lgb", "linear_l2"]]},
)

print("\nTraining LightAutoML (timeout=300s)...")
t0 = time.perf_counter()
oof_pred = automl.fit_predict(train_lama, roles=roles, verbose=1)
train_time = round(time.perf_counter() - t0, 1)
print(f"  train_time: {train_time}s")

# OOF метрики
oof_proba = oof_pred.data[:, 0]
oof_auc   = round(roc_auc_score(y_train, oof_proba), 4)
print(f"  OOF AUC: {oof_auc}")

# Предсказание на test
print("Predicting on test...")
t0 = time.perf_counter()
test_pred  = automl.predict(X_test)
infer_time = round(time.perf_counter() - t0, 4)

y_pred_proba = test_pred.data[:, 0]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc  = round(roc_auc_score(y_test, y_pred_proba), 4)
acc  = round(accuracy_score(y_test, y_pred), 4)
prec = round(precision_score(y_test, y_pred), 4)
rec  = round(recall_score(y_test, y_pred), 4)

print(f"\n=== Test metrics ===")
print(f"  auc_score:       {auc}")
print(f"  accuracy_score:  {acc}")
print(f"  precision_score: {prec}")
print(f"  recall_score:    {rec}")
print(f"  train_time:      {train_time}s")
print(f"  infer_time:      {infer_time}s")

auc_ok = auc > 0.88
acc_ok = acc > 0.80
print(f"\nROC AUC > 0.88: {'OK' if auc_ok else 'FAIL'}  |  Accuracy > 0.80: {'OK' if acc_ok else 'FAIL'}")

# ── Обновление overall.json ───────────────────────────────────────────────────
with open(OVERALL_PATH) as f:
    overall = json.load(f)
overall = [e for e in overall if e["branch"] != BRANCH]

overall.append({
    "branch": BRANCH,
    "date": "2026-04-15",
    "metrics": {
        "auc_score": auc,
        "accuracy_score": acc,
        "precision_score": prec,
        "recall_score": rec,
        "oof_auc": oof_auc,
    },
    "timing": {
        "train_time_s": train_time,
        "infer_time_s": infer_time,
    },
    "n_features": X_train.shape[1],
    "comment": (
        f"LightAutoML (Sber AI Lab) v{__import__('lightautoml').__version__}. "
        f"Те же 376 признаков из solution.py. "
        f"use_algos=[lgb, linear_l2], timeout=300s. "
        f"OOF AUC={oof_auc}."
    ),
})

with open(OVERALL_PATH, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nUpdated: {OVERALL_PATH}")
