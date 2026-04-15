"""
solution_best_200.py — top-170 из feature_importance.csv + 30 фич из solution_2021_old.

Новые фичи (аналогично solution_best.py):
  - per-hour individual shares: hour_share_0..23  (24 фичи)
  - pos_min, pos_max, pos_median                  (3 фичи)
  - neg_min, neg_max, neg_median                  (3 фичи)
Итого: 170 + 30 = 200 фич.
"""
import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb

from solution import (
    load_split, generate_features, add_target_encoding,
    LGB_PARAMS,
)

RESULTS_DIR = "results/solution_2026_04_15_best_from_all_200"
os.makedirs(RESULTS_DIR, exist_ok=True)
OVERALL_PATH = "results/overall.json"
BRANCH = "solution_2026_04_15_best_from_all_200"
FI_CSV = "results/solution_2026_04_06/feature_importance.csv"

# ── Top-170 из feature_importance.csv ────────────────────────────────────────
fi_df = pd.read_csv(FI_CSV)
TOP_170 = fi_df['feature'].head(170).tolist()


# ── Дополнительные фичи из solution_2021_old ─────────────────────────────────

def extra_features(df):
    """
    Per-hour shares + pos/neg min/max/median.
    Возвращает DataFrame с customer_id как индексом.
    """
    df = df.copy()
    df['tr_datetime'] = pd.to_datetime(df['tr_datetime'], errors='coerce')
    df['hour'] = df['tr_datetime'].dt.hour

    hour_shares = (
        df.groupby(['customer_id', 'hour']).size()
          .unstack(fill_value=0).astype('float32')
    )
    hour_shares = hour_shares.div(hour_shares.sum(axis=1), axis=0)
    hour_shares.columns = [f'hour_share_{int(c)}' for c in hour_shares.columns]

    pos_g = df[df['amount'] > 0].groupby('customer_id')['amount']
    neg_g = df[df['amount'] < 0].groupby('customer_id')['amount']

    pos_stats = pos_g.agg(['min', 'max', 'median'])
    pos_stats.columns = ['pos_min', 'pos_max', 'pos_median']

    neg_stats = neg_g.agg(['min', 'max', 'median'])
    neg_stats.columns = ['neg_min', 'neg_max', 'neg_median']

    return pd.concat([hour_shares, pos_stats, neg_stats], axis=1).fillna(0)


# ── Загрузка и генерация признаков ───────────────────────────────────────────

print("Loading splits...")
train_df = load_split("train")
val_df   = load_split("val")
test_df  = load_split("test")

print("Generating base features (solution.py)...")
X_train, y_train, top_mcc, top_tr = generate_features(train_df)
X_val,   y_val,   _,       _      = generate_features(val_df,  top_mcc=top_mcc, top_tr=top_tr)
X_test,  y_test,  _,       _      = generate_features(test_df, top_mcc=top_mcc, top_tr=top_tr)

print("Adding target encoding...")
X_train, X_val, X_test = add_target_encoding(
    X_train, y_train, X_val, X_test, train_df, val_df, test_df)

print("Computing extra features from solution_2021_old...")
extra_tr = extra_features(train_df)
extra_vl = extra_features(val_df)
extra_te = extra_features(test_df)

def get_customer_order(df):
    return df['customer_id'].drop_duplicates().values

cust_train = get_customer_order(train_df)
cust_val   = get_customer_order(val_df)
cust_test  = get_customer_order(test_df)
del train_df, val_df, test_df

def attach_extra(X, extra, cust_order):
    extra_aligned = extra.reindex(cust_order).fillna(0)
    extra_aligned = extra_aligned.reset_index(drop=True)
    new_cols = [c for c in extra_aligned.columns if c not in X.columns]
    return pd.concat([X.reset_index(drop=True), extra_aligned[new_cols]], axis=1)

X_train = attach_extra(X_train, extra_tr, cust_train)
X_val   = attach_extra(X_val,   extra_vl, cust_val)
X_test  = attach_extra(X_test,  extra_te, cust_test)
del extra_tr, extra_vl, extra_te

# Целевой набор: top-170 + новые фичи
extra_names = (
    [f'hour_share_{h}' for h in range(24)] +
    ['pos_min', 'pos_max', 'pos_median', 'neg_min', 'neg_max', 'neg_median']
)
extra_names = [f for f in extra_names if f in X_train.columns]

# Объединяем: top-170 + extra (без дублей)
selected = TOP_170 + [f for f in extra_names if f not in TOP_170]
selected = [f for f in selected if f in X_train.columns]

print(f"Features: top-170={len([f for f in TOP_170 if f in X_train.columns])}, "
      f"extra={len([f for f in extra_names if f not in TOP_170])}, total={len(selected)}")

X_train = X_train[selected]
X_val   = X_val[selected]
X_test  = X_test[selected]

assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]


# ── 5-fold LightGBM ансамбль ─────────────────────────────────────────────────

def fit_predict_proba(X_train, y_train, X_val, y_val, X_test):
    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)

    probe = lgb.LGBMClassifier(n_estimators=3000, random_state=0, **LGB_PARAMS)
    probe.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(period=200)])
    best_iter = int(probe.best_iteration_ * 1.1)
    print(f"  best iteration (×1.1): {best_iter}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_probas = []
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_full, y_full)):
        m = lgb.LGBMClassifier(n_estimators=best_iter, random_state=fold, **LGB_PARAMS)
        m.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx],
              callbacks=[lgb.log_evaluation(period=-1)])
        oof_auc = roc_auc_score(y_full.iloc[vl_idx],
                                 m.predict_proba(X_full.iloc[vl_idx])[:, 1])
        test_probas.append(m.predict_proba(X_test)[:, 1])
        print(f"  fold {fold+1}/5  OOF AUC: {oof_auc:.4f}")

    return np.mean(test_probas, axis=0)


print("\nTraining LightGBM 5-fold ensemble...")
t0 = time.perf_counter()
y_pred_proba = fit_predict_proba(X_train, y_train, X_val, y_val, X_test)
train_time = round(time.perf_counter() - t0, 1)

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
    "metrics": {"auc_score": auc, "accuracy_score": acc,
                 "precision_score": prec, "recall_score": rec},
    "timing": {"train_time_s": train_time},
    "n_features": len(selected),
    "feature_groups": {
        "top170_importance": len([f for f in TOP_170 if f in X_train.columns]),
        "per_hour_shares": len([f for f in extra_names if "hour_share" in f and f not in TOP_170]),
        "pos_neg_minmax_median": len([f for f in extra_names if "hour_share" not in f and f not in TOP_170]),
    },
    "comment": (
        f"Best-from-all-200: top-170 (LightGBM gain из feature_importance.csv) + "
        f"{len([f for f in extra_names if f not in TOP_170])} новых фич "
        f"из solution_2021_old (per-hour shares 0-23 + pos/neg min/max/median). "
        f"Модель: LightGBM 5-fold ensemble на train+val, threshold=0.5."
    ),
})
with open(OVERALL_PATH, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nUpdated: {OVERALL_PATH}")
