"""
solution_mcc_interactions_200.py — MCC × время + per-MCC percentiles.

Новые фичи (для top-20 MCC по частоте):
  MCC × время суток (4 × 20 = 80):
    mcc{k}_morning_ratio   — доля транзакций этого MCC утром (6-12)
    mcc{k}_afternoon_ratio — днём (12-18)
    mcc{k}_evening_ratio   — вечером (18-23)
    mcc{k}_night_ratio     — ночью (23-6)
  MCC × выходные (1 × 20 = 20):
    mcc{k}_weekend_ratio   — доля транзакций этого MCC в выходные
  Per-MCC percentiles (3 × 20 = 60):
    mcc{k}_q25, mcc{k}_q75, mcc{k}_median

Итого новых: 160. Кандидаты: ~376 base + 30 extra + 160 interaction = ~566.
Финальная модель: top-200 по LightGBM importance + 5-fold ensemble.
"""
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb

from solution import (
    load_split, generate_features, add_target_encoding,
    LGB_PARAMS,
)

RESULTS_DIR = "results/solution_2026_04_15_mcc_interactions_200"
os.makedirs(RESULTS_DIR, exist_ok=True)
OVERALL_PATH = "results/overall.json"
BRANCH = "solution_2026_04_15_mcc_interactions_200"
FI_CSV = "results/solution_2026_04_06/feature_importance.csv"

TOP_MCC_N = 20  # для скольких MCC-кодов строим взаимодействия

fi_df = pd.read_csv(FI_CSV)
TOP_170 = fi_df['feature'].head(170).tolist()

# ── Фичи из solution_2021_old ─────────────────────────────────────────────────
def extra_features(df):
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


# ── MCC × время + percentiles ─────────────────────────────────────────────────
def mcc_interaction_features(df, top_mcc_codes):
    """
    Для каждого MCC из top_mcc_codes вычисляет:
      - долю транзакций в каждое время суток (morning/afternoon/evening/night)
      - долю в выходные
      - q25, q75, median суммы
    Возвращает DataFrame с customer_id как индексом.
    """
    df = df.copy()
    df['tr_datetime'] = pd.to_datetime(df['tr_datetime'], errors='coerce')
    df['hour']       = df['tr_datetime'].dt.hour.astype('int8')
    df['dow']        = df['tr_datetime'].dt.dayofweek.astype('int8')
    df['is_morning']   = ((df['hour'] >= 6)  & (df['hour'] < 12)).astype('int8')
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype('int8')
    df['is_evening']   = ((df['hour'] >= 18) & (df['hour'] < 23)).astype('int8')
    df['is_night']     = ((df['hour'] >= 23) | (df['hour'] <  6)).astype('int8')
    df['is_weekend']   = (df['dow'] >= 5).astype('int8')

    parts = []
    for mcc in top_mcc_codes:
        sub = df[df['mcc_code'] == mcc]
        if len(sub) == 0:
            continue
        g = sub.groupby('customer_id')
        stats = pd.DataFrame({
            f'mcc{mcc}_morning_ratio':   g['is_morning'].mean().astype('float32'),
            f'mcc{mcc}_afternoon_ratio': g['is_afternoon'].mean().astype('float32'),
            f'mcc{mcc}_evening_ratio':   g['is_evening'].mean().astype('float32'),
            f'mcc{mcc}_night_ratio':     g['is_night'].mean().astype('float32'),
            f'mcc{mcc}_weekend_ratio':   g['is_weekend'].mean().astype('float32'),
            f'mcc{mcc}_q25':    g['amount'].quantile(0.25).astype('float32'),
            f'mcc{mcc}_q75':    g['amount'].quantile(0.75).astype('float32'),
            f'mcc{mcc}_median': g['amount'].median().astype('float32'),
        })
        parts.append(stats)

    return pd.concat(parts, axis=1).fillna(0) if parts else pd.DataFrame()


def get_customer_order(df):
    return df['customer_id'].drop_duplicates().values


def attach(X, extra_df, cust_order):
    if extra_df.empty:
        return X
    aligned = extra_df.reindex(cust_order).fillna(0).reset_index(drop=True)
    new_cols = [c for c in aligned.columns if c not in X.columns]
    return pd.concat([X.reset_index(drop=True), aligned[new_cols]], axis=1)


# ── Загрузка и генерация признаков ───────────────────────────────────────────
print("Loading splits...")
train_df = load_split("train")
val_df   = load_split("val")
test_df  = load_split("test")

# top-MCC по частоте в train
top_mcc_codes = train_df['mcc_code'].value_counts().nlargest(TOP_MCC_N).index.tolist()
print(f"Top-{TOP_MCC_N} MCC codes: {top_mcc_codes}")

print("Generating base features...")
X_train, y_train, top_mcc, top_tr = generate_features(train_df)
X_val,   y_val,   _,       _      = generate_features(val_df,  top_mcc=top_mcc, top_tr=top_tr)
X_test,  y_test,  _,       _      = generate_features(test_df, top_mcc=top_mcc, top_tr=top_tr)

print("Adding target encoding...")
X_train, X_val, X_test = add_target_encoding(
    X_train, y_train, X_val, X_test, train_df, val_df, test_df)

print("Computing extra features (hour shares + pos/neg stats)...")
extra_tr = extra_features(train_df)
extra_vl = extra_features(val_df)
extra_te = extra_features(test_df)

print(f"Computing MCC × time/percentile interactions (top-{TOP_MCC_N} MCC)...")
int_tr = mcc_interaction_features(train_df, top_mcc_codes)
int_vl = mcc_interaction_features(val_df,   top_mcc_codes)
int_te = mcc_interaction_features(test_df,  top_mcc_codes)

cust_train = get_customer_order(train_df)
cust_val   = get_customer_order(val_df)
cust_test  = get_customer_order(test_df)
del train_df, val_df, test_df

X_train = attach(X_train, extra_tr, cust_train)
X_val   = attach(X_val,   extra_vl, cust_val)
X_test  = attach(X_test,  extra_te, cust_test)

X_train = attach(X_train, int_tr, cust_train)
X_val   = attach(X_val,   int_vl, cust_val)
X_test  = attach(X_test,  int_te, cust_test)
del extra_tr, extra_vl, extra_te, int_tr, int_vl, int_te

# Все кандидаты
extra_names = (
    [f'hour_share_{h}' for h in range(24)] +
    ['pos_min', 'pos_max', 'pos_median', 'neg_min', 'neg_max', 'neg_median']
)
interaction_names = [
    f'mcc{mcc}_{stat}'
    for mcc in top_mcc_codes
    for stat in ('morning_ratio', 'afternoon_ratio', 'evening_ratio',
                 'night_ratio', 'weekend_ratio', 'q25', 'q75', 'median')
]

candidates = (
    list(dict.fromkeys(
        TOP_170
        + [f for f in extra_names       if f not in TOP_170]
        + [f for f in interaction_names if f not in TOP_170]
    ))
)
candidates = [f for f in candidates if f in X_train.columns]
print(f"Candidate features: {len(candidates)}")

X_tr_c = X_train[candidates]
X_vl_c = X_val[candidates]
X_te_c = X_test[candidates]


# ── Шаг 1: быстрый LGB → importance → top-200 ────────────────────────────────
print("\nStep 1: single LGB for feature selection...")
probe = lgb.LGBMClassifier(n_estimators=3000, random_state=0, **LGB_PARAMS)
probe.fit(X_tr_c, y_train,
          eval_set=[(X_vl_c, y_val)],
          callbacks=[lgb.early_stopping(150, verbose=False),
                     lgb.log_evaluation(period=200)])
best_iter = int(probe.best_iteration_ * 1.1)

fi = pd.Series(probe.feature_importances_, index=candidates).sort_values(ascending=False)
top200 = fi.head(200).index.tolist()

# Анализ: какие interaction-фичи попали в top-200
int_in_top = [f for f in top200 if any(f == f'mcc{m}_{s}'
              for m in top_mcc_codes
              for s in ('morning_ratio', 'afternoon_ratio', 'evening_ratio',
                        'night_ratio', 'weekend_ratio', 'q25', 'q75', 'median'))]
print(f"  best_iter (×1.1): {best_iter}")
print(f"  Interaction features in top-200: {len(int_in_top)}")
for f in int_in_top[:20]:
    print(f"    {f}  importance={fi[f]:.0f}")

fi_full = pd.DataFrame({'feature': fi.index, 'importance': fi.values})
fi_full.to_csv(f"{RESULTS_DIR}/feature_importance_all.csv", index=False)
print(f"  Saved full importance → {RESULTS_DIR}/feature_importance_all.csv")

X_train_s = X_train[top200]
X_val_s   = X_val[top200]
X_test_s  = X_test[top200]
del X_tr_c, X_vl_c, X_te_c, X_train, X_val, X_test


# ── Шаг 2: 5-fold ensemble на top-200 ────────────────────────────────────────
def fit_predict_proba(X_tr, y_tr, X_vl, y_vl, X_te):
    X_full = pd.concat([X_tr, X_vl]).reset_index(drop=True)
    y_full = pd.concat([y_tr, y_vl]).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_probas = []
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_full, y_full)):
        m = lgb.LGBMClassifier(n_estimators=best_iter, random_state=fold, **LGB_PARAMS)
        m.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx],
              callbacks=[lgb.log_evaluation(period=-1)])
        oof_auc = roc_auc_score(y_full.iloc[vl_idx],
                                 m.predict_proba(X_full.iloc[vl_idx])[:, 1])
        test_probas.append(m.predict_proba(X_te)[:, 1])
        print(f"  fold {fold+1}/5  OOF AUC: {oof_auc:.4f}")
    return np.mean(test_probas, axis=0)


print(f"\nStep 2: 5-fold ensemble on top-200 features...")
t0 = time.perf_counter()
y_pred_proba = fit_predict_proba(X_train_s, y_train, X_val_s, y_val, X_test_s)
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
    "n_features": 200,
    "feature_groups": {
        "from_top170":        len([f for f in top200 if f in set(TOP_170)]),
        "extra_2021old":      len([f for f in top200 if f in set(extra_names)]),
        "mcc_interactions":   len(int_in_top),
    },
    "interaction_features_selected": int_in_top,
    "comment": (
        f"MCC-interactions-200: top-170 base + 30 extra + "
        f"160 MCC×time/percentile interactions "
        f"(top-{TOP_MCC_N} MCC × morning/afternoon/evening/night/weekend_ratio + q25/q75/median). "
        f"Отобраны top-200 из {len(candidates)} кандидатов. "
        f"MCC-interaction фич в top-200: {len(int_in_top)}. "
        f"Модель: LightGBM 5-fold ensemble на train+val, threshold=0.5."
    ),
})
with open(OVERALL_PATH, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nUpdated: {OVERALL_PATH}")
