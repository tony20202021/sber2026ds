"""
solution_holidays_200.py — top-170 + 30 extra (из solution_2021_old) + праздничные фичи.

Праздники (ежегодные):
  new_year  01.01  xmas_orth 07.01  xmas_west 25.12
  feb23     23.02  mar8      08.03
  may1      01.05  may9      09.05
  sep1      01.09  nov7      07.11  halloween 31.10

Для каждого праздника × каждой транзакции:
  signed_days = дней до ближайшего вхождения праздника
                (отрицательные = до праздника, положительные = после)

Агрегация на клиента: mean, std, near7_ratio (доля транзакций в ±7 дней)
  10 праздников × 3 стат = 30 holiday-фич

Итого входных признаков: ~230 (170 base + 30 extra + 30 holiday)
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

RESULTS_DIR = "results/solution_2026_04_15_holidays_200"
os.makedirs(RESULTS_DIR, exist_ok=True)
OVERALL_PATH = "results/overall.json"
BRANCH = "solution_2026_04_15_holidays_200"
FI_CSV = "results/solution_2026_04_06/feature_importance.csv"

# ── Top-170 базовых признаков ─────────────────────────────────────────────────
fi_df = pd.read_csv(FI_CSV)
TOP_170 = fi_df['feature'].head(170).tolist()

# ── Праздники ─────────────────────────────────────────────────────────────────
HOLIDAYS = {
    'new_year':   (1,  1),
    'xmas_orth':  (1,  7),
    'feb23':      (2, 23),
    'mar8':       (3,  8),
    'may1':       (5,  1),
    'may9':       (5,  9),
    'halloween':  (10, 31),
    'nov7':       (11,  7),
    'xmas_west':  (12, 25),
    'new_year_e': (12, 31),   # канун Нового года — активная покупка подарков
}


def _signed_days_to_holiday(dates_ts, month, day):
    """Вычисляет знаковое расстояние в днях от каждой транзакции до ближайшего
    вхождения праздника (month/day) в любом году.
    Отрицательные = транзакция до праздника, положительные = после.
    """
    year = dates_ts.dt.year.values
    ts   = dates_ts.values.astype('datetime64[D]').astype(np.int64)

    results = np.empty(len(ts), dtype=np.int64)
    for offset in (-1, 0, 1):
        try:
            h = pd.to_datetime(
                {'year': year + offset, 'month': month, 'day': day}
            ).values.astype('datetime64[D]').astype(np.int64)
        except Exception:
            continue
        d = ts - h          # положительные = после праздника
        if offset == -1:
            results = d
        else:
            # Оставляем значение с меньшим |d|
            mask = np.abs(d) < np.abs(results)
            results = np.where(mask, d, results)
    return results.astype(np.int32)


def holiday_features(df):
    """
    Возвращает DataFrame [customer_id × holiday_features].
    """
    df = df.copy()
    df['tr_datetime'] = pd.to_datetime(df['tr_datetime'], errors='coerce')

    parts = []
    for name, (month, day) in HOLIDAYS.items():
        col = f'_hd_{name}'
        df[col] = _signed_days_to_holiday(df['tr_datetime'], month, day)

        grp = df.groupby('customer_id')[col]
        stats = pd.DataFrame({
            f'hd_{name}_mean':  grp.mean().astype('float32'),
            f'hd_{name}_std':   grp.std().fillna(0).astype('float32'),
            f'hd_{name}_near7': grp.apply(
                lambda x: (x.abs() <= 7).mean()
            ).astype('float32'),
        })
        parts.append(stats)

    return pd.concat(parts, axis=1)


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


def get_customer_order(df):
    return df['customer_id'].drop_duplicates().values


def attach(X, extra_df, cust_order):
    aligned = extra_df.reindex(cust_order).fillna(0).reset_index(drop=True)
    new_cols = [c for c in aligned.columns if c not in X.columns]
    return pd.concat([X.reset_index(drop=True), aligned[new_cols]], axis=1)


# ── Загрузка и генерация признаков ───────────────────────────────────────────
print("Loading splits...")
train_df = load_split("train")
val_df   = load_split("val")
test_df  = load_split("test")

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

print("Computing holiday features...")
hol_tr = holiday_features(train_df)
hol_vl = holiday_features(val_df)
hol_te = holiday_features(test_df)

cust_train = get_customer_order(train_df)
cust_val   = get_customer_order(val_df)
cust_test  = get_customer_order(test_df)
del train_df, val_df, test_df

X_train = attach(X_train, extra_tr, cust_train)
X_val   = attach(X_val,   extra_vl, cust_val)
X_test  = attach(X_test,  extra_te, cust_test)

X_train = attach(X_train, hol_tr, cust_train)
X_val   = attach(X_val,   hol_vl, cust_val)
X_test  = attach(X_test,  hol_te, cust_test)
del extra_tr, extra_vl, extra_te, hol_tr, hol_vl, hol_te

# Кандидаты: top-170 + extra + holiday
extra_names = (
    [f'hour_share_{h}' for h in range(24)] +
    ['pos_min', 'pos_max', 'pos_median', 'neg_min', 'neg_max', 'neg_median']
)
holiday_names = [
    f'hd_{name}_{stat}'
    for name in HOLIDAYS
    for stat in ('mean', 'std', 'near7')
]

candidates = (
    TOP_170
    + [f for f in extra_names   if f not in TOP_170]
    + [f for f in holiday_names if f not in TOP_170]
)
candidates = [f for f in candidates if f in X_train.columns]
print(f"Candidate features: {len(candidates)}")

X_train_c = X_train[candidates]
X_val_c   = X_val[candidates]
X_test_c  = X_test[candidates]


# ── Шаг 1: быстрый одиночный LGB → importance → top-200 ─────────────────────
print("\nStep 1: single LGB for feature selection...")
probe = lgb.LGBMClassifier(n_estimators=3000, random_state=0, **LGB_PARAMS)
probe.fit(X_train_c, y_train,
          eval_set=[(X_val_c, y_val)],
          callbacks=[lgb.early_stopping(150, verbose=False),
                     lgb.log_evaluation(period=200)])
best_iter = int(probe.best_iteration_ * 1.1)

fi = pd.Series(probe.feature_importances_, index=candidates).sort_values(ascending=False)
top200 = fi.head(200).index.tolist()

# Сколько holiday-фич попало в top-200?
hol_in_top = [f for f in top200 if f.startswith('hd_')]
print(f"  best_iter (×1.1): {best_iter}")
print(f"  Holiday features in top-200: {len(hol_in_top)}")
for f in hol_in_top:
    print(f"    {f}  importance={fi[f]:.0f}")

# Сохраняем полный importance для анализа
fi_full = pd.DataFrame({'feature': fi.index, 'importance': fi.values})
fi_full.to_csv(f"{RESULTS_DIR}/feature_importance_all.csv", index=False)
print(f"  Saved full importance → {RESULTS_DIR}/feature_importance_all.csv")

X_train_s = X_train[top200]
X_val_s   = X_val[top200]
X_test_s  = X_test[top200]
del X_train_c, X_val_c, X_test_c, X_train, X_val, X_test


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
        "from_top170":    len([f for f in top200 if f in set(TOP_170)]),
        "extra_2021old":  len([f for f in top200 if f in set(extra_names)]),
        "holiday":        len(hol_in_top),
    },
    "holiday_features_selected": hol_in_top,
    "comment": (
        f"Holidays-200: top-170 base + 30 extra (hour_shares + pos/neg stats) + "
        f"30 holiday proximity features (10 holidays × mean/std/near7). "
        f"Отобраны top-200 по LightGBM importance из ~{len(candidates)} кандидатов. "
        f"Holiday фич в top-200: {len(hol_in_top)}. "
        f"Модель: LightGBM 5-fold ensemble на train+val, threshold=0.5."
    ),
})
with open(OVERALL_PATH, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nUpdated: {OVERALL_PATH}")
