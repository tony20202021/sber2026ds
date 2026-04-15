import pandas as pd
import numpy as np
import xgboost as xgb
import re
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')

# ── Адаптация загрузки данных под текущий датасет (минимальные изменения) ─────
COLS = ['customer_id', 'tr_datetime', 'amount', 'mcc_code', 'tr_type', 'gender']

train_df = pd.read_csv('data/train.csv', usecols=COLS,
                        dtype={'customer_id':'int32','mcc_code':'int16',
                               'tr_type':'int16','amount':'float32','gender':'int8'})
val_df   = pd.read_csv('data/val.csv',   usecols=COLS,
                        dtype={'customer_id':'int32','mcc_code':'int16',
                               'tr_type':'int16','amount':'float32','gender':'int8'})
test_df  = pd.read_csv('data/test.csv',  usecols=COLS,
                        dtype={'customer_id':'int32','mcc_code':'int16',
                               'tr_type':'int16','amount':'float32','gender':'int8'})

transactions_train = pd.concat([train_df, val_df], ignore_index=True).set_index('customer_id')
transactions_test  = test_df.set_index('customer_id')
del train_df, val_df, test_df

gender_train = transactions_train['gender'].groupby(level=0).first().rename('gender')
gender_test  = transactions_test['gender'].groupby(level=0).first().rename('gender')

# ── Функции из оригинала (без изменений) ─────────────────────────────────────

def cv_score(params, train, y_true):
    dtrain = xgb.DMatrix(train.values.astype('float32'), y_true.values.astype('float32'),
                          feature_names=list(train.columns))
    cv_res = xgb.cv(params, dtrain,
                    early_stopping_rounds=10, maximize=True,
                    num_boost_round=500, nfold=5, stratified=True,
                    verbose_eval=False)
    auc_series = cv_res['test-auc-mean'].dropna()
    if len(auc_series) == 0:
        print('CV returned no valid results, using default 100 trees')
        return 100
    index_argmax = int(auc_series.values.argmax())
    print('Cross-validation, ROC AUC: {:.3f}+-{:.3f}, Trees: {}'.format(
        auc_series.iloc[index_argmax],
        cv_res['test-auc-std'].dropna().iloc[index_argmax],
        index_argmax))
    return max(index_argmax, 10)


def fit_predict(params, num_trees, train, test, target):
    params['learning_rate'] = params['eta']
    dtrain = xgb.DMatrix(train.values.astype('float32'), target.values.astype('float32'),
                          feature_names=list(train.columns))
    clf = xgb.train(params, dtrain, num_boost_round=num_trees,
                    maximize=True, verbose_eval=False)
    dtest = xgb.DMatrix(test.values.astype('float32'),
                         feature_names=list(train.columns))
    y_pred = clf.predict(dtest)
    submission = pd.DataFrame(index=test.index, data=y_pred,
                               columns=['probability'])
    return clf, submission


def draw_feature_importances(clf, top_k=20, save_path=None):
    importances = dict(sorted(clf.get_score().items(),
                               key=lambda x: x[1])[-top_k:])
    y_pos = np.arange(len(importances))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y_pos, list(importances.values()), align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importances.keys(), fontsize=10)
    ax.set_xlabel('Feature importance', fontsize=12)
    ax.set_title(f'Top-{top_k} Features (XGBoost)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


params = {
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'lambda': 0,
    'alpha': 0,
    'min_child_weight': 0,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'tree_method': 'approx',
}

# ── Advanced feature engineering (адаптация из оригинала) ────────────────────
# Оригинал использовал: day % 7, hour из строки datetime, night (не в 6-22)
# Адаптируем к нашему datetime-формату

print('-----------')
print('advanced features')
print('-----------')

for df in [transactions_train, transactions_test]:
    df['tr_datetime'] = pd.to_datetime(df['tr_datetime'], errors='coerce')
    df['day']   = df['tr_datetime'].dt.dayofweek          # 0=Mon..6=Sun (аналог day%7)
    df['hour']  = df['tr_datetime'].dt.hour
    df['night'] = (~df['hour'].between(6, 22)).astype(int)


def features_creation_advanced(x):
    features = []
    features.append(pd.Series(
        x['day'].value_counts(normalize=True).add_prefix('day_')))
    features.append(pd.Series(
        x['hour'].value_counts(normalize=True).add_prefix('hour_')))
    features.append(pd.Series(
        x['night'].value_counts(normalize=True).add_prefix('night_')))
    features.append(pd.Series(
        x[x['amount'] > 0]['amount']
        .agg(['min', 'max', 'mean', 'median', 'std', 'count'])
        .add_prefix('positive_transactions_')))
    features.append(pd.Series(
        x[x['amount'] < 0]['amount']
        .agg(['min', 'max', 'mean', 'median', 'std', 'count'])
        .add_prefix('negative_transactions_')))
    return pd.concat(features)


def features_vectorized(df):
    """Векторизованный аналог features_creation_advanced (memory-efficient)."""
    import gc
    idx = df.index
    g   = df.groupby(idx)
    parts = []

    # day shares: groupby + unstack (compact: n_customers × 7, not n_rows × 7)
    day_shares = (df.groupby([idx, 'day']).size()
                    .unstack(fill_value=0).astype('float32'))
    day_shares = day_shares.div(day_shares.sum(axis=1), axis=0)
    day_shares.columns = [f'day_{c}' for c in day_shares.columns]
    parts.append(day_shares); del day_shares; gc.collect()

    # hour shares
    hour_shares = (df.groupby([idx, 'hour']).size()
                     .unstack(fill_value=0).astype('float32'))
    hour_shares = hour_shares.div(hour_shares.sum(axis=1), axis=0)
    hour_shares.columns = [f'hour_{c}' for c in hour_shares.columns]
    parts.append(hour_shares); del hour_shares; gc.collect()

    # night share
    parts.append(g['night'].mean().rename('night_1').to_frame())

    # positive/negative stats (no median to save memory)
    pos_g = df[df['amount'] > 0].groupby(df[df['amount'] > 0].index)['amount']
    neg_g = df[df['amount'] < 0].groupby(df[df['amount'] < 0].index)['amount']

    pos_agg = pos_g.agg(['min', 'max', 'mean', 'std', 'count'])
    neg_agg = neg_g.agg(['min', 'max', 'mean', 'std', 'count'])
    pos_agg.columns = [f'positive_transactions_{c}' for c in pos_agg.columns]
    neg_agg.columns = [f'negative_transactions_{c}' for c in neg_agg.columns]
    parts.extend([pos_agg, neg_agg])
    del pos_g, neg_g, pos_agg, neg_agg; gc.collect()

    return pd.concat(parts, axis=1).fillna(0)


print("Computing features for train...")
t0 = time.perf_counter()
data_train = features_vectorized(transactions_train)
feat_time = round(time.perf_counter() - t0, 1)
print(f"  done in {feat_time}s, shape={data_train.shape}")

print("Computing features for test...")
data_test = features_vectorized(transactions_test)
print(f"  shape={data_test.shape}")

data_train = data_train.fillna(0)
data_test  = data_test.fillna(0)

# Align columns
cols = data_train.columns.intersection(data_test.columns)
data_train = data_train[cols]
data_test  = data_test[cols]

target_train = gender_train.reindex(data_train.index).fillna(0).astype(int)
target_test  = gender_test.reindex(data_test.index).fillna(0).astype(int)

# ── Обучение (как в оригинале) ────────────────────────────────────────────────
print("\nCross-validation...")
t0 = time.perf_counter()
num_trees = cv_score(params, data_train, target_train)
cv_time = round(time.perf_counter() - t0, 1)

print(f"\nFitting final model (num_trees={num_trees})...")
t0 = time.perf_counter()
clf, submission = fit_predict(params, num_trees, data_train, data_test, target_train)
train_time = round(time.perf_counter() - t0, 2)

t0 = time.perf_counter()
y_pred_proba = submission['probability'].reindex(data_test.index).values
infer_time = round(time.perf_counter() - t0, 4)

y_pred = (y_pred_proba >= 0.5).astype(int)
y_true = target_test.values

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
auc  = round(roc_auc_score(y_true, y_pred_proba), 4)
acc  = round(accuracy_score(y_true, y_pred), 4)
prec = round(precision_score(y_true, y_pred), 4)
rec  = round(recall_score(y_true, y_pred), 4)

print(f"\n=== Test metrics ===")
print(f"  auc_score:       {auc}")
print(f"  accuracy_score:  {acc}")
print(f"  precision_score: {prec}")
print(f"  recall_score:    {rec}")
print(f"  train_time:      {train_time}s  (cv: {cv_time}s, feat: {feat_time}s)")
print(f"  infer_time:      {infer_time}s")

# ── Feature importance ────────────────────────────────────────────────────────
RESULTS_DIR = "results/solution_2021_old"
os.makedirs(RESULTS_DIR, exist_ok=True)
draw_feature_importances(clf, top_k=20,
                          save_path=f"{RESULTS_DIR}/feature_importance_top20.png")

importances_list = sorted(clf.get_score().items(), key=lambda x: x[1], reverse=True)
fi_df = pd.DataFrame(importances_list, columns=['feature', 'importance'])
fi_df.to_csv(f"{RESULTS_DIR}/feature_importance.csv", index=False)

# ── Анализ фич, отсутствующих в solution_2026_04_06 ──────────────────────────
OVERALL_PATH = "results/overall.json"
with open(OVERALL_PATH) as f:
    overall = json.load(f)

fi_entry = next((e for e in overall if e.get("top_features")), None)
top_features_best = fi_entry["top_features"] if fi_entry else []
best_feat_set = set(top_features_best)
our_feat_set  = set(data_train.columns)

# Категории фич из main.py
our_feat_groups = {
    "per_hour_shares":  [f for f in our_feat_set if f.startswith("hour_")],
    "per_day_shares":   [f for f in our_feat_set if f.startswith("day_")],
    "night_shares":     [f for f in our_feat_set if f.startswith("night_")],
    "pos_tx_stats":     [f for f in our_feat_set if f.startswith("positive_")],
    "neg_tx_stats":     [f for f in our_feat_set if f.startswith("negative_")],
}

# Фичи из main.py, которых нет в solution_2026_04_06
missing_in_best = our_feat_set - best_feat_set

# Группировка по типу
missing_groups = {}
for g, feats in our_feat_groups.items():
    missing = [f for f in feats if f in missing_in_best]
    if missing:
        missing_groups[g] = missing

print(f"\n=== Анализ: фичи из main.py, отсутствующие в solution_2026_04_06 ===")
print(f"  Всего фич в main.py: {len(our_feat_set)}")
print(f"  Отсутствуют в лучшем решении: {len(missing_in_best)}")
for g, feats in missing_groups.items():
    print(f"  {g}: {len(feats)} шт. — напр.: {sorted(feats)[:5]}")

# Топ-10 important фич из main.py, отсутствующих в лучшем решении
top_missing = [(f, imp) for f, imp in importances_list if f in missing_in_best]
print(f"\n  Топ-10 важных фич из main.py, которых нет в solution_2026_04_06:")
for f, imp in top_missing[:10]:
    print(f"    {f}: {imp}")

# ── Обновление overall.json ───────────────────────────────────────────────────
overall = [e for e in overall if e["branch"] != "solution_2021_old"]
overall.append({
    "branch": "solution_2021_old",
    "date": "2026-04-15",
    "metrics": {
        "auc_score": auc,
        "accuracy_score": acc,
        "precision_score": prec,
        "recall_score": rec,
    },
    "timing": {
        "feature_time_s": feat_time,
        "cv_time_s": cv_time,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
    },
    "n_features": len(cols),
    "feature_importance_top20": [
        {"feature": f, "importance": int(imp)} for f, imp in importances_list[:20]
    ],
    "missing_vs_solution_2026_04_06": {
        "n_missing": len(missing_in_best),
        "by_group": {g: len(v) for g, v in missing_groups.items()},
        "top10_important_missing": [f for f, _ in top_missing[:10]],
        "analysis": (
            "Ключевые отличия main.py от solution_2026_04_06: "
            "(1) per-hour shares (24 фичи hour_0..23 vs наши hour_mean/hour_std); "
            "(2) per-day shares (day_0..6, аналог наших dow-ratios но через value_counts); "
            "(3) night share (аналог нашего night_ratio); "
            "(4) pos/neg median, min, max (у нас только mean/std/sum/count); "
            "Что стоит добавить: pos_median, neg_median, pos_min, neg_min, pos_max, neg_max."
        ),
    },
    "comment": (
        "Адаптация main.py (оригинал 2021) под текущий датасет. "
        "Advanced features: per-day/hour/night value_counts + pos/neg transaction stats. "
        "Модель: XGBoost (eta=0.1, max_depth=3) с 5-fold CV. "
        "Обучение на train+val, предсказание на test."
    ),
})

with open(OVERALL_PATH, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nUpdated: {OVERALL_PATH}")
