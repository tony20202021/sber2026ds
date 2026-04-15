"""
solution_gender_emb_200.py — MCC interactions + gender embedding features.

Добавляет к лучшему эксперименту (mcc_interactions) новые признаки:
  - mcc_gender_diff_mean/std  — средний/std «мужской скор» MCC по транзакциям клиента
  - mcc_male_sim_mean         — близость к мужскому центроиду
  - mcc_female_sim_mean       — близость к женскому центроиду
  - mcc_gender_diff_amt_wt    — то же, взвешенное по |amount|
  - tr_gender_diff_mean/std   — аналог для типа транзакции
  - tr_male_sim_mean, tr_female_sim_mean
  Итого: 8 новых фич.

Все gender-scores берутся из precomputed файлов (compute_gender_embeddings.py).
Отбираем top-200 из ~370 кандидатов по LightGBM importance, затем 5-fold ensemble.
"""
import os, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb

from solution import load_split, generate_features, add_target_encoding, LGB_PARAMS

RESULTS_DIR  = "results/solution_2026_04_15_gender_emb_200"
os.makedirs(RESULTS_DIR, exist_ok=True)
OVERALL_PATH = "results/overall.json"
BRANCH       = "solution_2026_04_15_gender_emb_200"
FI_CSV       = "results/solution_2026_04_06/feature_importance.csv"
MCC_EMB_CSV  = "results/gender_embeddings/mcc_gender.csv"
TR_EMB_CSV   = "results/gender_embeddings/tr_gender.csv"

fi_df  = pd.read_csv(FI_CSV)
TOP_170 = fi_df["feature"].head(170).tolist()

# ── Загружаем gender embeddings ───────────────────────────────────────────────
mcc_emb = pd.read_csv(MCC_EMB_CSV)[["mcc_code","mcc_male_sim","mcc_female_sim","mcc_gender_diff"]]
tr_emb  = pd.read_csv(TR_EMB_CSV)[["tr_type","tr_male_sim","tr_female_sim","tr_gender_diff"]]
mcc_emb["mcc_code"] = mcc_emb["mcc_code"].astype("int16")
tr_emb["tr_type"]   = tr_emb["tr_type"].astype("int16")

TOP_MCC_N = 20


# ── helper functions ──────────────────────────────────────────────────────────
def extra_features(df):
    df = df.copy()
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    df["hour"] = df["tr_datetime"].dt.hour
    hour_shares = (
        df.groupby(["customer_id","hour"]).size()
          .unstack(fill_value=0).astype("float32")
    )
    hour_shares = hour_shares.div(hour_shares.sum(axis=1), axis=0)
    hour_shares.columns = [f"hour_share_{int(c)}" for c in hour_shares.columns]
    pos_g = df[df["amount"] > 0].groupby("customer_id")["amount"]
    neg_g = df[df["amount"] < 0].groupby("customer_id")["amount"]
    pos_stats = pos_g.agg(["min","max","median"])
    pos_stats.columns = ["pos_min","pos_max","pos_median"]
    neg_stats = neg_g.agg(["min","max","median"])
    neg_stats.columns = ["neg_min","neg_max","neg_median"]
    return pd.concat([hour_shares, pos_stats, neg_stats], axis=1).fillna(0)


def mcc_interaction_features(df, top_mcc_codes):
    df = df.copy()
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    df["hour"]       = df["tr_datetime"].dt.hour.astype("int8")
    df["dow"]        = df["tr_datetime"].dt.dayofweek.astype("int8")
    df["is_morning"]   = ((df["hour"]>=6)  & (df["hour"]<12)).astype("int8")
    df["is_afternoon"] = ((df["hour"]>=12) & (df["hour"]<18)).astype("int8")
    df["is_evening"]   = ((df["hour"]>=18) & (df["hour"]<23)).astype("int8")
    df["is_night"]     = ((df["hour"]>=23) | (df["hour"]<6)).astype("int8")
    df["is_weekend"]   = (df["dow"]>=5).astype("int8")
    parts = []
    for mcc in top_mcc_codes:
        sub = df[df["mcc_code"]==mcc].groupby("customer_id")
        if len(sub) == 0:
            continue
        stats = pd.DataFrame({
            f"mcc{mcc}_morning_ratio":   sub["is_morning"].mean().astype("float32"),
            f"mcc{mcc}_afternoon_ratio": sub["is_afternoon"].mean().astype("float32"),
            f"mcc{mcc}_evening_ratio":   sub["is_evening"].mean().astype("float32"),
            f"mcc{mcc}_night_ratio":     sub["is_night"].mean().astype("float32"),
            f"mcc{mcc}_weekend_ratio":   sub["is_weekend"].mean().astype("float32"),
            f"mcc{mcc}_q25":    sub["amount"].quantile(0.25).astype("float32"),
            f"mcc{mcc}_q75":    sub["amount"].quantile(0.75).astype("float32"),
            f"mcc{mcc}_median": sub["amount"].median().astype("float32"),
        })
        parts.append(stats)
    return pd.concat(parts, axis=1).fillna(0) if parts else pd.DataFrame()


def gender_embedding_features(df, mcc_emb, tr_emb):
    """
    Для каждого клиента вычисляет агрегированные gender-similarity фичи.
    """
    df = df.merge(mcc_emb, on="mcc_code", how="left")
    df = df.merge(tr_emb,  on="tr_type",  how="left")

    # Заполняем NaN нулями (редкие коды)
    for c in ["mcc_male_sim","mcc_female_sim","mcc_gender_diff",
              "tr_male_sim","tr_female_sim","tr_gender_diff"]:
        df[c] = df[c].fillna(0).astype("float32")

    df["abs_amount"] = df["amount"].abs()

    g = df.groupby("customer_id")

    feats = pd.DataFrame({
        # MCC gender scores
        "mcc_gender_diff_mean":   g["mcc_gender_diff"].mean().astype("float32"),
        "mcc_gender_diff_std":    g["mcc_gender_diff"].std().fillna(0).astype("float32"),
        "mcc_male_sim_mean":      g["mcc_male_sim"].mean().astype("float32"),
        "mcc_female_sim_mean":    g["mcc_female_sim"].mean().astype("float32"),
        # TR gender scores
        "tr_gender_diff_mean":    g["tr_gender_diff"].mean().astype("float32"),
        "tr_gender_diff_std":     g["tr_gender_diff"].std().fillna(0).astype("float32"),
        "tr_male_sim_mean":       g["tr_male_sim"].mean().astype("float32"),
        "tr_female_sim_mean":     g["tr_female_sim"].mean().astype("float32"),
    })

    # Amount-weighted MCC gender diff
    def amt_weighted(group):
        w = group["abs_amount"].values
        d = group["mcc_gender_diff"].values
        total_w = w.sum()
        return float((w * d).sum() / total_w) if total_w > 0 else 0.0

    feats["mcc_gender_diff_amt_wt"] = (
        df.groupby("customer_id").apply(amt_weighted).astype("float32")
    )

    return feats


def get_customer_order(df):
    return df["customer_id"].drop_duplicates().values


def attach(X, extra_df, cust_order):
    if extra_df.empty:
        return X
    aligned = extra_df.reindex(cust_order).fillna(0).reset_index(drop=True)
    new_cols = [c for c in aligned.columns if c not in X.columns]
    return pd.concat([X.reset_index(drop=True), aligned[new_cols]], axis=1)


# ── Загрузка и генерация ──────────────────────────────────────────────────────
print("Loading splits...")
train_df = load_split("train")
val_df   = load_split("val")
test_df  = load_split("test")

top_mcc_codes = train_df["mcc_code"].value_counts().nlargest(TOP_MCC_N).index.tolist()

print("Generating base features...")
X_train, y_train, top_mcc, top_tr = generate_features(train_df)
X_val,   y_val,   _,       _      = generate_features(val_df,  top_mcc=top_mcc, top_tr=top_tr)
X_test,  y_test,  _,       _      = generate_features(test_df, top_mcc=top_mcc, top_tr=top_tr)

print("Adding target encoding...")
X_train, X_val, X_test = add_target_encoding(
    X_train, y_train, X_val, X_test, train_df, val_df, test_df)

print("Computing extra features (hour shares + pos/neg stats)...")
extra_tr_f = extra_features(train_df)
extra_vl_f = extra_features(val_df)
extra_te_f = extra_features(test_df)

print(f"Computing MCC×time interactions (top-{TOP_MCC_N})...")
int_tr = mcc_interaction_features(train_df, top_mcc_codes)
int_vl = mcc_interaction_features(val_df,   top_mcc_codes)
int_te = mcc_interaction_features(test_df,  top_mcc_codes)

print("Computing gender embedding features...")
gemb_tr = gender_embedding_features(train_df, mcc_emb, tr_emb)
gemb_vl = gender_embedding_features(val_df,   mcc_emb, tr_emb)
gemb_te = gender_embedding_features(test_df,  mcc_emb, tr_emb)

cust_train = get_customer_order(train_df)
cust_val   = get_customer_order(val_df)
cust_test  = get_customer_order(test_df)
del train_df, val_df, test_df

for extra, cust in [(extra_tr_f, cust_train), (extra_vl_f, cust_val), (extra_te_f, cust_test)]:
    pass  # attach below

X_train = attach(X_train, extra_tr_f, cust_train)
X_val   = attach(X_val,   extra_vl_f, cust_val)
X_test  = attach(X_test,  extra_te_f, cust_test)
X_train = attach(X_train, int_tr,     cust_train)
X_val   = attach(X_val,   int_vl,     cust_val)
X_test  = attach(X_test,  int_te,     cust_test)
X_train = attach(X_train, gemb_tr, cust_train)
X_val   = attach(X_val,   gemb_vl, cust_val)
X_test  = attach(X_test,  gemb_te, cust_test)
del extra_tr_f, extra_vl_f, extra_te_f, int_tr, int_vl, int_te, gemb_tr, gemb_vl, gemb_te

# Кандидаты
EXTRA_NAMES = [f"hour_share_{h}" for h in range(24)] + \
              ["pos_min","pos_max","pos_median","neg_min","neg_max","neg_median"]
INT_NAMES   = [f"mcc{m}_{s}" for m in top_mcc_codes
               for s in ("morning_ratio","afternoon_ratio","evening_ratio",
                         "night_ratio","weekend_ratio","q25","q75","median")]
GEMB_NAMES  = ["mcc_gender_diff_mean","mcc_gender_diff_std",
               "mcc_male_sim_mean","mcc_female_sim_mean",
               "mcc_gender_diff_amt_wt",
               "tr_gender_diff_mean","tr_gender_diff_std",
               "tr_male_sim_mean","tr_female_sim_mean"]

candidates = list(dict.fromkeys(
    TOP_170
    + [f for f in EXTRA_NAMES if f not in TOP_170]
    + [f for f in INT_NAMES   if f not in TOP_170]
    + [f for f in GEMB_NAMES  if f not in TOP_170]
))
candidates = [f for f in candidates if f in X_train.columns]
print(f"Candidate features: {len(candidates)}")

X_tr_c = X_train[candidates]
X_vl_c = X_val[candidates]


# ── Шаг 1: быстрый LGB → importance → top-200 ────────────────────────────────
print("\nStep 1: feature selection (single LGB)...")
probe = lgb.LGBMClassifier(n_estimators=3000, random_state=0, **LGB_PARAMS)
probe.fit(X_tr_c, y_train,
          eval_set=[(X_vl_c, y_val)],
          callbacks=[lgb.early_stopping(150, verbose=False),
                     lgb.log_evaluation(period=200)])
best_iter = int(probe.best_iteration_ * 1.1)

fi = pd.Series(probe.feature_importances_, index=candidates).sort_values(ascending=False)
top200 = fi.head(200).index.tolist()

gemb_in_top = [f for f in top200 if f in set(GEMB_NAMES)]
print(f"  best_iter (×1.1): {best_iter}")
print(f"  Gender-embedding features in top-200: {len(gemb_in_top)}")
for f in gemb_in_top:
    print(f"    {f}  importance={fi[f]:.0f}")

fi_full = pd.DataFrame({"feature": fi.index, "importance": fi.values})
fi_full.to_csv(f"{RESULTS_DIR}/feature_importance_all.csv", index=False)

del X_tr_c, X_vl_c
X_train_s = X_train[top200]
X_val_s   = X_val[top200]
X_test_s  = X_test[top200]
del X_train, X_val, X_test


# ── Шаг 2: 5-fold ensemble ────────────────────────────────────────────────────
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


print(f"\nStep 2: 5-fold ensemble on top-200...")
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
        "from_top170":       len([f for f in top200 if f in set(TOP_170)]),
        "mcc_interactions":  len([f for f in top200 if f in set(INT_NAMES)]),
        "gender_embeddings": len(gemb_in_top),
        "extra_2021old":     len([f for f in top200 if f in set(EXTRA_NAMES)]),
    },
    "gender_emb_features_selected": gemb_in_top,
    "comment": (
        f"Gender-embedding-200: top-170 + MCC×time interactions + "
        f"9 gender-embedding фич (spaCy ru_core_news_md, 300d Word2Vec, "
        f"cosine similarity с male/female word centroids). "
        f"Отобраны top-200 из {len(candidates)} кандидатов. "
        f"Gender-emb в top-200: {len(gemb_in_top)}. "
        f"Модель: LightGBM 5-fold ensemble, threshold=0.5."
    ),
})
with open(OVERALL_PATH, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nUpdated: {OVERALL_PATH}")
