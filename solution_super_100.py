"""
solution_super_100.py — те же 685 кандидатов, top-100 по super_200 importance.

Использует feature_importance из results/solution_2026_04_15_super_200/
чтобы не пересчитывать importance заново.
"""
import os, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb

from solution import load_split, generate_features, add_target_encoding, LGB_PARAMS

RESULTS_DIR  = "results/solution_2026_04_15_super_100"
os.makedirs(RESULTS_DIR, exist_ok=True)
OVERALL_PATH = "results/overall.json"
BRANCH       = "solution_2026_04_15_super_100"
MCC_EMB_CSV  = "results/gender_embeddings/mcc_gender.csv"
TR_EMB_CSV   = "results/gender_embeddings/tr_gender.csv"
FI_CSV       = "results/solution_2026_04_15_super_200/feature_importance_all.csv"

TOP_MCC_N = 20
HOLIDAYS = {
    'new_year':   (1,  1), 'xmas_orth':  (1,  7), 'feb23':  (2, 23),
    'mar8':       (3,  8), 'may1':       (5,  1), 'may9':   (5,  9),
    'halloween':  (10, 31), 'nov7':      (11,  7), 'xmas_west': (12, 25),
    'new_year_e': (12, 31),
}

fi_df = pd.read_csv(FI_CSV)
top100 = fi_df["feature"].head(100).tolist()
print(f"Loaded top-100 from {FI_CSV}")
print(f"  Top-5: {top100[:5]}")

mcc_emb = pd.read_csv(MCC_EMB_CSV)[["mcc_code","mcc_male_sim","mcc_female_sim","mcc_gender_diff"]]
tr_emb  = pd.read_csv(TR_EMB_CSV)[["tr_type","tr_male_sim","tr_female_sim","tr_gender_diff"]]
mcc_emb["mcc_code"] = mcc_emb["mcc_code"].astype("int16")
tr_emb["tr_type"]   = tr_emb["tr_type"].astype("int16")


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
    pos_stats = pos_g.agg(["min","max","median"]); pos_stats.columns = ["pos_min","pos_max","pos_median"]
    neg_stats = neg_g.agg(["min","max","median"]); neg_stats.columns = ["neg_min","neg_max","neg_median"]
    return pd.concat([hour_shares, pos_stats, neg_stats], axis=1).fillna(0)


def _signed_days(dates_ts, month, day):
    year = dates_ts.dt.year.values
    ts   = dates_ts.values.astype("datetime64[D]").astype(np.int64)
    results = np.empty(len(ts), dtype=np.int64)
    for offset in (-1, 0, 1):
        try:
            h = pd.to_datetime({"year": year+offset, "month": month, "day": day}
                               ).values.astype("datetime64[D]").astype(np.int64)
        except Exception:
            continue
        d = ts - h
        if offset == -1: results = d
        else:
            mask = np.abs(d) < np.abs(results)
            results = np.where(mask, d, results)
    return results.astype(np.int32)


def holiday_features(df):
    df = df.copy()
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    parts = []
    for name, (month, day) in HOLIDAYS.items():
        col = f"_hd_{name}"
        df[col] = _signed_days(df["tr_datetime"], month, day)
        grp = df.groupby("customer_id")[col]
        stats = pd.DataFrame({
            f"hd_{name}_mean":  grp.mean().astype("float32"),
            f"hd_{name}_std":   grp.std().fillna(0).astype("float32"),
            f"hd_{name}_near7": grp.apply(lambda x: (x.abs()<=7).mean()).astype("float32"),
        })
        parts.append(stats)
    return pd.concat(parts, axis=1)


def mcc_interaction_features(df, top_mcc_codes):
    df = df.copy()
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    df["hour"]      = df["tr_datetime"].dt.hour.astype("int8")
    df["dow"]       = df["tr_datetime"].dt.dayofweek.astype("int8")
    df["is_morning"]   = ((df["hour"]>=6)  & (df["hour"]<12)).astype("int8")
    df["is_afternoon"] = ((df["hour"]>=12) & (df["hour"]<18)).astype("int8")
    df["is_evening"]   = ((df["hour"]>=18) & (df["hour"]<23)).astype("int8")
    df["is_night"]     = ((df["hour"]>=23) | (df["hour"]<6)).astype("int8")
    df["is_weekend"]   = (df["dow"]>=5).astype("int8")
    parts = []
    for mcc in top_mcc_codes:
        sub = df[df["mcc_code"]==mcc].groupby("customer_id")
        if len(sub) == 0: continue
        parts.append(pd.DataFrame({
            f"mcc{mcc}_morning_ratio":   sub["is_morning"].mean().astype("float32"),
            f"mcc{mcc}_afternoon_ratio": sub["is_afternoon"].mean().astype("float32"),
            f"mcc{mcc}_evening_ratio":   sub["is_evening"].mean().astype("float32"),
            f"mcc{mcc}_night_ratio":     sub["is_night"].mean().astype("float32"),
            f"mcc{mcc}_weekend_ratio":   sub["is_weekend"].mean().astype("float32"),
            f"mcc{mcc}_q25":    sub["amount"].quantile(0.25).astype("float32"),
            f"mcc{mcc}_q75":    sub["amount"].quantile(0.75).astype("float32"),
            f"mcc{mcc}_median": sub["amount"].median().astype("float32"),
        }))
    return pd.concat(parts, axis=1).fillna(0) if parts else pd.DataFrame()


def mcc_extended_features(df, top_mcc_codes):
    parts = []
    for mcc in top_mcc_codes:
        sub = df[df["mcc_code"]==mcc]
        if len(sub) == 0: continue
        g = sub.groupby("customer_id")["amount"]
        neg_ratio = (sub["amount"]<0).groupby(sub["customer_id"]).mean().rename(f"mcc{mcc}_neg_ratio")
        agg = pd.DataFrame({
            f"mcc{mcc}_max_amount": g.max().astype("float32"),
            f"mcc{mcc}_amount_cv":  (g.std()/(g.mean().abs()+1e-6)).astype("float32"),
        })
        agg[f"mcc{mcc}_neg_ratio"] = neg_ratio.astype("float32")
        parts.append(agg)
    return pd.concat(parts, axis=1).fillna(0) if parts else pd.DataFrame()


def gender_embedding_features(df, mcc_emb, tr_emb):
    df = df.merge(mcc_emb, on="mcc_code", how="left").merge(tr_emb, on="tr_type", how="left")
    for c in ["mcc_male_sim","mcc_female_sim","mcc_gender_diff","tr_male_sim","tr_female_sim","tr_gender_diff"]:
        df[c] = df[c].fillna(0).astype("float32")
    df["abs_amount"] = df["amount"].abs()
    g = df.groupby("customer_id")
    feats = pd.DataFrame({
        "mcc_gender_diff_mean":  g["mcc_gender_diff"].mean().astype("float32"),
        "mcc_gender_diff_std":   g["mcc_gender_diff"].std().fillna(0).astype("float32"),
        "mcc_male_sim_mean":     g["mcc_male_sim"].mean().astype("float32"),
        "mcc_female_sim_mean":   g["mcc_female_sim"].mean().astype("float32"),
        "tr_gender_diff_mean":   g["tr_gender_diff"].mean().astype("float32"),
        "tr_gender_diff_std":    g["tr_gender_diff"].std().fillna(0).astype("float32"),
        "tr_male_sim_mean":      g["tr_male_sim"].mean().astype("float32"),
        "tr_female_sim_mean":    g["tr_female_sim"].mean().astype("float32"),
    })
    def amt_weighted(group):
        w = group["abs_amount"].values; d = group["mcc_gender_diff"].values; s = w.sum()
        return float((w*d).sum()/s) if s > 0 else 0.0
    feats["mcc_gender_diff_amt_wt"] = (
        df.groupby("customer_id").apply(amt_weighted, include_groups=False).astype("float32"))
    return feats


def get_customer_order(df):
    return df["customer_id"].drop_duplicates().values

def attach(X, extra_df, cust_order):
    if extra_df.empty: return X
    aligned = extra_df.reindex(cust_order).fillna(0).reset_index(drop=True)
    new_cols = [c for c in aligned.columns if c not in X.columns]
    return pd.concat([X.reset_index(drop=True), aligned[new_cols]], axis=1)


print("Loading splits...")
train_df = load_split("train"); val_df = load_split("val"); test_df = load_split("test")
top_mcc_codes = train_df["mcc_code"].value_counts().nlargest(TOP_MCC_N).index.tolist()

print("Generating base features...")
X_train, y_train, top_mcc, top_tr = generate_features(train_df)
X_val,   y_val,   _,       _      = generate_features(val_df,  top_mcc=top_mcc, top_tr=top_tr)
X_test,  y_test,  _,       _      = generate_features(test_df, top_mcc=top_mcc, top_tr=top_tr)

print("Adding target encoding...")
X_train, X_val, X_test = add_target_encoding(
    X_train, y_train, X_val, X_test, train_df, val_df, test_df)

for name, fn, args in [
    ("[B] hour+pos/neg", extra_features, []),
    ("[C] holidays",     holiday_features, []),
    ("[D] MCC×time",     mcc_interaction_features, [top_mcc_codes]),
    ("[E] MCC ext",      mcc_extended_features, [top_mcc_codes]),
    ("[F] gender emb",   gender_embedding_features, [mcc_emb, tr_emb]),
]:
    print(f"Computing {name}...")
    f_tr = fn(train_df, *args); f_vl = fn(val_df, *args); f_te = fn(test_df, *args)
    cust_tr = get_customer_order(train_df)
    cust_vl = get_customer_order(val_df)
    cust_te = get_customer_order(test_df)
    X_train = attach(X_train, f_tr, cust_tr)
    X_val   = attach(X_val,   f_vl, cust_vl)
    X_test  = attach(X_test,  f_te, cust_te)

del train_df, val_df, test_df

# Фильтруем по top-100 из super_200 importance
selected = [f for f in top100 if f in X_train.columns]
print(f"\nFeatures selected: {len(selected)} / 100")
X_train = X_train[selected]; X_val = X_val[selected]; X_test = X_test[selected]


# 5-fold ensemble
print("\nTraining 5-fold ensemble on top-100...")
probe = lgb.LGBMClassifier(n_estimators=3000, random_state=0, **LGB_PARAMS)
probe.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(150, verbose=False),
                     lgb.log_evaluation(period=200)])
best_iter = int(probe.best_iteration_ * 1.1)
print(f"  best_iter (×1.1): {best_iter}")

X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
t0 = time.perf_counter()
test_probas = []
for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_full, y_full)):
    m = lgb.LGBMClassifier(n_estimators=best_iter, random_state=fold, **LGB_PARAMS)
    m.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx], callbacks=[lgb.log_evaluation(period=-1)])
    oof_auc = roc_auc_score(y_full.iloc[vl_idx], m.predict_proba(X_full.iloc[vl_idx])[:,1])
    test_probas.append(m.predict_proba(X_test)[:,1])
    print(f"  fold {fold+1}/5  OOF AUC: {oof_auc:.4f}")
train_time = round(time.perf_counter()-t0, 1)

y_pred_proba = np.mean(test_probas, axis=0)
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
print(f"\nROC AUC > 0.88: {'OK' if auc>0.88 else 'FAIL'}  |  Accuracy > 0.80: {'OK' if acc>0.80 else 'FAIL'}")

with open(OVERALL_PATH) as f:
    overall = json.load(f)
overall = [e for e in overall if e["branch"] != BRANCH]
overall.append({
    "branch": BRANCH, "date": "2026-04-15",
    "metrics": {"auc_score": auc, "accuracy_score": acc,
                "precision_score": prec, "recall_score": rec},
    "timing": {"train_time_s": train_time},
    "n_features": len(selected),
    "comment": (
        f"Super-ensemble top-100: top-100 фич из финального feature importance "
        f"(685 кандидатов, все типы фич). "
        f"Модель: LightGBM 5-fold ensemble, threshold=0.5."
    ),
})
with open(OVERALL_PATH, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nUpdated: {OVERALL_PATH}")
