"""
solution_super_200.py — super-ensemble: все типы фич → top-200 → 5-fold LGB.

Типы фич:
  A. 376 base (solution.py: amount stats, temporal, MCC/TR per-code stats, target encoding)
  B. 30  extra (per-hour shares 0-23, pos/neg min/max/median)
  C. 160 MCC×time interactions (top-20 MCC × morning/afternoon/evening/night/weekend/q25/q75/median)
  D. 60  per-MCC depth (top-20 MCC × neg_ratio, max_amount, amount_cv, tx_count_ratio)
  E. 30  holiday proximity (10 праздников × mean/std/near7)
  F. 9   gender embeddings (spaCy ru_core_news_md cosine sim)
  Итого кандидатов: ~665. Финал: top-200 по LGB importance + 5-fold ensemble.
"""
import os, time, json, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb

from solution import load_split, generate_features, add_target_encoding, LGB_PARAMS

RESULTS_DIR  = "results/solution_2026_04_15_super_200"
OVERALL_PATH = "results/overall.json"
BRANCH       = "solution_2026_04_15_super_200"
MCC_EMB_CSV  = "results/gender_embeddings/mcc_gender.csv"
TR_EMB_CSV   = "results/gender_embeddings/tr_gender.csv"

TOP_MCC_N = 20

HOLIDAYS = {
    "new_year": (1,1), "xmas_orth": (1,7), "feb23": (2,23), "mar8": (3,8),
    "may1": (5,1), "may9": (5,9), "halloween": (10,31),
    "nov7": (11,7), "xmas_west": (12,25), "new_year_e": (12,31),
}

# ── Helper functions ──────────────────────────────────────────────────────────

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
        if offset == -1:
            results = d
        else:
            results = np.where(np.abs(d) < np.abs(results), d, results)
    return results.astype(np.int32)


def extra_features(df):
    """B: per-hour shares + pos/neg min/max/median"""
    df = df.copy()
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    df["hour"] = df["tr_datetime"].dt.hour
    hs = df.groupby(["customer_id","hour"]).size().unstack(fill_value=0).astype("float32")
    hs = hs.div(hs.sum(axis=1), axis=0)
    hs.columns = [f"hour_share_{int(c)}" for c in hs.columns]
    pos = df[df["amount"]>0].groupby("customer_id")["amount"].agg(["min","max","median"])
    pos.columns = ["pos_min","pos_max","pos_median"]
    neg = df[df["amount"]<0].groupby("customer_id")["amount"].agg(["min","max","median"])
    neg.columns = ["neg_min","neg_max","neg_median"]
    return pd.concat([hs, pos, neg], axis=1).fillna(0)


def mcc_interaction_features(df, top_mcc_codes):
    """C: MCC × time/percentile  +  D: MCC depth (neg_ratio, max, cv, cnt_ratio)"""
    df = df.copy()
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    df["hour"] = df["tr_datetime"].dt.hour.astype("int8")
    df["dow"]  = df["tr_datetime"].dt.dayofweek.astype("int8")
    df["is_morning"]   = ((df["hour"]>=6)  & (df["hour"]<12)).astype("int8")
    df["is_afternoon"] = ((df["hour"]>=12) & (df["hour"]<18)).astype("int8")
    df["is_evening"]   = ((df["hour"]>=18) & (df["hour"]<23)).astype("int8")
    df["is_night"]     = ((df["hour"]>=23) | (df["hour"]<6)).astype("int8")
    df["is_weekend"]   = (df["dow"]>=5).astype("int8")
    df["is_neg"]       = (df["amount"]<0).astype("int8")
    total_tx = df.groupby("customer_id").size().rename("_total")

    parts = []
    for mcc in top_mcc_codes:
        sub = df[df["mcc_code"]==mcc]
        if len(sub) == 0:
            continue
        g = sub.groupby("customer_id")
        mcc_tx = g.size().rename(f"_mcc{mcc}_n")
        stats = pd.DataFrame({
            f"mcc{mcc}_morning_ratio":   g["is_morning"].mean().astype("float32"),
            f"mcc{mcc}_afternoon_ratio": g["is_afternoon"].mean().astype("float32"),
            f"mcc{mcc}_evening_ratio":   g["is_evening"].mean().astype("float32"),
            f"mcc{mcc}_night_ratio":     g["is_night"].mean().astype("float32"),
            f"mcc{mcc}_weekend_ratio":   g["is_weekend"].mean().astype("float32"),
            f"mcc{mcc}_q25":    g["amount"].quantile(0.25).astype("float32"),
            f"mcc{mcc}_q75":    g["amount"].quantile(0.75).astype("float32"),
            f"mcc{mcc}_median": g["amount"].median().astype("float32"),
            f"mcc{mcc}_neg_ratio":   g["is_neg"].mean().astype("float32"),
            f"mcc{mcc}_max_amount":  g["amount"].max().astype("float32"),
            f"mcc{mcc}_amount_cv":   (g["amount"].std() / (g["amount"].mean().abs()+1e-6)).astype("float32"),
            f"mcc{mcc}_cnt_ratio":   (mcc_tx / total_tx).astype("float32"),
        })
        parts.append(stats)
    return pd.concat(parts, axis=1).fillna(0) if parts else pd.DataFrame()


def holiday_features(df):
    """E: 10 праздников × mean/std/near7"""
    df = df.copy()
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    parts = []
    for name, (month, day) in HOLIDAYS.items():
        col = f"_hd_{name}"
        df[col] = _signed_days(df["tr_datetime"], month, day)
        g = df.groupby("customer_id")[col]
        parts.append(pd.DataFrame({
            f"hd_{name}_mean":  g.mean().astype("float32"),
            f"hd_{name}_std":   g.std().fillna(0).astype("float32"),
            f"hd_{name}_near7": g.apply(lambda x: (x.abs()<=7).mean()).astype("float32"),
        }))
    return pd.concat(parts, axis=1)


def gender_emb_features(df, mcc_emb, tr_emb):
    """F: gender embedding cosine similarities"""
    df = df.merge(mcc_emb, on="mcc_code", how="left")
    df = df.merge(tr_emb,  on="tr_type",  how="left")
    for c in ["mcc_male_sim","mcc_female_sim","mcc_gender_diff",
              "tr_male_sim","tr_female_sim","tr_gender_diff"]:
        df[c] = df[c].fillna(0).astype("float32")
    df["abs_amount"] = df["amount"].abs()
    g = df.groupby("customer_id")

    def amt_wt(grp):
        w = grp["abs_amount"].values
        d = grp["mcc_gender_diff"].values
        s = w.sum()
        return float((w*d).sum()/s) if s>0 else 0.0

    feats = pd.DataFrame({
        "mcc_gender_diff_mean":   g["mcc_gender_diff"].mean().astype("float32"),
        "mcc_gender_diff_std":    g["mcc_gender_diff"].std().fillna(0).astype("float32"),
        "mcc_male_sim_mean":      g["mcc_male_sim"].mean().astype("float32"),
        "mcc_female_sim_mean":    g["mcc_female_sim"].mean().astype("float32"),
        "mcc_gender_diff_amt_wt": df.groupby("customer_id").apply(amt_wt).astype("float32"),
        "tr_gender_diff_mean":    g["tr_gender_diff"].mean().astype("float32"),
        "tr_gender_diff_std":     g["tr_gender_diff"].std().fillna(0).astype("float32"),
        "tr_male_sim_mean":       g["tr_male_sim"].mean().astype("float32"),
        "tr_female_sim_mean":     g["tr_female_sim"].mean().astype("float32"),
    })
    return feats


def get_cust_order(df):
    return df["customer_id"].drop_duplicates().values

def attach(X, feat_df, cust_order):
    if feat_df is None or feat_df.empty:
        return X
    aligned = feat_df.reindex(cust_order).fillna(0).reset_index(drop=True)
    new_cols = [c for c in aligned.columns if c not in X.columns]
    return pd.concat([X.reset_index(drop=True), aligned[new_cols]], axis=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    mcc_emb = pd.read_csv(MCC_EMB_CSV)[["mcc_code","mcc_male_sim","mcc_female_sim","mcc_gender_diff"]]
    tr_emb  = pd.read_csv(TR_EMB_CSV)[["tr_type","tr_male_sim","tr_female_sim","tr_gender_diff"]]
    mcc_emb["mcc_code"] = mcc_emb["mcc_code"].astype("int16")
    tr_emb["tr_type"]   = tr_emb["tr_type"].astype("int16")

    print("Loading splits...")
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    top_mcc_codes = train_df["mcc_code"].value_counts().nlargest(TOP_MCC_N).index.tolist()

    print("Generating base features (A)...")
    X_train, y_train, top_mcc, top_tr = generate_features(train_df)
    X_val,   y_val,   _,       _      = generate_features(val_df,  top_mcc=top_mcc, top_tr=top_tr)
    X_test,  y_test,  _,       _      = generate_features(test_df, top_mcc=top_mcc, top_tr=top_tr)

    print("Adding target encoding...")
    X_train, X_val, X_test = add_target_encoding(
        X_train, y_train, X_val, X_test, train_df, val_df, test_df)

    print("Computing extra features (B)...")
    b_tr = extra_features(train_df)
    b_vl = extra_features(val_df)
    b_te = extra_features(test_df)

    print(f"Computing MCC interactions + depth (C+D, top-{TOP_MCC_N})...")
    cd_tr = mcc_interaction_features(train_df, top_mcc_codes)
    cd_vl = mcc_interaction_features(val_df,   top_mcc_codes)
    cd_te = mcc_interaction_features(test_df,  top_mcc_codes)

    print("Computing holiday features (E)...")
    e_tr = holiday_features(train_df)
    e_vl = holiday_features(val_df)
    e_te = holiday_features(test_df)

    print("Computing gender embedding features (F)...")
    f_tr = gender_emb_features(train_df, mcc_emb, tr_emb)
    f_vl = gender_emb_features(val_df,   mcc_emb, tr_emb)
    f_te = gender_emb_features(test_df,  mcc_emb, tr_emb)

    cust_train = get_cust_order(train_df)
    cust_val   = get_cust_order(val_df)
    cust_test  = get_cust_order(test_df)
    del train_df, val_df, test_df
    gc.collect()

    for extra, vl, te, co_tr, co_vl, co_te in [
        (b_tr,  b_vl,  b_te,  cust_train, cust_val, cust_test),
        (cd_tr, cd_vl, cd_te, cust_train, cust_val, cust_test),
        (e_tr,  e_vl,  e_te,  cust_train, cust_val, cust_test),
        (f_tr,  f_vl,  f_te,  cust_train, cust_val, cust_test),
    ]:
        X_train = attach(X_train, extra, co_tr)
        X_val   = attach(X_val,   vl,    co_vl)
        X_test  = attach(X_test,  te,    co_te)

    del b_tr,b_vl,b_te, cd_tr,cd_vl,cd_te, e_tr,e_vl,e_te, f_tr,f_vl,f_te
    gc.collect()

    candidates = list(X_train.columns)
    print(f"Total candidate features: {len(candidates)}")

    # Step 1: feature selection
    print("\nStep 1: feature selection...")
    probe = lgb.LGBMClassifier(n_estimators=3000, random_state=0, **LGB_PARAMS)
    probe.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(period=200)])
    best_iter = int(probe.best_iteration_ * 1.1)

    fi = pd.Series(probe.feature_importances_, index=candidates).sort_values(ascending=False)
    top200 = fi.head(200).index.tolist()

    extra_names = set([f"hour_share_{h}" for h in range(24)] +
                      ["pos_min","pos_max","pos_median","neg_min","neg_max","neg_median"])
    int_names = set(f"mcc{m}_{s}" for m in top_mcc_codes
                    for s in ("morning_ratio","afternoon_ratio","evening_ratio",
                              "night_ratio","weekend_ratio","q25","q75","median"))
    depth_names = set(f"mcc{m}_{s}" for m in top_mcc_codes
                      for s in ("neg_ratio","max_amount","amount_cv","cnt_ratio"))
    hol_names = set(f"hd_{n}_{s}" for n in HOLIDAYS for s in ("mean","std","near7"))
    gemb_names = {"mcc_gender_diff_mean","mcc_gender_diff_std","mcc_male_sim_mean",
                  "mcc_female_sim_mean","mcc_gender_diff_amt_wt",
                  "tr_gender_diff_mean","tr_gender_diff_std","tr_male_sim_mean","tr_female_sim_mean"}
    known = extra_names | int_names | depth_names | hol_names | gemb_names

    groups = {"A_base":      [f for f in top200 if f not in known],
              "B_extra":     [f for f in top200 if f in extra_names],
              "C_mcc_time":  [f for f in top200 if f in int_names],
              "D_mcc_depth": [f for f in top200 if f in depth_names],
              "E_holiday":   [f for f in top200 if f in hol_names],
              "F_gender_emb":[f for f in top200 if f in gemb_names]}

    print(f"  best_iter (×1.1): {best_iter}")
    print("  Feature groups in top-200:")
    for grp, feats in groups.items():
        print(f"    {grp}: {len(feats)}")

    fi_full = pd.DataFrame({"feature": fi.index, "importance": fi.values})
    fi_full.to_csv(f"{RESULTS_DIR}/feature_importance_all.csv", index=False)

    X_train_s = X_train[top200]
    X_val_s   = X_val[top200]
    X_test_s  = X_test[top200]
    del X_train, X_val, X_test
    gc.collect()

    # Step 2: 5-fold ensemble
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

    print("\nStep 2: 5-fold ensemble on top-200...")
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
    print(f"\nROC AUC > 0.88: {'OK' if auc>0.88 else 'FAIL'}  |  Accuracy > 0.80: {'OK' if acc>0.80 else 'FAIL'}")

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
        "feature_groups_in_top200": {k: len(v) for k, v in groups.items()},
        "total_candidates": len(candidates),
        "comment": (
            f"Super-200: все типы фич (A=base376, B=hour_shares+pos/neg, "
            f"C=MCC×time, D=MCC_depth(neg_ratio/max/cv/cnt), E=holidays, F=gender_emb) — "
            f"{len(candidates)} кандидатов → top-200 по LGB importance. "
            f"Модель: LightGBM 5-fold ensemble на train+val, threshold=0.5."
        ),
    })
    with open(OVERALL_PATH, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"\nUpdated: {OVERALL_PATH}")

    return {"auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "train_time": train_time}


if __name__ == "__main__":
    main()
