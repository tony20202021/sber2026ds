import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


# ── Data loading ─────────────────────────────────────────────────────────────

def load_split(split, path="data/"):
    df = pd.read_csv(f"{path}{split}.csv",
                     usecols=['customer_id', 'tr_datetime', 'mcc_code',
                               'tr_type', 'amount', 'term_id', 'gender'],
                     dtype={'customer_id': 'int32', 'mcc_code': 'int16',
                            'tr_type': 'int16', 'amount': 'float32',
                            'gender': 'int8'})
    df['term_id_missing'] = df['term_id'].isna().astype('int8')
    df.drop(columns=['term_id'], inplace=True)
    df['tr_datetime'] = pd.to_datetime(df['tr_datetime'], errors='coerce')
    df.sort_values(['customer_id', 'tr_datetime'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Feature engineering ──────────────────────────────────────────────────────

def generate_features(df, top_mcc=None, top_tr=None):
    g = df.groupby('customer_id', sort=False)
    agg_parts = []

    # ---- global amount stats ----
    base = g['amount'].agg(['count', 'sum', 'mean', 'std', 'min', 'max'])
    base.columns = ['tx_count', 'amount_sum', 'amount_mean',
                    'amount_std', 'amount_min', 'amount_max']
    base['amount_q25']    = g['amount'].quantile(0.25)
    base['amount_q75']    = g['amount'].quantile(0.75)
    base['amount_median'] = g['amount'].quantile(0.5)
    base['amount_cv']     = base['amount_std'] / (base['amount_mean'].abs() + 1e-6)
    base['amount_range']  = base['amount_max'] - base['amount_min']
    agg_parts.append(base)

    # ---- pos / neg ----
    pos = (df[df['amount'] > 0].groupby('customer_id', sort=False)['amount']
           .agg(pos_count='count', pos_sum='sum', pos_mean='mean', pos_std='std'))
    neg = (df[df['amount'] < 0].groupby('customer_id', sort=False)['amount']
           .agg(neg_count='count', neg_sum='sum', neg_mean='mean', neg_std='std'))
    pn = pd.concat([pos, neg], axis=1)
    pn['pos_ratio']            = pn['pos_count'] / (base['tx_count'] + 1e-6)
    pn['pos_neg_amount_ratio'] = pn['pos_sum'] / (pn['neg_sum'].abs() + 1e-6)
    agg_parts.append(pn)

    # ---- diversity ----
    div = g.agg(mcc_nunique=('mcc_code', 'nunique'),
                tr_type_nunique=('tr_type', 'nunique'),
                term_missing_ratio=('term_id_missing', 'mean'))
    agg_parts.append(div)

    # ---- temporal ----
    df['hour']         = df['tr_datetime'].dt.hour.astype('int8')
    df['dow']          = df['tr_datetime'].dt.dayofweek.astype('int8')
    df['month']        = df['tr_datetime'].dt.month.astype('int8')
    df['is_weekend']   = (df['dow'] >= 5).astype('int8')
    df['is_morning']   = ((df['hour'] >= 6)  & (df['hour'] < 12)).astype('int8')
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype('int8')
    df['is_evening']   = ((df['hour'] >= 18) & (df['hour'] < 23)).astype('int8')
    df['is_night']     = ((df['hour'] >= 23) | (df['hour'] <  6)).astype('int8')

    g2 = df.groupby('customer_id', sort=False)
    temp = g2.agg(
        hour_mean=('hour', 'mean'), hour_std=('hour', 'std'),
        weekend_ratio=('is_weekend', 'mean'),
        morning_ratio=('is_morning', 'mean'),
        afternoon_ratio=('is_afternoon', 'mean'),
        evening_ratio=('is_evening', 'mean'),
        night_ratio=('is_night', 'mean'),
        month_nunique=('month', 'nunique'),
    )
    for d in range(7):
        temp[f'dow{d}_ratio'] = (df['dow'] == d).groupby(df['customer_id'], sort=False).mean()
    agg_parts.append(temp)

    # ---- active span + frequency ----
    dt_agg = g2['tr_datetime'].agg(['min', 'max'])
    span = pd.DataFrame(index=dt_agg.index)
    span['active_days']  = (dt_agg['max'] - dt_agg['min']).dt.days
    span['tx_per_day']   = base['tx_count'] / (span['active_days'] + 1)
    span['amount_per_day'] = base['amount_sum'] / (span['active_days'] + 1)
    df['delta_days'] = g2['tr_datetime'].transform(
        lambda x: x.diff().dt.total_seconds() / 86400)
    span['avg_days_between_tx'] = g2['delta_days'].mean()
    span['std_days_between_tx'] = g2['delta_days'].std()
    df.drop(columns=['delta_days'], inplace=True)
    agg_parts.append(span)

    # ---- recency features ----
    df['time_frac'] = g2['tr_datetime'].transform(
        lambda x: (x - x.min()).dt.total_seconds() /
                  (max((x.max() - x.min()).total_seconds(), 1)))
    df['is_recent'] = (df['time_frac'] > 0.75).astype('int8')
    df['is_early']  = (df['time_frac'] < 0.25).astype('int8')
    rec_mean   = df[df['is_recent'] == 1].groupby('customer_id', sort=False)['amount'].mean().rename('recent_amount_mean')
    early_mean = df[df['is_early']  == 1].groupby('customer_id', sort=False)['amount'].mean().rename('early_amount_mean')
    rec = pd.concat([g2['is_recent'].mean().rename('recent_ratio'), rec_mean, early_mean], axis=1).fillna(0)
    rec['amount_trend'] = rec['recent_amount_mean'] - rec['early_amount_mean']
    agg_parts.append(rec)
    df.drop(columns=['time_frac', 'is_recent', 'is_early'], inplace=True)

    # ---- entropy ----
    def entropy(series):
        p = series.value_counts(normalize=True)
        return -(p * np.log2(p + 1e-9)).sum()

    ent = pd.DataFrame({
        'mcc_entropy':     g['mcc_code'].apply(entropy),
        'tr_type_entropy': g['tr_type'].apply(entropy),
    })
    agg_parts.append(ent)

    # ---- weekend vs weekday amounts ----
    wknd_mean = df[df['is_weekend'] == 1].groupby('customer_id', sort=False)['amount'].mean().rename('weekend_amount_mean')
    wkdy_mean = df[df['is_weekend'] == 0].groupby('customer_id', sort=False)['amount'].mean().rename('weekday_amount_mean')
    wknd_df = pd.concat([wknd_mean, wkdy_mean], axis=1).fillna(0)
    wknd_df['wknd_wkdy_diff'] = wknd_df['weekend_amount_mean'] - wknd_df['weekday_amount_mean']
    agg_parts.append(wknd_df)

    # ---- amount buckets ----
    bins   = [-np.inf, -10000, -1000, 0, 1000, 10000, np.inf]
    labels = ['vn', 'n', 'nz', 'sp', 'mp', 'lp']
    df['amt_bucket'] = pd.cut(df['amount'], bins=bins, labels=labels)
    bkt = df.groupby(['customer_id', 'amt_bucket'], sort=False, observed=False).size().unstack(fill_value=0)
    bkt = bkt.div(bkt.sum(axis=1), axis=0)
    bkt.columns = [f'bucket_{c}' for c in bkt.columns]
    agg_parts.append(bkt)
    df.drop(columns=['hour', 'dow', 'month', 'is_weekend',
                     'is_morning', 'is_afternoon', 'is_evening', 'is_night',
                     'amt_bucket'], inplace=True)

    # ---- per-MCC amount stats ----
    if top_mcc is None:
        top_mcc = df['mcc_code'].value_counts().nlargest(60).index.tolist()
    mcc_frames = []
    for mcc in top_mcc:
        sub = df[df['mcc_code'] == mcc].groupby('customer_id', sort=False)['amount']
        mcc_frames.extend([
            sub.count().rename(f'mcc{mcc}_cnt'),
            sub.mean().rename(f'mcc{mcc}_mean'),
            sub.std().rename(f'mcc{mcc}_std'),
        ])
    if mcc_frames:
        mcc_df = pd.concat(mcc_frames, axis=1).reindex(base.index).fillna(0)
        for mcc in top_mcc:
            mcc_df[f'mcc{mcc}_share'] = mcc_df[f'mcc{mcc}_cnt'] / (base['tx_count'] + 1e-6)
        agg_parts.append(mcc_df)

    # ---- per-tr_type amount stats ----
    if top_tr is None:
        top_tr = df['tr_type'].value_counts().nlargest(25).index.tolist()
    tr_frames = []
    for tr in top_tr:
        sub = df[df['tr_type'] == tr].groupby('customer_id', sort=False)['amount']
        tr_frames.extend([
            sub.count().rename(f'tr{tr}_cnt'),
            sub.mean().rename(f'tr{tr}_mean'),
        ])
    if tr_frames:
        tr_df = pd.concat(tr_frames, axis=1).reindex(base.index).fillna(0)
        for tr in top_tr:
            tr_df[f'tr{tr}_share'] = tr_df[f'tr{tr}_cnt'] / (base['tx_count'] + 1e-6)
        agg_parts.append(tr_df)

    feats = pd.concat(agg_parts, axis=1).fillna(0)
    feats.reset_index(inplace=True)

    target = df[['customer_id', 'gender']].drop_duplicates().set_index('customer_id')
    feats = feats.join(target, on='customer_id')

    X = feats.drop(columns=['customer_id', 'gender'])
    y = feats['gender']
    return X, y, top_mcc, top_tr


def compute_category_te(df_train, y_train, col, smoothing=20):
    """Compute smoothed target encoding (p_male) for each category value."""
    train_cust_ids = df_train['customer_id'].drop_duplicates().values
    global_m = y_train.mean()
    gender_map = dict(zip(train_cust_ids, y_train.values))
    tx = df_train[['customer_id', col]].copy()
    tx['_g'] = tx['customer_id'].map(gender_map)
    agg = tx.groupby(col)['_g'].agg(['mean', 'count'])
    agg['te'] = (agg['mean'] * agg['count'] + global_m * smoothing) / (agg['count'] + smoothing)
    return agg['te'], global_m


def apply_te_feature(df, te_map, global_mean, col, feat_name):
    """Compute per-customer mean target-encoded value."""
    tx = df[['customer_id', col]].copy()
    tx['_te'] = tx[col].map(te_map).fillna(global_mean)
    return tx.groupby('customer_id')['_te'].mean().rename(feat_name)


def add_target_encoding(X_train, y_train, X_val, X_test, df_train, df_val, df_test):
    """
    Target-encode MCC and tr_type using 5-fold OOF on train,
    full encoding on val/test.
    """
    train_cust_ids = df_train['customer_id'].drop_duplicates().reset_index(drop=True)
    val_cust_ids   = df_val['customer_id'].drop_duplicates().reset_index(drop=True)
    test_cust_ids  = df_test['customer_id'].drop_duplicates().reset_index(drop=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for col in ['mcc_code', 'tr_type']:
        feat_name = f'{col}_te'
        global_m = y_train.mean()

        # OOF for train
        oof_te = pd.Series(global_m, index=train_cust_ids)
        for tr_idx, vl_idx in skf.split(train_cust_ids, y_train):
            tr_cust = train_cust_ids.iloc[tr_idx]
            vl_cust = train_cust_ids.iloc[vl_idx]
            tr_y    = y_train.iloc[tr_idx]
            te_map, gm = compute_category_te(
                df_train[df_train['customer_id'].isin(tr_cust)], tr_y, col)
            cust_te = apply_te_feature(
                df_train[df_train['customer_id'].isin(vl_cust)], te_map, gm, col, feat_name)
            oof_te.update(cust_te)

        X_train = X_train.copy()
        X_train[feat_name] = oof_te.values

        # Full encoding for val/test
        te_map, gm = compute_category_te(df_train, y_train, col)
        for split_df, split_cids in [(df_val, val_cust_ids), (df_test, test_cust_ids)]:
            cust_te = apply_te_feature(split_df, te_map, gm, col, feat_name)
            # align by customer order
            te_vals = cust_te.reindex(split_cids).fillna(gm).values
            if split_df is df_val:
                X_val = X_val.copy()
                X_val[feat_name] = te_vals
            else:
                X_test = X_test.copy()
                X_test[feat_name] = te_vals

    return X_train, X_val, X_test


# ── Model ─────────────────────────────────────────────────────────────────────

LGB_PARAMS = dict(
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
)


def find_best_threshold(y_true, y_proba):
    """Find threshold maximizing accuracy."""
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.25, 0.75, 0.005):
        acc = accuracy_score(y_true, (y_proba >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def fit_predict_proba(X_train, y_train, X_val, y_val, X_test):
    """
    1. 5-fold CV on train+val → OOF probabilities for threshold search.
    2. Ensemble: 5 fold models predict test → average.
    Best iteration from early-stopping on val.
    """
    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)

    # Best iteration from val early stopping
    probe = lgb.LGBMClassifier(n_estimators=3000, random_state=0, **LGB_PARAMS)
    probe.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
    best_iter = int(probe.best_iteration_ * 1.1)
    print(f"  best iteration (×1.1): {best_iter}")

    # 5-fold CV: OOF on full data + test ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba  = np.zeros(len(X_full))
    test_probas = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_full, y_full)):
        m = lgb.LGBMClassifier(n_estimators=best_iter, random_state=fold, **LGB_PARAMS)
        m.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx],
              callbacks=[lgb.log_evaluation(period=-1)])
        oof_proba[vl_idx] = m.predict_proba(X_full.iloc[vl_idx])[:, 1]
        test_probas.append(m.predict_proba(X_test)[:, 1])
        print(f"  fold {fold+1}/5  OOF AUC: "
              f"{roc_auc_score(y_full.iloc[vl_idx], oof_proba[vl_idx]):.4f}")

    oof_auc = roc_auc_score(y_full, oof_proba)
    print(f"  OOF AUC: {oof_auc:.4f}")

    y_pred_proba = np.mean(test_probas, axis=0)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return y_pred_proba, y_pred


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, y_pred_proba):
    return {
        "auc_score":       round(roc_auc_score(y_true, y_pred_proba), 4),
        "accuracy_score":  round(accuracy_score(y_true, y_pred), 4),
        "precision_score": round(precision_score(y_true, y_pred), 4),
        "recall_score":    round(recall_score(y_true, y_pred), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading splits...")
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    print("Generating features...")
    X_train, y_train, top_mcc, top_tr = generate_features(train_df)
    X_val,   y_val,   _,       _      = generate_features(val_df,  top_mcc=top_mcc, top_tr=top_tr)
    X_test,  y_test,  _,       _      = generate_features(test_df, top_mcc=top_mcc, top_tr=top_tr)

    print("Adding target encoding features...")
    X_train, X_val, X_test = add_target_encoding(
        X_train, y_train, X_val, X_test, train_df, val_df, test_df)
    del train_df, val_df, test_df

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        f"Feature mismatch: {X_train.shape[1]} / {X_val.shape[1]} / {X_test.shape[1]}"

    print(f"Features: {X_train.shape[1]}  |  "
          f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    print("Training (early stop + 5-fold threshold search + final on train+val)...")
    y_pred_proba, y_pred = fit_predict_proba(X_train, y_train, X_val, y_val, X_test)

    metrics = evaluate(y_test, y_pred, y_pred_proba)
    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    auc_ok = metrics["auc_score"] > 0.88
    acc_ok = metrics["accuracy_score"] > 0.80
    print(f"\nROC AUC > 0.88: {'OK' if auc_ok else 'FAIL'}  |  "
          f"Accuracy > 0.80: {'OK' if acc_ok else 'FAIL'}")
