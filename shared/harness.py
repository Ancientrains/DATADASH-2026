"""Shared evaluation harness for idea experiments (SSU Data Dash 2026).

Usage:
    from harness import evaluate, load, base_features
    def fit_predict(train, test, track):   # train: full listings rows sold before cut
        ...                                # test: ONLY columns that exist in the real test file for that track
        return np.array(predictions)       # sold_price predictions, aligned to test rows
    evaluate(fit_predict, track="off_market")  # or "list_aware"

Folds mimic the hidden test: model trained on everything sold before the cut,
the next 4 months of sales in one shot (hidden test = Jun-Sep 2026 from data ending May 2026).
F1: train sold < 2026-02-01, test sold 2026-02-01..2026-05-31   (primary)
F2: train sold < 2025-10-01, test sold 2025-10-01..2026-01-31   (secondary, checks stability)
Test frames have sold_price/date_sold/days_on_market/external_avm_* REMOVED, and list_price removed for off_market.
"""

import re, numpy as np, pandas as pd
import os
DATA = os.environ.get("DATADASH_DATA", "data")  # folder holding listings/test_*.parquet
FOLDS = {"F1": ("2026-02-01", "2026-05-31"), "F2": ("2025-10-01", "2026-01-31")}
OFF_COLS = ['ml_num','date_listed','house_type_name','house_style','bedrooms','washrooms','parking_total','garage',
            'house_area','lot_front','lot_depth','municipality','community','latitude','longitude','remarks']
LA_COLS = OFF_COLS[:1] + ['list_price'] + OFF_COLS[1:]
_cache = {}

def load(name="listings"):
    if name not in _cache:
        d = pd.read_parquet(f"{DATA}/{name}.parquet")
        for c in ("date_listed", "date_sold", "external_avm_estimate_date"):
            if c in d: d[c] = pd.to_datetime(d[c])
        _cache[name] = d
    return _cache[name].copy()

def _num(s):
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    def f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return np.nan
        p = re.findall(r"\d+\.?\d*", str(v)); return float(sum(map(float,p))) if p else np.nan
    return s.map(f)

def _area(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return np.nan
    p = re.findall(r"\d+\.?\d*", str(v).replace(",",""))
    return np.mean([float(x) for x in p]) if p else np.nan

T0 = pd.Timestamp("2021-01-01")
CATS = ["house_type_name","house_style","municipality","community"]
def base_features(df):
    """Simple features computable from test-file columns only."""
    X = pd.DataFrame(index=df.index)
    for c in ["bedrooms","washrooms","parking_total","garage","lot_front","lot_depth","latitude","longitude"]:
        X[c] = _num(df[c])
    X["house_area"] = df["house_area"].map(_area)
    X["area_is_range"] = df["house_area"].astype(str).str.contains("-").astype(int)
    X["t_listed"] = (df["date_listed"] - T0).dt.days / 30.44
    X["list_month"] = df["date_listed"].dt.month
    for c in CATS: X[c] = df[c].astype("category")
    return X

def metrics(y, p):
    ape = np.abs(np.asarray(p,float) - np.asarray(y,float)) / np.asarray(y,float)
    return {"MdAPE": float(np.median(ape)*100), "MAPE": float(np.mean(ape)*100), "n": int(len(y))}

def split(fold, track):
    L = load(); a, b = map(pd.Timestamp, FOLDS[fold])
    tr = L[L.date_sold < a].reset_index(drop=True)
    te_full = L[(L.date_sold >= a) & (L.date_sold <= b)].reset_index(drop=True)
    cols = OFF_COLS if track == "off_market" else LA_COLS
    return tr, te_full[cols].copy(), te_full

def evaluate(fit_predict, track="off_market", folds=("F1","F2"), verbose=True, return_preds=False):
    out, preds = {}, {}
    for f in folds:
        tr, te, full = split(f, track)
        # align categories so train/test categories match
        p = np.asarray(fit_predict(tr, te, track), float)
        assert len(p) == len(te) and np.isfinite(p).all(), "bad predictions"
        out[f] = metrics(full.sold_price.values, p); preds[f] = (full, p)
        if verbose: print(f"[{track}] {f}: MdAPE {out[f]['MdAPE']:.2f}% MAPE {out[f]['MAPE']:.2f}%  n={out[f]['n']}", flush=True)
    return (out, preds) if return_preds else out

# ---------------- reference model: plain LightGBM on log price ----------------
def lgbm_baseline(train, test, track, params=None):
    import lightgbm as lgb
    Xtr, Xte = base_features(train), base_features(test)
    for c in CATS:
        cats = pd.api.types.union_categoricals([Xtr[c], Xte[c]]).categories
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
    if track == "list_aware":
        Xtr["log_list"] = np.log(train.list_price.values); Xte["log_list"] = np.log(test.list_price.values)
    prm = dict(objective="regression", learning_rate=0.05, n_estimators=1500,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=0)
    if params: prm.update(params)
    m = lgb.LGBMRegressor(**prm).fit(Xtr, np.log(train.sold_price.values))
    return np.exp(m.predict(Xte))

if __name__ == "__main__":
    import time; t=time.time()
    evaluate(lambda tr,te,k: te.list_price.values, "list_aware")
    evaluate(lgbm_baseline, "off_market"); print(time.time()-t)
    evaluate(lgbm_baseline, "list_aware"); print(time.time()-t)
