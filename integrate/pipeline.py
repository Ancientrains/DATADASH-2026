"""Configurable integrated model. cfg keys:
    kw: bool             remarks keyword flags
    comps: None|'raw'|'lf'  building/prior-sale/community/KNN comps; raw = re-inflated to last month's level (+lvl),
                            lf = level-free (comp log price minus market index at the comp's sale month, no lvl)
    detrend: bool        off-market only: y = log price - I[seg, sale month]; add back forecast; t_listed dropped
    age: bool            listing age at forecast origin (models agent's definition)
    ff: float            colsample_bytree
    target: 'log'|'ratio'  list-aware only (ratio = log(sold/list))
    stack: None|<stage1 name>  list-aware only: stage-1 out-of-time off-market estimate + gap features
    seeds: tuple         geometric mean over seeds
"""
from common import *

def comps_for(fold, track, te):
    if fold == "REAL":
        F = comps_test_role(None, None, CUTS["REAL"], "test_off_market" if track == "off_market" else "test_list_aware")
    else:
        F = comps_test_role(None, None, CUTS[fold], "blk")
    return F

def block_start(dates, cut, months=4):
    mi = (dates.dt.year - cut.year) * 12 + (dates.dt.month - cut.month)
    k = np.floor(mi / months).astype(int) * months
    return pd.Series([cut + pd.DateOffset(months=int(x)) for x in k], index=dates.index)

_S1 = {}
def stage1_lookup(name, ml):
    if name not in _S1:
        _S1[name] = pd.read_parquet(f"{CACHE}/stage1_{name}.parquet").set_index("ml_num").om_log
    return _S1[name].reindex(np.asarray(ml)).values

def build_X(tr, te, cut, fold, track, cfg, te_comps=None):
    Xtr, Xte = align(base_features(tr), base_features(te))
    if cfg.get("kw"):
        Xtr = Xtr.join(kw_feats(tr)); Xte = Xte.join(kw_feats(te))
    if cfg.get("comps"):
        lf = cfg["comps"] == "lf"
        A = comps_frame(tr.ml_num.values, comps_train_role(), lf)
        B = comps_frame(te.ml_num.values, te_comps if te_comps is not None else comps_for(fold, track, te), lf)
        assert B.index.equals(Xte.index) and A.index.equals(Xtr.index)
        Xtr = pd.concat([Xtr, A], axis=1); Xte = pd.concat([Xte, B], axis=1)
    if cfg.get("age"):
        Xtr["listing_age"] = ((block_start(tr.date_sold, cut) - tr.date_listed).dt.days / 30.44).clip(lower=0).values
        Xte["listing_age"] = ((cut - te.date_listed).dt.days / 30.44).clip(lower=0).values
    if track == "list_aware":
        Xtr["log_list"] = np.log(tr.list_price.values); Xte["log_list"] = np.log(te.list_price.values)
        if cfg.get("stack"):
            Xtr["om_log"] = stage1_lookup(cfg["stack"], tr.ml_num); Xte["om_log"] = stage1_lookup(cfg["stack"], te.ml_num)
            assert np.isfinite(Xte["om_log"]).all()
            Xtr["gap"] = Xtr.log_list - Xtr.om_log; Xte["gap"] = Xte.log_list - Xte.om_log
    if cfg.get("detrend"):
        Xtr = Xtr.drop(columns="t_listed"); Xte = Xte.drop(columns="t_listed")
    return Xtr, Xte

def predict(tr, te, cut, fold, track, cfg, seed=0, te_comps=None):
    """Returns predicted sold_price for te."""
    Xtr, Xte = build_X(tr, te, cut, fold, track, cfg, te_comps)
    y = np.log(tr.sold_price.values); off_te = np.zeros(len(te))
    if track == "off_market" and cfg.get("detrend"):
        itr, add = index_and_addback(tr, te, cut); y = y - itr; off_te = add
    if track == "list_aware" and cfg.get("target") == "ratio":
        y = y - Xtr.log_list.values; off_te = Xte.log_list.values
    params = {"colsample_bytree": cfg["ff"]} if cfg.get("ff") else None
    return np.exp(lgb_fit_predict(Xtr, y, Xte, seed=seed, params=params) + off_te)

def run_cfg(name, track, cfg, folds=FOLDS, verbose=True):
    """Fits each fold x seed (cached). returns {fold: DataFrame(ml_num, y, p, type, list_price)} with seed geo-mean."""
    out = {}
    for f in folds:
        tr, te, full, cut = fold_data(f, track)
        P = []
        for s in cfg.get("seeds", (0,)):
            key = {k: v for k, v in cfg.items() if k != "seeds"}
            tag = "_".join(f"{k}-{v}" for k, v in sorted(key.items()) if v) or "ref"
            fn = f"{PRED}/{track}_{tag}_{f}_s{s}.parquet"
            if os.path.exists(fn):
                p = pd.read_parquet(fn).p.values
            else:
                t = time.time(); p = predict(tr, te, cut, f, track, cfg, seed=s)
                pd.DataFrame({"ml_num": te.ml_num.values, "p": p}).to_parquet(fn)
                if verbose: print(f"   fit {name} {f} s{s} {time.time()-t:.0f}s", flush=True)
            P.append(np.log(p))
        p = np.exp(np.mean(P, axis=0))
        out[f] = pd.DataFrame({"ml_num": te.ml_num.values, "p": p,
                               "y": full.sold_price.values if full is not None else np.nan,
                               "type": te.house_type_name.values,
                               "list_price": te.list_price.values if "list_price" in te else np.nan})
    return out

def summarize(name, res):
    row = {"config": name}
    for f, d in res.items():
        m = metrics(d.y, d.p)
        row[f"{f} MdAPE"] = round(m["MdAPE"], 2); row[f"{f} MAPE"] = round(m["MAPE"], 2); row[f"{f} MdSE"] = round(m["MdSE"], 2)
    return row
