"""models.py - stage-1 off-market model, stage-2 list-aware model, tuning and seed averaging (LightGBM).

Stage 1 (off-market): target ln(price) - index(sale month); features OFF_MARKET_FEATS (no list price);
    prediction = exp(model + forecast-index add-back).
Stage 2 (list-aware): target ln(sold / list); features = raw comps + text + physical + log_list, om_log, gap, where
    om_log is the stage-1 valuation produced out-of-time (4-month blocks, each fitted only on earlier sales).
All models: objective/hyper-parameters from params/*.json (see `python run.py tune`), seeds (0, 1, 2) averaged in log
space, deterministic=True.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import common as C
from features import (LIST_AWARE_BASE_FEATS, OFF_MARKET_FEATS, build_frames, fold_queries, train_comps,
                      window_positions, data_key)
from trend import MIN_VINTAGE, month_id

NUM_THREADS = 4                    # fixed (not "all cores") so results do not depend on the machine
FAST = False                       # set by `run.py --fast`: tiny rounds for a smoke test
BLOCK_SEEDS = (0,)                 # seeds for the out-of-time stage-1 blocks that feed the stack (cost control)
TUNE_WEIGHTS = (0.5, 0.5)          # tuning score = w1 * MdAPE + w2 * MAPE (mean over folds)
CHECKPOINTS = (200, 400, 700, 1000, 1500, 2000)
LIST_FEATS = LIST_AWARE_BASE_FEATS + ["log_list", "om_log", "gap"]

DEFAULT_OFF = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=40, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, n_rounds=800)
DEFAULT_LIST = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_child_samples=40, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, n_rounds=500)


# ------------------------------------------------------------------------------------------------
# parameters / metrics
# ------------------------------------------------------------------------------------------------
def load_params(kind: str) -> dict:
    p = C.PARAMS_DIR / f"{kind}.json"
    base = dict(DEFAULT_OFF if kind == "off_market" else DEFAULT_LIST)
    if p.exists():
        base.update(json.loads(p.read_text()))
    return base


def save_params(kind: str, params: dict):
    C.PARAMS_DIR.mkdir(exist_ok=True)
    (C.PARAMS_DIR / f"{kind}.json").write_text(json.dumps(params, indent=2, sort_keys=True))


def _rounds(params) -> int:
    return min(int(params["n_rounds"]), 120) if FAST else int(params["n_rounds"])


def price_metrics(actual, pred) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    signed = (pred - actual) / actual
    ape = np.abs(signed)
    return dict(n=len(actual), MdAPE=100 * np.median(ape), MAPE=100 * ape.mean(), RMSE=float(np.sqrt(np.mean((pred - actual) ** 2))),
                signed_med=100 * np.median(signed), within10=100 * (ape <= 0.10).mean())


def tune_score(m: dict) -> float:
    return TUNE_WEIGHTS[0] * m["MdAPE"] + TUNE_WEIGHTS[1] * m["MAPE"]


# ------------------------------------------------------------------------------------------------
# LightGBM wrappers
# ------------------------------------------------------------------------------------------------
def _lgb_params(p: dict, seed: int) -> dict:
    q = {k: v for k, v in p.items() if k != "n_rounds"}
    q.update(seed=int(seed), deterministic=True, force_row_wise=True, num_threads=NUM_THREADS, verbosity=-1)
    return q


def fit_lgb(X: pd.DataFrame, y, params: dict, seed: int, rounds: int | None = None) -> lgb.Booster:
    return lgb.train(_lgb_params(params, seed), lgb.Dataset(X, label=np.asarray(y, float)),
                     num_boost_round=rounds or _rounds(params))


# ------------------------------------------------------------------------------------------------
# stage 1
# ------------------------------------------------------------------------------------------------
def stage1_fit(fr, params, seeds):
    C.assert_no_leak(OFF_MARKET_FEATS, "off_market")
    X = fr.X_tr[OFF_MARKET_FEATS]
    return [fit_lgb(X, fr.y_tr, params, s) for s in seeds]


def stage1_predict(models, X_q: pd.DataFrame, addback_q: np.ndarray) -> np.ndarray:
    """ln(price) prediction = mean over seeds of the detrended-premium model + add-back."""
    X = X_q[OFF_MARKET_FEATS]
    return np.mean([m.predict(X) for m in models], axis=0) + addback_q


def _blocks_key(train, params, seeds) -> str:
    raw = json.dumps([data_key(train), params, list(seeds), OFF_MARKET_FEATS, FAST, NUM_THREADS], sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def block_starts():
    """4-month block starts going back from the final cutoff while a vintage with >= MIN_VINTAGE months exists."""
    out, k = [], 1
    while True:
        s = C.FINAL_CUTOFF - pd.DateOffset(months=C.TEST_MONTHS * k)
        if month_id(s) < MIN_VINTAGE:
            return sorted(out)
        out.append(s)
        k += 1


def stage1_blocks(train, V, comps_all, params, seeds=BLOCK_SEEDS, max_start=None, verbose=True) -> np.ndarray:
    """Out-of-time stage-1 valuations (ln price) for rows sold in blocks [s, s+4m): model fitted on sales < s only,
    features referenced to s.  Returns an array aligned with `train` (NaN where no block covers the row)."""
    om = np.full(len(train), np.nan)
    key = _blocks_key(train, params, seeds)
    for s in block_starts():
        if max_start is not None and s > max_start:
            continue
        pos = window_positions(train, s)
        path = C.CACHE_DIR / f"block_{key}_{s:%Y%m}.npy"
        if path.exists():
            om[pos] = np.load(path)
            continue
        t0 = time.time()
        fr = build_frames(train, train.iloc[pos], s, V, verbose=False, comps_all=comps_all)
        vals = stage1_predict(stage1_fit(fr, params, seeds), fr.X_q, fr.addback_q)
        C.CACHE_DIR.mkdir(exist_ok=True)
        np.save(path, vals)
        om[pos] = vals
        if verbose:
            print(f"    stage-1 block {s:%Y-%m}: {len(pos):,} rows ({time.time() - t0:.0f}s)")
    return om


# ------------------------------------------------------------------------------------------------
# stage 2
# ------------------------------------------------------------------------------------------------
def list_features(X_full: pd.DataFrame, list_price, om_log) -> pd.DataFrame:
    X = X_full[LIST_AWARE_BASE_FEATS].copy()
    X["log_list"] = np.log(np.asarray(list_price, float))
    X["om_log"] = np.asarray(om_log, float)
    X["gap"] = X["log_list"] - X["om_log"]
    return X[LIST_FEATS]


def list_training(fr, om_all: np.ndarray):
    """Stage-2 training set: training rows that have an out-of-time stage-1 valuation."""
    om = om_all[fr.pos_tr]
    ok = np.where(~np.isnan(om) & (fr.meta_tr["list_price"].to_numpy() > 0))[0]
    X = list_features(fr.X_tr.iloc[ok], fr.meta_tr["list_price"].to_numpy()[ok], om[ok])
    y = np.log(fr.meta_tr["sold_price"].to_numpy()[ok] / fr.meta_tr["list_price"].to_numpy()[ok])
    return X.reset_index(drop=True), y


def stage2_fit(X, y, params, seeds):
    C.assert_no_leak(LIST_FEATS, "list_aware")
    return [fit_lgb(X, y, params, s) for s in seeds]


def stage2_predict(models, X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)        # predicted ln(sold / list)


# ------------------------------------------------------------------------------------------------
# validation (horizon-matched single-fit folds)
# ------------------------------------------------------------------------------------------------
def validate(folds=None, seeds=C.SEEDS, do_list=True, quiet=False):
    train = C.load_train()
    from trend import get_vintage_index
    V = get_vintage_index(train)
    comps_all = train_comps(train, V)
    p_off, p_list = load_params("off_market"), load_params("list_aware")
    rows, by_type = [], []
    for name, cutoff in C.FOLDS.items():
        if folds and name not in folds:
            continue
        print(f"\n[{name}] cutoff {cutoff.date()}")
        q = fold_queries(train, cutoff)
        fr = build_frames(train, q, cutoff, V, comps_all=comps_all)
        t0 = time.time()
        lp_off = stage1_predict(stage1_fit(fr, p_off, seeds), fr.X_q, fr.addback_q)
        actual, list_price = fr.meta_q["sold_price"].to_numpy(), fr.meta_q["list_price"].to_numpy()
        m_off = price_metrics(actual, np.exp(lp_off))
        rows.append(dict(fold=name, model="off-market (3 seeds)", **m_off))
        print(f"  off-market done ({time.time() - t0:.0f}s)")
        rows.append(dict(fold=name, model="list_price baseline", **price_metrics(actual, list_price)))
        mk = fr.X_q["mkt_lr"].to_numpy()
        rows.append(dict(fold=name, model=f"list x trailing {C.RATIO_DAYS}d sold/list", **price_metrics(actual, list_price * np.exp(np.nan_to_num(mk)))))
        ptype = fr.X_q["house_type_name"].astype(str).to_numpy()
        pred_la = None
        if do_list:
            om_all = stage1_blocks(train, V, comps_all, p_off, max_start=cutoff - pd.DateOffset(months=C.TEST_MONTHS))
            Xl, yl = list_training(fr, om_all)
            models2 = stage2_fit(Xl, yl, p_list, seeds)
            Xq = list_features(fr.X_q, list_price, lp_off)
            pred_la = list_price * np.exp(stage2_predict(models2, Xq))
            rows.append(dict(fold=name, model="list-aware (3 seeds)", **price_metrics(actual, pred_la)))
            print(f"  list-aware trained on {len(Xl):,} rows with an out-of-time stage-1 valuation")
        for t in sorted(set(ptype)):
            msk = ptype == t
            by_type.append(dict(fold=name, house_type=t, n=int(msk.sum()),
                                off_MdAPE=price_metrics(actual[msk], np.exp(lp_off)[msk])["MdAPE"],
                                list_MdAPE=price_metrics(actual[msk], pred_la[msk])["MdAPE"] if pred_la is not None else np.nan))
    res = pd.DataFrame(rows)
    ref = []
    for _, r in res.iterrows():
        key = "off_market" if r["model"].startswith("off") else "list_aware" if r["model"].startswith("list-aware") else None
        h = C.HANDOFF_REF.get(key, {}).get(r["fold"]) if key else None
        ref.append(f"{h[0]:.2f} / {h[1]:.2f}" if h else "")
    res["handoff MdAPE/MAPE"] = ref
    C.REPORT_DIR.mkdir(exist_ok=True)
    res.to_csv(C.REPORT_DIR / "validation.csv", index=False)
    pd.DataFrame(by_type).to_csv(C.REPORT_DIR / "validation_by_type.csv", index=False)
    if not quiet:
        print("\n== VALIDATION (MdAPE / MAPE / signed median in %, RMSE in $) ==")
        print(res.round(2).to_string(index=False))
        bt = pd.DataFrame(by_type)
        last = bt[bt.fold == bt.fold.iloc[-1]] if len(bt) else bt
        print(f"\nMdAPE by house type, fold {last.fold.iloc[0] if len(last) else ''}:")
        print(last.round(2).to_string(index=False))
    return res


# ------------------------------------------------------------------------------------------------
# tuning
# ------------------------------------------------------------------------------------------------
def _sample(rng: np.random.Generator, kind: str) -> dict:
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    p = dict(objective=str(rng.choice(["regression", "huber", "regression_l1"])), learning_rate=lu(0.03, 0.12),
             num_leaves=int(lu(15, 255)), min_child_samples=int(lu(10, 200)), feature_fraction=float(rng.uniform(0.5, 1.0)),
             bagging_fraction=float(rng.uniform(0.6, 1.0)), bagging_freq=1, lambda_l2=lu(0.01, 50),
             cat_smooth=lu(5, 100), cat_l2=lu(1, 50))
    if p["objective"] == "huber":
        p["alpha"] = float(rng.uniform(0.15, 0.8)) if kind == "off_market" else float(rng.uniform(0.02, 0.12))
    return p


def tune(kind: str, n_iter: int = 30, seed: int = 0):
    """Random search; every trial is fitted once per fold (seed 0) and scored at several boosting-round checkpoints."""
    from trend import get_vintage_index
    train = C.load_train()
    V = get_vintage_index(train)
    comps_all = train_comps(train, V)
    cks = tuple(c for c in CHECKPOINTS if c <= (150 if FAST else 2000))
    cks = cks or (50, 100, 150)
    top = max(cks)
    p_off = load_params("off_market")
    folds = []
    for name, cutoff in C.FOLDS.items():
        fr = build_frames(train, fold_queries(train, cutoff), cutoff, V, verbose=False, comps_all=comps_all)
        actual = fr.meta_q["sold_price"].to_numpy()
        item = dict(name=name, fr=fr, actual=actual)
        if kind == "list_aware":
            lp = stage1_predict(stage1_fit(fr, p_off, (0,)), fr.X_q, fr.addback_q)
            om_all = stage1_blocks(train, V, comps_all, p_off, max_start=cutoff - pd.DateOffset(months=C.TEST_MONTHS), verbose=False)
            Xl, yl = list_training(fr, om_all)
            lst = fr.meta_q["list_price"].to_numpy()
            item.update(Xl=Xl, yl=yl, Xq=list_features(fr.X_q, lst, lp), list=lst)
        folds.append(item)
    rng = np.random.default_rng(seed)
    base = load_params(kind)
    log, best = [], (np.inf, None)
    for t in range(n_iter):
        p = {k: v for k, v in base.items() if k != "n_rounds"} if t == 0 else _sample(rng, kind)
        scores = {k: [] for k in cks}
        t0 = time.time()
        for f in folds:
            if kind == "off_market":
                X, y, Xq = f["fr"].X_tr[OFF_MARKET_FEATS], f["fr"].y_tr, f["fr"].X_q[OFF_MARKET_FEATS]
            else:
                X, y, Xq = f["Xl"], f["yl"], f["Xq"]
            b = fit_lgb(X, y, p, 0, rounds=top)
            for k in cks:
                pr = b.predict(Xq, num_iteration=k)
                price = np.exp(pr + f["fr"].addback_q) if kind == "off_market" else f["list"] * np.exp(pr)
                scores[k].append(price_metrics(f["actual"], price))
        for k in cks:
            ms = scores[k]
            sc = float(np.mean([tune_score(m) for m in ms]))
            log.append(dict(trial=t, rounds=k, score=sc, MdAPE=np.mean([m["MdAPE"] for m in ms]), MAPE=np.mean([m["MAPE"] for m in ms]),
                            **{f"{f['name']}_MdAPE": m["MdAPE"] for f, m in zip(folds, ms)}, **p))
            if sc < best[0]:
                best = (sc, dict(p, n_rounds=int(k)))
        cur = min(l["score"] for l in log if l["trial"] == t)
        print(f"  trial {t:>3}/{n_iter}: best-of-checkpoints score {cur:.3f}   overall best {best[0]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    C.REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(C.REPORT_DIR / f"tuning_{kind}.csv", index=False)
    if not FAST:
        save_params(kind, best[1])
    print(f"\nbest {kind} params (score {best[0]:.3f} = {TUNE_WEIGHTS[0]}*MdAPE + {TUNE_WEIGHTS[1]}*MAPE):\n{json.dumps(best[1], indent=2)}")
    return best[1]