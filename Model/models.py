"""models.py - stage-1 off-market model, stage-2 list-aware model, tuning and seed averaging."""
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

NUM_THREADS = 4
FAST = False
BLOCK_SEEDS = (0,)
TUNE_WEIGHTS = (0.5, 0.5)
CHECKPOINTS = (200, 400, 700, 1000, 1500)
LIST_FEATS = LIST_AWARE_BASE_FEATS + ["log_list", "om_log", "gap"]
TEXT_B_COLS = [f"la_tfidf_{i}" for i in range(8)]

DEFAULT_OFF = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=40, feature_fraction=0.65,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, n_rounds=900)
DEFAULT_LIST = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_child_samples=40, feature_fraction=0.65,
                    bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, n_rounds=500)


def get_test_mix() -> dict[str, float] | None:
    for p in (C.TEST_OFF_MARKET_CSV, C.TEST_LIST_AWARE_CSV):
        if p.exists():
            try:
                df = pd.read_csv(p, usecols=["house_type_name"])
                return df["house_type_name"].value_counts(normalize=True).to_dict()
            except Exception:
                pass
    return None


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


def price_metrics(actual, pred, house_types=None, test_mix=None) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    signed = (pred - actual) / actual
    ape = np.abs(signed)
    res = dict(n=len(actual), MdAPE=100 * np.median(ape), MAPE=100 * ape.mean(),
               RMSE=float(np.sqrt(np.mean((pred - actual) ** 2))),
               signed_med=100 * np.median(signed), within10=100 * (ape <= 0.10).mean())
    if house_types is not None and test_mix is not None:
        try:
            types = pd.Series(house_types)
            counts = types.value_counts(normalize=True)
            w = types.map(pd.Series(test_mix) / counts).fillna(0).to_numpy()
            order = np.argsort(ape)
            cw = np.cumsum(w[order]) / np.sum(w)
            res["MdAPE_testmix"] = float(100 * ape[order][np.searchsorted(cw, 0.5)])
        except Exception:
            res["MdAPE_testmix"] = res["MdAPE"]
    return res


def tune_score(m: dict) -> float:
    mdape = m.get("MdAPE_testmix", m["MdAPE"])
    return TUNE_WEIGHTS[0] * mdape + TUNE_WEIGHTS[1] * m["MAPE"]


def _lgb_params(p: dict, seed: int) -> dict:
    q = {k: v for k, v in p.items() if k != "n_rounds"}
    q.update(seed=int(seed), deterministic=True, force_row_wise=True, num_threads=NUM_THREADS, verbosity=-1)
    return q


def fit_lgb(X: pd.DataFrame, y, params: dict, seed: int, rounds: int | None = None) -> lgb.Booster:
    return lgb.train(_lgb_params(params, seed), lgb.Dataset(X, label=np.asarray(y, float)),
                     num_boost_round=rounds or _rounds(params))


def stage1_fit(fr, params, seeds):
    C.assert_no_leak(OFF_MARKET_FEATS, "off_market")
    X = fr.X_tr[OFF_MARKET_FEATS]
    return [fit_lgb(X, fr.y_tr, params, s) for s in seeds]


def stage1_predict(models, X_q: pd.DataFrame, addback_q: np.ndarray) -> np.ndarray:
    X = X_q[OFF_MARKET_FEATS]
    return np.mean([m.predict(X) for m in models], axis=0) + addback_q


def _blocks_key(train, params, seeds) -> str:
    raw = json.dumps([data_key(train), params, list(seeds), OFF_MARKET_FEATS, FAST, NUM_THREADS], sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def block_starts():
    out, k = [], 1
    while True:
        s = C.FINAL_CUTOFF - pd.DateOffset(months=C.TEST_MONTHS * k)
        if month_id(s) < MIN_VINTAGE:
            return sorted(out)
        out.append(s)
        k += 1


def stage1_blocks(train, V, comps_all, params, seeds=BLOCK_SEEDS, max_start=None, verbose=True) -> np.ndarray:
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


def _text_a(remarks) -> pd.DataFrame:
    text = pd.Series(remarks).fillna("").astype(str).str.lower()
    return pd.DataFrame({f"la_text_{k}": text.str.contains(p, regex=True).astype("int8")
                         for k, p in C.LA_TEXT_RULES.items()})


def fit_list_text_encoder(remarks, mode: str):
    if mode not in C.LIST_TEXT_MODES or "b" not in mode:
        return None
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = pd.Series(remarks).fillna("").astype(str)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=4000, sublinear_tf=True)
    try:
        mat = vec.fit_transform(docs)
    except ValueError:
        return None
    n = min(len(TEXT_B_COLS), mat.shape[0] - 1, mat.shape[1] - 1)
    if n < 1:
        return None
    svd = TruncatedSVD(n_components=n, random_state=0)
    svd.fit(mat)
    return vec, svd


def list_features(X_full: pd.DataFrame, list_price, om_log, remarks=None, text_mode="none", text_encoder=None) -> pd.DataFrame:
    X = X_full[LIST_AWARE_BASE_FEATS].copy()
    X["log_list"] = np.log(np.asarray(list_price, float))
    X["om_log"] = np.asarray(om_log, float)
    X["gap"] = X["log_list"] - X["om_log"]
    if "a" in text_mode:
        X = pd.concat([X, _text_a(remarks).set_axis(X.index)], axis=1)
    if "b" in text_mode:
        vals = np.zeros((len(X), len(TEXT_B_COLS)))
        if text_encoder is not None:
            vec, svd = text_encoder
            z = svd.transform(vec.transform(pd.Series(remarks).fillna("").astype(str)))
            vals[:, :z.shape[1]] = z
        X[TEXT_B_COLS] = vals
    return X


def list_training(fr, om_all: np.ndarray, text_mode="none"):
    om = om_all[fr.pos_tr]
    ok = np.where(~np.isnan(om) & (fr.meta_tr["list_price"].to_numpy() > 0))[0]
    remarks = fr.meta_tr["remarks"].iloc[ok].reset_index(drop=True)
    X = list_features(fr.X_tr.iloc[ok], fr.meta_tr["list_price"].to_numpy()[ok], om[ok], remarks, text_mode)
    y = np.log(fr.meta_tr["sold_price"].to_numpy()[ok] / fr.meta_tr["list_price"].to_numpy()[ok])
    return X.reset_index(drop=True), y, remarks


def stage2_fit(X, y, params, seeds):
    C.assert_no_leak(X.columns, "list_aware")
    return [fit_lgb(X, y, params, s) for s in seeds]


def stage2_predict(models, X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def validate(folds=None, seeds=C.SEEDS, do_list=True, quiet=False, list_text="none"):
    train = C.load_train()
    from trend import get_vintage_index
    V = get_vintage_index(train)
    comps_all = train_comps(train, V)
    p_off, p_list = load_params("off_market"), load_params("list_aware")
    test_mix = get_test_mix()
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
        ptype = fr.X_q["house_type_name"].astype(str).to_numpy()

        m_off = price_metrics(actual, np.exp(lp_off), ptype, test_mix)
        rows.append(dict(fold=name, model="off-market (3 seeds)", **m_off))
        print(f"  off-market done ({time.time() - t0:.0f}s)")
        rows.append(dict(fold=name, model="list_price baseline", **price_metrics(actual, list_price, ptype, test_mix)))

        pred_la = None
        if do_list:
            om_all = stage1_blocks(train, V, comps_all, p_off, max_start=cutoff - pd.DateOffset(months=C.TEST_MONTHS))
            Xl, yl, text_tr = list_training(fr, om_all, list_text)
            enc = fit_list_text_encoder(text_tr, list_text)
            if "b" in list_text and enc is not None:
                vec, svd = enc
                z = svd.transform(vec.transform(text_tr))
                Xl.loc[:, TEXT_B_COLS] = 0.0
                Xl.loc[:, TEXT_B_COLS[:z.shape[1]]] = z
            models2 = stage2_fit(Xl, yl, p_list, seeds)
            Xq = list_features(fr.X_q, list_price, lp_off, fr.meta_q["remarks"], list_text, enc)
            pred_la = list_price * np.exp(stage2_predict(models2, Xq))
            m_la = price_metrics(actual, pred_la, ptype, test_mix)
            rows.append(dict(fold=name, model="list-aware (3 seeds)", **m_la))
            print(f"  list-aware trained on {len(Xl):,} rows with stage-1 out-of-time valuation")

        for t in sorted(set(ptype)):
            msk = ptype == t
            by_type.append(dict(fold=name, house_type=t, n=int(msk.sum()),
                                off_MdAPE=price_metrics(actual[msk], np.exp(lp_off)[msk])["MdAPE"],
                                list_MdAPE=price_metrics(actual[msk], pred_la[msk])["MdAPE"] if pred_la is not None else np.nan))

    res = pd.DataFrame(rows)
    C.REPORT_DIR.mkdir(exist_ok=True)
    res.to_csv(C.REPORT_DIR / "validation.csv", index=False)
    pd.DataFrame(by_type).to_csv(C.REPORT_DIR / "validation_by_type.csv", index=False)
    if not quiet:
        print("\n== VALIDATION (MdAPE / MdAPE_testmix / MAPE / RMSE) ==")
        cols_show = ["fold", "model", "n", "MdAPE"] + (["MdAPE_testmix"] if "MdAPE_testmix" in res else []) + ["MAPE", "RMSE"]
        print(res[cols_show].round(2).to_string(index=False))
    return res


def _sample(rng: np.random.Generator, kind: str) -> dict:
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    p = dict(
        objective=str(rng.choice(["regression", "huber"])),
        learning_rate=lu(0.03, 0.09),
        num_leaves=int(lu(24, 96)) if kind == "off_market" else int(lu(16, 48)),
        min_child_samples=int(lu(20, 100)),
        feature_fraction=float(rng.uniform(0.50, 0.75)),  # Prevents comp greediness
        bagging_fraction=float(rng.uniform(0.70, 0.95)),
        bagging_freq=1,
        lambda_l2=lu(0.5, 30.0),
        cat_smooth=lu(10, 80),
        cat_l2=lu(2, 40),
    )
    if p["objective"] == "huber":
        p["alpha"] = float(rng.uniform(0.12, 0.35)) if kind == "off_market" else float(rng.uniform(0.02, 0.08))
    return p


def tune(kind: str, n_iter: int = 30, seed: int = 0):
    from trend import get_vintage_index
    train = C.load_train()
    V = get_vintage_index(train)
    comps_all = train_comps(train, V)
    test_mix = get_test_mix()
    cks = tuple(c for c in CHECKPOINTS if c <= (150 if FAST else 1800)) or (50, 100, 150)
    top = max(cks)
    p_off = load_params("off_market")
    folds = []
    for name, cutoff in C.FOLDS.items():
        fr = build_frames(train, fold_queries(train, cutoff), cutoff, V, verbose=False, comps_all=comps_all)
        actual = fr.meta_q["sold_price"].to_numpy()
        ptypes = fr.X_q["house_type_name"].astype(str).to_numpy()
        item = dict(name=name, fr=fr, actual=actual, types=ptypes)
        if kind == "list_aware":
            lp = stage1_predict(stage1_fit(fr, p_off, (0,)), fr.X_q, fr.addback_q)
            om_all = stage1_blocks(train, V, comps_all, p_off, max_start=cutoff - pd.DateOffset(months=C.TEST_MONTHS), verbose=False)
            Xl, yl, text_tr = list_training(fr, om_all)
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
            X = f["fr"].X_tr[OFF_MARKET_FEATS] if kind == "off_market" else f["Xl"]
            y = f["fr"].y_tr if kind == "off_market" else f["yl"]
            Xq = f["fr"].X_q[OFF_MARKET_FEATS] if kind == "off_market" else f["Xq"]
            b = fit_lgb(X, y, p, 0, rounds=top)
            for k in cks:
                pr = b.predict(Xq, num_iteration=k)
                price = np.exp(pr + f["fr"].addback_q) if kind == "off_market" else f["list"] * np.exp(pr)
                scores[k].append(price_metrics(f["actual"], price, f["types"], test_mix))

        for k in cks:
            ms = scores[k]
            sc = float(np.mean([tune_score(m) for m in ms]))
            log.append(dict(trial=t, rounds=k, score=sc,
                            MdAPE=np.mean([m["MdAPE"] for m in ms]),
                            MdAPE_testmix=np.mean([m.get("MdAPE_testmix", m["MdAPE"]) for m in ms]),
                            MAPE=np.mean([m["MAPE"] for m in ms]), **p))
            if sc < best[0]:
                best = (sc, dict(p, n_rounds=int(k)))

        cur = min(l["score"] for l in log if l["trial"] == t)
        print(f"  trial {t:>3}/{n_iter}: score {cur:.3f} | best {best[0]:.3f} ({time.time() - t0:.0f}s)", flush=True)

    C.REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(C.REPORT_DIR / f"tuning_{kind}.csv", index=False)
    if not FAST:
        save_params(kind, best[1])
    print(f"\nbest {kind} params: {json.dumps(best[1], indent=2)}")
    return best[1]