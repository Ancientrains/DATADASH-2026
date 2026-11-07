"""run.py - competition entry point.

    python run.py validate [--folds F1 S25] [--no-list] [--fast]   OOT validation (F1, F2, S25) with baselines
    python run.py tune --stage off_market|list_aware [--n-iter 30] [--fast]   random search on F1/F2/S25
    python run.py predict [--fast]                                  final fit on all sales, writes both submissions
    python run.py check                                             leakage / OOT / determinism / submission checks
"""

import argparse
import sys

import numpy as np
import pandas as pd

import common as C
import models as M
from features import LIST_AWARE_BASE_FEATS, OFF_MARKET_FEATS, build_frames, train_comps
from trend import get_vintage_index


def _write(df_ids: pd.Series, price: np.ndarray, path):
    out = pd.DataFrame({C.ID_COL: df_ids.to_numpy(), C.PRED_COL: np.round(price, 0)})
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out):,} rows, median {out[C.PRED_COL].median():,.0f}, mean {out[C.PRED_COL].mean():,.0f})")


def predict():
    train = C.load_train()
    V = get_vintage_index(train)
    comps_all = train_comps(train, V)
    off, la = C.load_test_off_market(), C.load_test_list_aware()
    cutoff = C.FINAL_CUTOFF
    p_off, p_list = M.load_params("off_market"), M.load_params("list_aware")
    print(f"final fit: {len(train):,} sales, cutoff {cutoff.date()}, seeds {C.SEEDS}")

    # --- off-market track -------------------------------------------------------------------------
    fr = build_frames(train, off, cutoff, V, comps_all=comps_all)
    models1 = M.stage1_fit(fr, p_off, C.SEEDS)
    lp_off = M.stage1_predict(models1, fr.X_q, fr.addback_q)
    _write(off[C.ID_COL] if C.ID_COL in off else off["ml_num"], np.exp(lp_off), C.SUBMISSION_OFF_MARKET)

    # --- list-aware track (same stage-1 models; stage 2 on top) ---------------------------------------
    fl = build_frames(train, la, cutoff, V, comps_all=comps_all)
    lp_la = M.stage1_predict(models1, fl.X_q, fl.addback_q)
    om_all = M.stage1_blocks(train, V, comps_all, p_off, max_start=cutoff - pd.DateOffset(months=C.TEST_MONTHS))
    Xl, yl = M.list_training(fl, om_all)
    print(f"  stage 2 trained on {len(Xl):,} rows with an out-of-time stage-1 valuation")
    models2 = M.stage2_fit(Xl, yl, p_list, C.SEEDS)
    list_price = la["list_price"].to_numpy(dtype=float)
    ratio = np.exp(M.stage2_predict(models2, M.list_features(fl.X_q, list_price, lp_la)))
    _write(la["ml_num"], list_price * ratio, C.SUBMISSION_LIST_AWARE)

    # --- sold/list sanity (sold prices of the test rows are hidden) ---------------------------------------
    t = train.assign(r=train.sold_price / train.list_price)
    summer25 = t[(t.date_sold >= "2025-06-01") & (t.date_sold < "2025-10-01")].r.median()
    spring26 = t[(t.date_sold >= "2026-02-01") & (t.date_sold < "2026-06-01")].r.median()
    print(f"\n  list-aware: median predicted sold/list = {np.median(ratio):.4f}   (training: Jun-Sep 2025 {summer25:.4f}, "
          f"Feb-May 2026 {spring26:.4f}; handoff expects ~0.9512)")
    print(f"  off-market vs list-aware valuation of the same listings is not comparable row-wise (different test files); "
          f"medians: {np.median(np.exp(lp_off)):,.0f} vs {np.median(list_price * ratio):,.0f}")


def _determinism() -> bool:
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(4000, 8)), columns=list("abcdefgh"))
    y = X.a * 2 + X.b ** 2 + rng.normal(size=4000) * 0.1
    pr = [M.fit_lgb(X, y, dict(M.DEFAULT_OFF, n_rounds=60), 1).predict(X) for _ in range(2)]
    return bool(np.array_equal(pr[0], pr[1]))


def check() -> int:
    bad = []
    def ok(cond, msg):
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        if not cond:
            bad.append(msg)
    import lightgbm, sklearn
    print(f"environment: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, lightgbm {lightgbm.__version__}")
    print("\n== leakage guards ==")
    for feats, track in ((OFF_MARKET_FEATS, "off_market"), (M.LIST_FEATS, "list_aware")):
        try:
            C.assert_no_leak(feats, track); ok(True, f"{track} features clean ({len(feats)})")
        except AssertionError as e:
            ok(False, str(e))
    ok(not any("list" in c for c in OFF_MARKET_FEATS), "no list-price column in the off-market feature list")
    train = C.load_train()
    print("\n== out-of-time folds ==")
    for name, c in {**C.FOLDS, "FINAL": C.FINAL_CUTOFF}.items():
        tr = train[train.date_sold < c]
        ok(tr.date_sold.max() < c, f"{name}: every training sale is before {c.date()} (latest {tr.date_sold.max().date()})")
    print("\n== determinism ==")
    ok(_determinism(), "two LightGBM fits with the same seed give identical predictions (deterministic=True)")
    print("\n== submissions ==")
    for path, test_path, track in ((C.SUBMISSION_OFF_MARKET, C.TEST_OFF_MARKET_CSV, "off_market"),
                                   (C.SUBMISSION_LIST_AWARE, C.TEST_LIST_AWARE_CSV, "list_aware")):
        if not path.exists():
            print(f"  (skip) {path} not found - run `python run.py predict` first"); continue
        sub, test = pd.read_csv(path), pd.read_csv(test_path)
        ok(list(sub.columns) == [C.ID_COL, C.PRED_COL], f"{path.name}: columns {list(sub.columns)}")
        ok(len(sub) == len(test), f"{path.name}: {len(sub):,} rows == {len(test):,} test rows")
        ok(sub[C.ID_COL].is_unique and set(sub[C.ID_COL]) == set(test["ml_num"]), f"{path.name}: ids unique and equal to the test ids")
        p = sub[C.PRED_COL].to_numpy(dtype=float)
        ok(bool(np.isfinite(p).all() and (p > 0).all()), f"{path.name}: all predictions finite and positive "
           f"(min {p.min():,.0f}, median {np.median(p):,.0f}, max {p.max():,.0f})")
        if track == "list_aware":
            r = p / test.set_index("ml_num").loc[sub[C.ID_COL], "list_price"].to_numpy(dtype=float)
            ok(0.8 < np.median(r) < 1.05, f"{path.name}: median predicted/list = {np.median(r):.4f} (5%-95%: "
               f"{np.quantile(r, .05):.3f}-{np.quantile(r, .95):.3f})")
    print("\nALL CHECKS PASSED" if not bad else f"\n{len(bad)} CHECK(S) FAILED")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["validate", "tune", "predict", "check"])
    ap.add_argument("--folds", nargs="*", help="subset of F1 F2 S25 (validate)")
    ap.add_argument("--no-list", action="store_true", help="skip the list-aware stack (validate)")
    ap.add_argument("--stage", choices=["off_market", "list_aware"], default="off_market", help="what to tune")
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--fast", action="store_true", help="tiny boosting rounds: smoke test only")
    a = ap.parse_args(argv)
    M.FAST = a.fast
    if a.command == "validate":
        M.validate(a.folds, do_list=not a.no_list)
    elif a.command == "tune":
        M.tune(a.stage, a.n_iter)
    elif a.command == "predict":
        predict()
    else:
        sys.exit(check())


if __name__ == "__main__":
    main()
