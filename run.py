"""run.py - competition entry point."""

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
    print(f"  wrote {path}  ({len(out):,} rows, median {out[C.PRED_COL].median():,.0f})")


def predict(list_text="none"):
    train = C.load_train()
    V = get_vintage_index(train)
    comps_all = train_comps(train, V)
    off, la = C.load_test_off_market(), C.load_test_list_aware()
    cutoff = C.FINAL_CUTOFF
    p_off, p_list = M.load_params("off_market"), M.load_params("list_aware")
    print(f"final fit: {len(train):,} sales, cutoff {cutoff.date()}, seeds {C.SEEDS}")

    # Off-market
    fr = build_frames(train, off, cutoff, V, comps_all=comps_all)
    models1 = M.stage1_fit(fr, p_off, C.SEEDS)
    lp_off = M.stage1_predict(models1, fr.X_q, fr.addback_q)
    _write(off[C.ID_COL] if C.ID_COL in off else off["ml_num"], np.exp(lp_off), C.SUBMISSION_OFF_MARKET)

    # List-aware
    fl = build_frames(train, la, cutoff, V, comps_all=comps_all)
    lp_la = M.stage1_predict(models1, fl.X_q, fl.addback_q)
    om_all = M.stage1_blocks(train, V, comps_all, p_off, max_start=cutoff - pd.DateOffset(months=C.TEST_MONTHS))
    Xl, yl, text_tr = M.list_training(fl, om_all, list_text)
    enc = M.fit_list_text_encoder(text_tr, list_text)
    if "b" in list_text and enc is not None:
        vec, svd = enc
        z = svd.transform(vec.transform(text_tr))
        Xl.loc[:, M.TEXT_B_COLS] = 0.0
        Xl.loc[:, M.TEXT_B_COLS[:z.shape[1]]] = z
    models2 = M.stage2_fit(Xl, yl, p_list, C.SEEDS)
    list_price = la["list_price"].to_numpy(dtype=float)
    ratio = np.exp(M.stage2_predict(models2, M.list_features(fl.X_q, list_price, lp_la, fl.meta_q["remarks"], list_text, enc)))
    _write(la["ml_num"], list_price * ratio, C.SUBMISSION_LIST_AWARE)


def _determinism() -> bool:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(2000, 8)), columns=list("abcdefgh"))
    y = X.a * 2 + X.b ** 2 + rng.normal(size=2000) * 0.1
    pr = [M.fit_lgb(X, y, dict(M.DEFAULT_OFF, n_rounds=40), 1).predict(X) for _ in range(2)]
    return bool(np.array_equal(pr[0], pr[1]))


def check() -> int:
    bad = []
    def ok(cond, msg):
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        if not cond:
            bad.append(msg)
    print("\n== leakage guards ==")
    for feats, track in ((OFF_MARKET_FEATS, "off_market"), (M.LIST_FEATS, "list_aware")):
        try:
            C.assert_no_leak(feats, track); ok(True, f"{track} features clean ({len(feats)})")
        except AssertionError as e:
            ok(False, str(e))
    ok(not any("list" in c for c in OFF_MARKET_FEATS), "no list-price column in the off-market feature list")
    print("\n== determinism ==")
    ok(_determinism(), "two LightGBM fits with the same seed give identical predictions")
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
    ap.add_argument("--list-text", choices=C.LIST_TEXT_MODES, default="none", help="list-aware text mode")
    a = ap.parse_args(argv)
    M.FAST = a.fast
    if a.command == "validate":
        M.validate(a.folds, do_list=not a.no_list, list_text=a.list_text)
    elif a.command == "tune":
        M.tune(a.stage, a.n_iter)
    elif a.command == "predict":
        predict(a.list_text)
    else:
        sys.exit(check())


if __name__ == "__main__":
    main()