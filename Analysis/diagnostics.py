"""diagnostics.py - functional checks for Part 1 (run:  python diagnostics.py).

Prints only what is needed to trust the inputs: schema/date sanity, remark-flag prevalence and *controlled* price
effect (to compare with the handoff table), the segment index and its forecast against what was realised, and
comparable-sales coverage per fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common as C
from features import build_frames, fold_queries
from trend import (RARE_LEVEL_N, _month_dummies, _solve, addback, get_vintage_index, hedonic_controls,
                   month_id, realised_window_index)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def data_checks(train, off=None, la=None):
    print("== DATA CHECKS ==")
    print(f"train rows kept {len(train):,} (dropped for missing price/dates: {train.attrs.get('dropped_rows', 'n/a')})")
    print(f"date_sold {train.date_sold.min().date()} .. {train.date_sold.max().date()}   "
          f"date_listed {train.date_listed.min().date()} .. {train.date_listed.max().date()}")
    print(f"listed after sold: {(train.date_listed > train.date_sold).sum()} rows")
    print(f"house_area not integer / 'x-y': {train.sqft.isna().mean():.2%} NaN;  ranges {train.sqft_is_range.mean():.1%} "
          f"(width != 199: {int(((train.sqft_width.notna()) & (train.sqft_width != 199)).sum())})")
    print(f"segments: condo {train.seg_id.mean():.1%}; rows without coordinates: {train.coord_key.str.startswith('NOCOORD').mean():.2%}")
    print("missing share:", train[["sqft", "bedrooms_n", "washrooms_n", "parking_n", "garage_n", "lot_front", "latitude"]]
          .isna().mean().round(3).to_dict())
    for name, df in (("test_off_market", off), ("test_list_aware", la)):
        if df is None:
            continue
        seen_coord = df.coord_key.isin(set(train.coord_key)).mean()
        seen_unit = df.unit_key.isin(set(train.unit_key)).mean()
        print(f"{name}: {len(df):,} rows; coordinate seen in training {seen_coord:.1%}; exact unit seen {seen_unit:.1%}")


def flag_report(train):
    """One regression: ln(price) ~ mix controls + month fixed effects + the 22 flags.  Prevalence and effect in %."""
    print("\n== REMARK FLAGS (controlled for mix and month) ==")
    m = month_id(train.date_sold)
    nm = int(m.max()) + 1
    flags = train[C.KW_COLS].to_numpy(dtype=float)
    X = sp.hstack([sp.csr_matrix(np.ones((len(train), 1))), hedonic_controls(train), _month_dummies(m, nm),
                   sp.csr_matrix(flags)], format="csr")
    y = np.log(train.sold_price.to_numpy(dtype=float))
    A = (X.T @ X).toarray()
    A[np.diag_indices_from(A)] += 1e-6 * np.mean(np.diag(A))
    Ainv = np.linalg.inv(A)
    beta = Ainv @ (X.T @ y)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    k = len(C.KW_COLS)
    b, se = beta[-k:], np.sqrt(np.diag(Ainv)[-k:] * s2)
    out = pd.DataFrame({"prevalence_%": flags.mean(0) * 100, "effect_%": (np.exp(b) - 1) * 100, "se_%": se * 100},
                       index=[c[3:] for c in C.KW_COLS]).sort_values("effect_%", ascending=False)
    print(out.round(2).to_string())
    print(f"model R2 on log price: {1 - resid.var() / y.var():.3f}")
    F = pd.DataFrame(flags, columns=[c[3:] for c in C.KW_COLS])
    live = F.columns[F.std() > 0]
    cc = F[live].corr().where(np.triu(np.ones((len(live), len(live))), 1) > 0).stack()
    cc = cc[cc > 0.7].sort_values(ascending=False)
    print("flag pairs that fire together (phi > 0.7):", "none" if cc.empty else "\n" + cc.round(2).to_string())
    print(f"kw_cond_score: mean {train.kw_cond_score.mean():.4f}, share non-zero {(train.kw_cond_score != 0).mean():.1%}")


def index_report(V, train):
    print("\n== SEGMENT INDEX (final vintage; log points vs Jan 2021) and 4-month forecasts ==")
    vl = V.shape[1] - 1
    months = pd.date_range(C.BASE_MONTH, periods=vl, freq="MS")
    show = list(range(0, vl, 6)) + [vl - 1]
    print(pd.DataFrame({"free": V[0, vl, show], "condo": V[1, vl, show]}, index=[months[i].strftime("%Y-%m") for i in show]).round(3).T.to_string())
    rows = []
    for name, cutoff in list(C.FOLDS.items()) + [("FINAL", C.FINAL_CUTOFF)]:
        f, r = addback(V, cutoff), realised_window_index(V, cutoff)
        rows.append({"fold": name, "cutoff": cutoff.date(), "fcst_free": f[0], "real_free": r[0], "err_free_%": 100 * (f[0] - r[0]),
                     "fcst_condo": f[1], "real_condo": r[1], "err_condo_%": 100 * (f[1] - r[1])})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("(err = forecast - realised mean index over the window, in %; NaN for the final window, which is unobserved)")


def comps_coverage(fr, label):
    print(f"\n== COMPS COVERAGE ({label}) ==")
    for name, X in (("train", fr.X_tr), ("query", fr.X_q)):
        cov = {c: float(X[c].notna().mean()) for c in ("rep_lp", "bld_lp", "comm_lp", "rep_lp_raw") if c in X}
        print(f"{name:<6} " + "  ".join(f"{k} {v:.1%}" for k, v in cov.items()))
    print(f"training target (detrended ln price): mean {fr.y_tr.mean():.3f}, sd {fr.y_tr.std():.3f}; "
          f"add-back [free, condo] = {np.round(np.unique(fr.addback_q), 4)}")


def main(fast=False):
    train = C.load_train()
    off = C.load_test_off_market() if C.TEST_OFF_MARKET_CSV.exists() else None
    la = C.load_test_list_aware() if C.TEST_LIST_AWARE_CSV.exists() else None
    data_checks(train, off, la)
    if not fast:
        flag_report(train)
    V = get_vintage_index(train)
    index_report(V, train)
    cutoff = C.FOLDS["S25"]
    fr = build_frames(train, fold_queries(train, cutoff), cutoff, V, "off_market")
    comps_coverage(fr, "S25 off-market")
    print("\nPart 1 diagnostics finished.")


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
