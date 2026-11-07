"""comps.py - as-of hierarchical comparable-sales features.

For every query row the engine looks only at history rows with date_sold < ref_date (strict).
  training rows : ref_date = the row's own date_listed          (vintage = month of date_listed)
  held-out rows : ref_date = the fold / competition cutoff       (vintage = month of the cutoff)

Levels
  rep_*   most recent prior sale of the exact unit (coord_key + type, style, bedrooms, washrooms, house_area)
  bld_*   median of the last BLD_N sales at the exact coordinate
  comm_*  median of the last 12 months of sales of the same house_type_name in the same community

Two variants are produced in one pass
  level-free (no suffix): comp price deflated by the segment index of the comp's sale month, using only the
        vintage known at ref_date  (comps sold inside the reference month use the last completed month, v-1)
  raw (suffix _raw): undeflated (used by the list-aware model, whose list price already anchors the level)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import BASE_MONTH, RATIO_DAYS
from trend import MIN_VINTAGE, month_id

BLD_N = 20
COMM_DAYS = 365
BIG = 100_000                     # day numbers are < 100_000, so code * BIG + day orders by (group, day)
_EPOCH = pd.Timestamp("2000-01-01")

LEVEL_FREE_FEATS = ["rep_lp", "rep_age", "bld_lp", "bld_ppsf", "bld_n", "bld_age", "comm_lp", "comm_ppsf", "comm_n"]
RAW_FEATS = ["rep_lp_raw", "bld_lp_raw", "bld_ppsf_raw", "comm_lp_raw", "comm_ppsf_raw"]
COMP_FEATS = LEVEL_FREE_FEATS + RAW_FEATS


def _days(ts) -> np.ndarray:
    return (pd.DatetimeIndex(ts) - _EPOCH).days.to_numpy().astype(np.int64)


def _med(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return float(np.median(a)) if a.size else np.nan


class _Level:
    """History sorted by (group code, day) with a combined searchable key."""

    def __init__(self, codes: np.ndarray, day: np.ndarray, arrays: dict):
        order = np.lexsort((day, codes))
        self.key = codes[order].astype(np.int64) * BIG + day[order]
        self.day = day[order]
        self.a = {k: v[order] for k, v in arrays.items()}

    def window(self, qcode: np.ndarray, qday: np.ndarray, back_days: int | None = None):
        start = qcode.astype(np.int64) * BIG + (0 if back_days is None else (qday - back_days))
        lo = np.searchsorted(self.key, start, side="left")
        hi = np.searchsorted(self.key, qcode.astype(np.int64) * BIG + qday, side="left")   # strict: date_sold < ref
        if back_days is None:                                                              # whole group history
            lo = np.searchsorted(self.key, qcode.astype(np.int64) * BIG, side="left")
        return lo, hi


def build_comp_features(history: pd.DataFrame, queries: pd.DataFrame, ref_dates, V: np.ndarray) -> pd.DataFrame:
    """history: cleaned sold rows.  queries: cleaned rows to featurise.  ref_dates: Timestamp or per-row Series/array."""
    n = len(queries)
    ref = pd.Series(pd.to_datetime(ref_dates)).reset_index(drop=True) if not isinstance(ref_dates, pd.Timestamp) \
        else pd.Series([ref_dates] * n)
    qday = _days(ref)
    qv = month_id(ref)                                   # vintage of each query
    h = history.reset_index(drop=True)
    hday = _days(h["date_sold"])
    hlp = np.log(h["sold_price"].to_numpy(dtype=float))
    sq = h["sqft"].to_numpy(dtype=float)
    hppsf = np.where(sq > 0, np.log(h["sold_price"].to_numpy(dtype=float) / np.where(sq > 0, sq, 1.0)), np.nan)
    hseg = h["seg_id"].to_numpy()
    hmid = month_id(h["date_sold"])
    arrays = {"lp": hlp, "ppsf": hppsf, "seg": hseg, "mid": hmid, "day": hday}

    def codes(hcol, qcol):
        both = pd.concat([hcol.reset_index(drop=True), qcol.reset_index(drop=True)], ignore_index=True)
        c, _ = pd.factorize(both)
        return c[:len(hcol)], c[len(hcol):]

    hc, qc = codes(h["coord_key"], queries["coord_key"])
    hu, qu = codes(h["unit_key"], queries["unit_key"])
    hm, qm = codes(h["house_type_name"].astype(str) + "|" + h["community"].astype(str),
                   queries["house_type_name"].astype(str) + "|" + queries["community"].astype(str))
    L_c, L_u, L_m = _Level(hc, hday, arrays), _Level(hu, hday, arrays), _Level(hm, hday, arrays)
    lo_c, hi_c = L_c.window(qc, qday)
    lo_u, hi_u = L_u.window(qu, qday)
    lo_m, hi_m = L_m.window(qm, qday, COMM_DAYS)

    cols = {k: np.full(n, np.nan) for k in COMP_FEATS}
    deflate_ok = qv >= MIN_VINTAGE

    def defl(L, sl, v):
        idx = V[L.a["seg"][sl], v, np.minimum(L.a["mid"][sl], v - 1)]
        return L.a["lp"][sl] - idx, L.a["ppsf"][sl] - idx

    for i in range(n):
        v = int(qv[i])
        # unit repeat sale: most recent prior sale
        if hi_u[i] > lo_u[i]:
            j = hi_u[i] - 1
            cols["rep_age"][i] = (qday[i] - L_u.a["day"][j]) / 30.4375
            cols["rep_lp_raw"][i] = L_u.a["lp"][j]
            if deflate_ok[i]:
                cols["rep_lp"][i] = defl(L_u, slice(j, j + 1), v)[0][0]
        # building: last BLD_N sales at the exact coordinate
        s0 = max(lo_c[i], hi_c[i] - BLD_N)
        if hi_c[i] > s0:
            sl = slice(s0, hi_c[i])
            cols["bld_n"][i] = hi_c[i] - s0
            cols["bld_age"][i] = (qday[i] - L_c.a["day"][hi_c[i] - 1]) / 30.4375
            cols["bld_lp_raw"][i] = _med(L_c.a["lp"][sl])
            cols["bld_ppsf_raw"][i] = _med(L_c.a["ppsf"][sl])
            if deflate_ok[i]:
                a, b = defl(L_c, sl, v)
                cols["bld_lp"][i], cols["bld_ppsf"][i] = _med(a), _med(b)
        # community x type, trailing 12 months
        if hi_m[i] > lo_m[i]:
            sl = slice(lo_m[i], hi_m[i])
            cols["comm_n"][i] = hi_m[i] - lo_m[i]
            cols["comm_lp_raw"][i] = _med(L_m.a["lp"][sl])
            cols["comm_ppsf_raw"][i] = _med(L_m.a["ppsf"][sl])
            if deflate_ok[i]:
                a, b = defl(L_m, sl, v)
                cols["comm_lp"][i], cols["comm_ppsf"][i] = _med(a), _med(b)
    return pd.DataFrame(cols, index=queries.index)


def market_ratio(history: pd.DataFrame, ref_dates, seg_ids, days: int = RATIO_DAYS) -> np.ndarray:
    """As-of market level of the sold/list ratio: median ln(sold/list) of the same segment over [ref - days, ref)
    (strictly before the reference date).  One value per query; NaN if the window is empty."""
    seg_ids = np.asarray(seg_ids)
    n = len(seg_ids)
    ref = pd.Series(pd.to_datetime(ref_dates)).reset_index(drop=True) if not isinstance(ref_dates, pd.Timestamp) \
        else pd.Series([ref_dates] * n)
    qday = _days(ref)
    lp = history["list_price"].to_numpy(dtype=float)
    lr = np.where(lp > 0, np.log(history["sold_price"].to_numpy(dtype=float) / np.where(lp > 0, lp, 1.0)), np.nan)
    hday, hseg = _days(history["date_sold"]), history["seg_id"].to_numpy().astype(np.int64)
    order = np.lexsort((hday, hseg))
    key, lr = hseg[order] * BIG + hday[order], lr[order]
    pairs, inv = np.unique(np.c_[seg_ids.astype(np.int64), qday], axis=0, return_inverse=True)
    out = np.full(len(pairs), np.nan)
    for i, (sg, d) in enumerate(pairs):
        lo = np.searchsorted(key, sg * BIG + d - days, side="left")
        hi = np.searchsorted(key, sg * BIG + d, side="left")
        if hi > lo:
            out[i] = _med(lr[lo:hi])
    return out[inv.ravel()]