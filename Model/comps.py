"""comps.py - as-of hierarchical comparable-sales features + spatial k-NN.

For every query row the engine looks only at history rows with date_sold < ref_date (strict).
Levels:
  rep_*     most recent prior sale of the exact unit
  bld_*     median of the last BLD_N sales at the exact coordinate + bedroom-matched building comp
  comm_*    median of the last 12 months of sales of the same house_type_name in the same community
  knn_*     median of the K=10 nearest historical sales of the same house_type_name (cKDTree)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import BASE_MONTH, RATIO_DAYS
from trend import MIN_VINTAGE, month_id

BLD_N = 20
COMM_DAYS = 365
KNN_K = 10
BIG = 100_000
_EPOCH = pd.Timestamp("2000-01-01")

LEVEL_FREE_FEATS = [
    "rep_lp", "rep_age",
    "bld_lp", "bld_ppsf", "bld_est", "bld_n", "bld_age", "bldbed_lp",
    "comm_lp", "comm_ppsf", "comm_n",
    "knn_lp", "knn_ppsf", "knn_dist", "knn_est"
]
RAW_FEATS = [
    "rep_lp_raw",
    "bld_lp_raw", "bld_ppsf_raw", "bld_est_raw", "bldbed_lp_raw",
    "comm_lp_raw", "comm_ppsf_raw",
    "knn_lp_raw", "knn_ppsf_raw", "knn_est_raw"
]
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
        hi = np.searchsorted(self.key, qcode.astype(np.int64) * BIG + qday, side="left")
        if back_days is None:
            lo = np.searchsorted(self.key, qcode.astype(np.int64) * BIG, side="left")
        return lo, hi


def build_comp_features(history: pd.DataFrame, queries: pd.DataFrame, ref_dates, V: np.ndarray) -> pd.DataFrame:
    n = len(queries)
    ref = pd.Series(pd.to_datetime(ref_dates)).reset_index(drop=True) if not isinstance(ref_dates, pd.Timestamp) \
        else pd.Series([ref_dates] * n)
    qday = _days(ref)
    qv = month_id(ref)
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
    # Building + Bedroom key
    hb, qb = codes(h["coord_key"] + "#" + h["bedrooms_n"].map(str),
                   queries["coord_key"] + "#" + queries["bedrooms_n"].map(str))

    L_c, L_u, L_m, L_b = _Level(hc, hday, arrays), _Level(hu, hday, arrays), _Level(hm, hday, arrays), _Level(hb, hday, arrays)
    lo_c, hi_c = L_c.window(qc, qday)
    lo_u, hi_u = L_u.window(qu, qday)
    lo_m, hi_m = L_m.window(qm, qday, COMM_DAYS)
    lo_b, hi_b = L_b.window(qb, qday)

    cols = {k: np.full(n, np.nan) for k in COMP_FEATS}
    deflate_ok = qv >= MIN_VINTAGE

    def defl(L, sl, v):
        idx = V[L.a["seg"][sl], v, np.minimum(L.a["mid"][sl], v - 1)]
        return L.a["lp"][sl] - idx, L.a["ppsf"][sl] - idx

    for i in range(n):
        v = int(qv[i])
        # Unit repeat sale
        if hi_u[i] > lo_u[i]:
            j = hi_u[i] - 1
            cols["rep_age"][i] = (qday[i] - L_u.a["day"][j]) / 30.4375
            cols["rep_lp_raw"][i] = L_u.a["lp"][j]
            if deflate_ok[i]:
                cols["rep_lp"][i] = defl(L_u, slice(j, j + 1), v)[0][0]

        # Building coordinate median
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

        # Building + Bedroom median
        s0_b = max(lo_b[i], hi_b[i] - 10)
        if hi_b[i] > s0_b:
            sl_b = slice(s0_b, hi_b[i])
            cols["bldbed_lp_raw"][i] = _med(L_b.a["lp"][sl_b])
            if deflate_ok[i]:
                cols["bldbed_lp"][i] = _med(defl(L_b, sl_b, v)[0])

        # Community x type (trailing 12m)
        if hi_m[i] > lo_m[i]:
            sl = slice(lo_m[i], hi_m[i])
            cols["comm_n"][i] = hi_m[i] - lo_m[i]
            cols["comm_lp_raw"][i] = _med(L_m.a["lp"][sl])
            cols["comm_ppsf_raw"][i] = _med(L_m.a["ppsf"][sl])
            if deflate_ok[i]:
                a, b = defl(L_m, sl, v)
                cols["comm_lp"][i], cols["comm_ppsf"][i] = _med(a), _med(b)

    # Building sqft-scaled price estimate
    q_sqft = queries["sqft"].to_numpy(dtype=float)
    log_q_sqft = np.where(q_sqft > 0, np.log(q_sqft), np.nan)
    cols["bld_est"] = cols["bld_ppsf"] + log_q_sqft
    cols["bld_est_raw"] = cols["bld_ppsf_raw"] + log_q_sqft

    # -------------------------------------------------------------------------
    # Spatial k-NN via cKDTree
    # -------------------------------------------------------------------------
    coslat = np.cos(np.deg2rad(43.7))
    h_lat, h_lon = h["latitude"].to_numpy(dtype=float), h["longitude"].to_numpy(dtype=float)
    q_lat, q_lon = queries["latitude"].to_numpy(dtype=float), queries["longitude"].to_numpy(dtype=float)
    h_geo_ok = np.isfinite(h_lat) & np.isfinite(h_lon)
    q_geo_ok = np.isfinite(q_lat) & np.isfinite(q_lon)
    q_types = queries["house_type_name"].astype(str).to_numpy()
    h_types = h["house_type_name"].astype(str).to_numpy()

    for t in np.unique(q_types):
        hi = np.where(h_geo_ok & (h_types == t))[0]
        qi = np.where(q_geo_ok & (q_types == t))[0]
        if len(hi) == 0 or len(qi) == 0:
            continue

        XY_h = np.c_[h_lat[hi] * 111.0, h_lon[hi] * 111.0 * coslat]
        XY_q = np.c_[q_lat[qi] * 111.0, q_lon[qi] * 111.0 * coslat]
        tree = cKDTree(XY_h)
        kk = min(len(hi), 1200 if t in ("Condo Apt", "Condo Townhouse") else 500)

        # Batch queries to manage memory
        for s in range(0, len(qi), 15000):
            qq = qi[s:s + 15000]
            dist, nb = tree.query(XY_q[s:s + 15000], k=kk)
            nb_actual = hi[nb]
            # Must be strictly sold before reference day, and distance > 0.01km (10m) to avoid self-match
            ok = (hday[nb_actual] < qday[qq, None]) & (dist > 1e-4)
            ok = ok & (np.cumsum(ok, axis=1) <= KNN_K)

            with np.errstate(all="ignore"):
                cols["knn_dist"][qq] = np.nanmedian(np.where(ok, dist, np.nan), axis=1)
                cols["knn_lp_raw"][qq] = np.nanmedian(np.where(ok, hlp[nb_actual], np.nan), axis=1)
                cols["knn_ppsf_raw"][qq] = np.nanmedian(np.where(ok, hppsf[nb_actual], np.nan), axis=1)

            # Deflate using each query's vintage tensor slice
            qv_batch = qv.iloc[qq].to_numpy() if isinstance(qv, pd.Series) else qv[qq]
            for v_val in np.unique(qv_batch):
                if v_val < MIN_VINTAGE:
                    continue
                m_v = (qv_batch == v_val)
                sub_q = qq[m_v]
                sub_ok = ok[m_v]
                sub_h = nb_actual[m_v]
                v_int = int(v_val)
                c_seg = hseg[sub_h]
                c_mid = hmid[sub_h]
                idx_comp = V[c_seg, v_int, np.minimum(c_mid, v_int - 1)]
                with np.errstate(all="ignore"):
                    cols["knn_lp"][sub_q] = np.nanmedian(np.where(sub_ok, hlp[sub_h] - idx_comp, np.nan), axis=1)
                    cols["knn_ppsf"][sub_q] = np.nanmedian(np.where(sub_ok, hppsf[sub_h] - idx_comp, np.nan), axis=1)

    cols["knn_est"] = cols["knn_ppsf"] + log_q_sqft
    cols["knn_est_raw"] = cols["knn_ppsf_raw"] + log_q_sqft

    return pd.DataFrame(cols, index=queries.index)


def market_ratio(history: pd.DataFrame, ref_dates, seg_ids, days: int = RATIO_DAYS) -> np.ndarray:
    """As-of market level of the sold/list ratio: median ln(sold/list) over [ref - days, ref)."""
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