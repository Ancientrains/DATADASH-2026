"""As-of comparable-sales / repeat-sales features.

Every feature for a query row with reference date r uses ONLY history sales with date_sold < r.
- fold test rows: r = fold cut (all training data is legitimately available)
- fold train rows: r = date_listed (mode 'listed') or date_sold - lag (mode 'lag60' / 'lag0')
Prices of comps are time-adjusted with a market index I[group, month] = median log price of sales in months
(m-1, m) of that type-group, then re-inflated to the level of the last complete month before r.
"""

import sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from harness import _area, _num

TG = {"Condo Apt": "CA", "Co-Op Apt": "CA", "Condo Townhouse": "CT", "Detached": "H", "Semi-Detached": "H",
      "Freehold Townhouse": "H"}
M0 = pd.Period("2020-01", "M")

def mon(d):  # integer month index
    return (d.dt.year - 2020) * 12 + d.dt.month - 1

def prep(df):
    d = pd.DataFrame(index=df.index)
    d["key"] = df.latitude.round(6).astype(str) + "_" + df.longitude.round(6).astype(str)
    d["ukey"] = d["key"] + "|" + df.house_type_name.astype(str) + "|" + df.house_style.astype(str) + "|" + \
                df.bedrooms.map(str) + "|" + df.washrooms.map(str) + "|" + df.house_area.map(str) + "|" + df.lot_front.map(str)
    d["tg"] = df.house_type_name.map(TG).fillna("H")
    d["htype"] = df.house_type_name.replace({"Co-Op Apt": "Condo Apt"}).astype(str)
    d["comm"] = df.community.astype(str) + "|" + d["htype"]
    d["area"] = df.house_area.map(_area)
    d["beds"] = _num(df.bedrooms)
    d["lat"] = df.latitude.values; d["lon"] = df.longitude.values
    return d

class Index:
    def __init__(self, hist):
        h = prep(hist); h["m"] = mon(hist.date_sold); h["lp"] = np.log(hist.sold_price.values)
        self.mmax = int(h.m.max())
        self.tab = {}
        for g, s in h.groupby("tg"):
            arr = np.full(self.mmax + 2, np.nan)
            for m in range(int(s.m.min()), self.mmax + 1):
                v = s.lp[(s.m >= m - 1) & (s.m <= m)]
                if len(v) >= 5: arr[m] = np.median(v)
            arr = pd.Series(arr).ffill().bfill().values
            self.tab[g] = arr

    def at(self, tg, m):
        m = np.clip(np.asarray(m, int), 0, self.mmax)
        out = np.empty(len(m))
        for g, arr in self.tab.items():
            k = (tg == g); out[k] = arr[m[k]]
        return out

def build(hist, query, ref, idx, K=10):
    """hist: training sales (with sold_price/date_sold). query: rows (test cols). ref: Timestamps Series aligned to query."""
    h = prep(hist).reset_index(drop=True); q = prep(query).reset_index(drop=True)
    ref = pd.to_datetime(pd.Series(np.asarray(ref))).reset_index(drop=True)
    hd = hist.date_sold.values.astype("datetime64[D]").astype(np.int64)
    rd = ref.values.astype("datetime64[D]").astype(np.int64)
    hm = mon(hist.date_sold.reset_index(drop=True)).values
    # deflated log price of history sales (relative to their own month's index)
    h["dlp"] = np.log(hist.sold_price.values) - idx.at(h.tg.values, hm)
    h["dppsf"] = h["dlp"] - np.log(h["area"])
    h["d"] = hd
    lvl = idx.at(q.tg.values, mon(ref).values - 1)  # level of last complete month before ref
    F = pd.DataFrame(index=q.index)

    def grp_lookup(hkey, qkey, cols, fn, name, maxn=None, window=None, cap=100):
        # vectorised: sort history by (group, date); for each query find slice of strictly-earlier sales
        cats = pd.Index(pd.unique(np.asarray(hkey)))
        hc = cats.get_indexer(hkey); qc = cats.get_indexer(qkey)
        o = np.lexsort((h.d.values, hc)); hcs = hc[o]; hds = h.d.values[o]
        comb_h = hcs.astype(np.int64) * 10**6 + hds
        comb_q = qc.astype(np.int64) * 10**6 + rd
        j = np.searchsorted(comb_h, comb_q, side="left")
        start = np.searchsorted(hcs, qc, side="left")
        lo = start.copy()
        if window is not None: lo = np.maximum(lo, np.searchsorted(comb_h, comb_q - window, side="left"))
        n = maxn if maxn is not None else cap
        lo = np.maximum(lo, j - n)
        valid = (qc >= 0) & (j > lo)
        res = {f"{name}_n": np.where(valid, j - lo, 0).astype(float), f"{name}_nall": np.where(qc >= 0, j - start, 0).astype(float)}
        jj = np.clip(j - 1, 0, len(hds) - 1)
        res[f"{name}_age"] = np.where(valid, (rd - hds[jj]) / 30.44, np.nan)
        offs = np.arange(1, n + 1)
        pos = j[:, None] - offs[None, :]
        okm = (pos >= lo[:, None]) & valid[:, None]
        pos = np.clip(pos, 0, len(hds) - 1)
        import warnings
        for c in cols:
            v = h[c].values[o][pos]; v = np.where(okm, v, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res[f"{name}_{c}"] = v[:, 0] if fn == "last" else np.nanmedian(v, axis=1)
        return res

    # (a) prior sale of the same unit (same coords + type/style/beds/baths/area/lot)
    r = grp_lookup(h.ukey.values, q.ukey.values, ["dlp"], "last", "rep", maxn=1)
    F["rep_lp"] = r["rep_dlp"] + lvl; F["rep_age"] = r["rep_age"]
    F["rep_n_all"] = grp_lookup(h.ukey.values, q.ukey.values, [], None, "repc", cap=1)["repc_nall"]

    # (b) same building / coordinate: last 20 earlier sales
    r = grp_lookup(h.key.values, q.key.values, ["dlp", "dppsf"], "med", "bld", maxn=20)
    F["bld_lp"] = r["bld_dlp"] + lvl; F["bld_ppsf"] = r["bld_dppsf"] + lvl; F["bld_n"] = r["bld_n"]; F["bld_age"] = r["bld_age"]
    F["bld_est"] = F["bld_ppsf"] + np.log(q["area"])

    # same building + same bedrooms
    r = grp_lookup((h.key + "#" + h.beds.astype(str)).values, (q.key + "#" + q.beds.astype(str)).values,
                   ["dlp"], "med", "bb", maxn=10)
    F["bldbed_lp"] = r["bb_dlp"] + lvl

    # (d) community x type, 12-month window, earlier only
    r = grp_lookup(h.comm.values, q.comm.values, ["dlp", "dppsf"], "med", "cm", window=365)
    F["comm_lp"] = r["cm_dlp"] + lvl; F["comm_ppsf"] = r["cm_dppsf"] + lvl; F["comm_n"] = r["cm_n"]

    # (c) k nearest comparable sales: same house type, different coordinate, earlier only
    for c in ["knn_lp", "knn_ppsf", "knn_dist", "knn_est"]: F[c] = np.nan
    coslat = np.cos(np.deg2rad(43.7))
    for t in q.htype.unique():
        hi = np.where(h.htype.values == t)[0]; qi = np.where(q.htype.values == t)[0]
        if len(hi) == 0: continue
        XY = lambda d, ii: np.c_[d.lat.values[ii] * 111.0, d.lon.values[ii] * 111.0 * coslat]
        tree = cKDTree(XY(h, hi))
        kk = min(len(hi), 2000 if t in ("Condo Apt", "Condo Townhouse") else 800)
        for s in range(0, len(qi), 20000):
            qq = qi[s:s + 20000]
            dist, nb = tree.query(XY(q, qq), k=kk)
            nb = hi[nb]
            ok = (h.d.values[nb] < rd[qq, None]) & (dist > 1e-6)
            ok &= np.cumsum(ok, axis=1) <= K
            def med(a):
                a = np.where(ok, a, np.nan)
                with np.errstate(all="ignore"): return np.nanmedian(a, axis=1)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                F.loc[qq, "knn_lp"] = med(h.dlp.values[nb]) + lvl[qq]
                F.loc[qq, "knn_ppsf"] = med(h.dppsf.values[nb]) + lvl[qq]
                F.loc[qq, "knn_dist"] = med(dist)
    F["knn_est"] = F["knn_ppsf"] + np.log(q["area"])
    F["lvl"] = lvl
    return F
