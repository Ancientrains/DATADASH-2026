"""Integrated pipeline (off-market + list-aware) — shared pieces.

Folds = 4-month blocks of SALE date (train = all sales sold before the block start):
S25: 2025-06..09 (same season as hidden test), F2: 2025-10..2026-01, F1: 2026-02..05, REAL: hidden test (cut 2026-06-01).
Feature timing rules
  * kw (remarks keyword flags): row-local, no other sales.
  * comps (building / prior sale / community / KNN): for training rows the reference is
    date_listed (only sales sold strictly before it); for fold-test rows the reference date is the fold cut.
    Because a training row's comps only use sales before its date_listed (< its sale < cut) and the deflation index
    at month m uses only sales of months m-1..m, the training-row features do not depend on the cut -> built once.
  * trend index (detrended target): built per fold from the fold's training sales only; test add-back is a
    forecast of the index for the 4 test months (no test-period information).
"""

import sys, os, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import harness
from harness import load, base_features, CATS
from comps import build as comps_build, Index as CompsIndex
from trend import build_index, forecast, seg, mperiod

WORK = os.environ.get("INTEGRATE_WORK", HERE)  # set to an empty dir to rebuild everything from scratch
CACHE = os.path.join(WORK, "cache"); PRED = os.path.join(WORK, "preds")
os.makedirs(CACHE, exist_ok=True); os.makedirs(PRED, exist_ok=True)
CUTS = {"S25": "2025-06-01", "F2": "2025-10-01", "F1": "2026-02-01", "REAL": "2026-06-01"}
FOLDS = ("F1", "F2", "S25")
OFF_COLS, LA_COLS = harness.OFF_COLS, harness.LA_COLS

# ---------------- keyword flags (from ideas_text/text_feats.py, unchanged) ----------------
KW = {
    "asis": r"as[- ]is\b",
    "needs_work": r"needs? (?:work|replacing|paint|tlc|updat|new|a full|some)|cosmetic overhaul|renovator|contractor|handyman|fixer|worn|bring your tools|tired",
    "original": r"original|older|bygone|dated|nostalgic|vintage",
    "potential": r"potential|great bones|blank (?:slate|canvas)|personal touch|your vision|make it your own",
    "clean": r"\bclean\b|well[- ]kept|well[- ]maintained|cared for|good (?:shape|condition)|sound and liv",
    "updated": r"updated|refreshed|newer|new countertops|freshly painted|refinished|renewed",
    "renovated": r"renovat|redone|rebuilt|gut",
    "luxury": r"luxur|designer|custom|high-end|spa-like|heated (?:bathroom )?floors|frameless|quartz|chef|built-in appliances|smooth ceilings|premium|upscale|exquisite|wide-plank|pot lights",
    "stunning": r"stunning|exceptional|expansive|truly special|rare (?:find|combination)",
    "turnkey": r"turnkey|move-in ready|nothing (?:but|more)",
    "motivated": r"motivated|best offer|bring (?:your )?offers|quick (?:or flexible )?closing|flexible closing",
    "act_fast": r"won.t last|act quickly|don.t miss|don.t wait",
    "investor": r"investor|investment|income|rental",
    "view": r"unobstructed view|\bviews?\b(?!ing)|skyline|treetops",
    "f2c": r"floor-to-ceiling",
    "amen": r"concierge|gym|rooftop|party room|guest suite",
    "ravine": r"ravine|trails",
    "south": r"south-facing",
    "cozy": r"cozy|compact|efficient|starter|first[- ]time",
    "spacious": r"spacious|room to grow|room for everyone|space for everyone",
    "locker": r"locker",
    "deep_lot": r"deep \d|deep lot|mature trees",
}

def kw_feats(df):
    lr = df.remarks.fillna("").str.lower()
    X = pd.DataFrame({f"kw_{k}": lr.str.contains(p).astype(int) for k, p in KW.items()}, index=df.index)
    X["kw_cond_score"] = (X.kw_luxury + X.kw_renovated + X.kw_updated + X.kw_turnkey
                          - X.kw_asis - X.kw_needs_work - X.kw_original)
    return X

# ---------------- comps ----------------
COMP_LEVEL = ["rep_lp", "bld_lp", "bld_ppsf", "bld_est", "bldbed_lp", "comm_lp", "comm_ppsf", "knn_lp", "knn_ppsf", "knn_est"]
COMP_OTHER = ["rep_age", "rep_n_all", "bld_n", "bld_age", "comm_n", "knn_dist"]

def comps_train_role():
    """Leak-safe training-role comps using only months complete before listing.

    Rounding the reference down to the listing month's start prevents a
    same-month index median from incorporating sales that occur later than a
    row's actual listing date (planner item E1).
    """
    fn = f"{CACHE}/comps_train_role.parquet"
    if os.path.exists(fn): return pd.read_parquet(fn)
    L = load(); t = time.time()
    ref = L.date_listed.dt.to_period("M").dt.to_timestamp().reset_index(drop=True)
    F = comps_build(L, L, ref, CompsIndex(L))
    F.index = L.ml_num.values; F.to_parquet(fn); print(f"comps train-role built {time.time()-t:.0f}s", flush=True)
    return F

def comps_test_role(hist, query, cut, tag):
    """Comps for rows predicted from origin `cut`: history = sales sold < cut, reference date = cut."""
    fn = f"{CACHE}/comps_test_{tag}_{pd.Timestamp(cut).date()}.parquet"
    if os.path.exists(fn): return pd.read_parquet(fn)
    assert (hist.date_sold < pd.Timestamp(cut)).all()
    F = comps_build(hist.reset_index(drop=True), query.reset_index(drop=True),
                    pd.Series([pd.Timestamp(cut)] * len(query)), CompsIndex(hist))
    F.index = query.ml_num.values; F.to_parquet(fn); return F

def comps_frame(ml, F, level_free):
    X = F.reindex(ml).reset_index(drop=True)
    if level_free:  # comp prices relative to the market index at the comp's own sale month (no price level)
        for c in COMP_LEVEL: X[c] = X[c] - X["lvl"]
        return X[COMP_LEVEL + COMP_OTHER]
    return X[COMP_LEVEL + COMP_OTHER + ["lvl"]]

# ---------------- data per fold ----------------
def fold_data(fold, track):
    """train (full rows sold < cut), test (test-file columns only), truth (sold_price or None for REAL)."""
    L = load(); cut = pd.Timestamp(CUTS[fold])
    tr = L[L.date_sold < cut].reset_index(drop=True)
    cols = OFF_COLS if track == "off_market" else LA_COLS
    if fold == "REAL":
        te = load("test_off_market" if track == "off_market" else "test_list_aware")
        return tr, te[cols].copy(), None, cut
    end = cut + pd.DateOffset(months=4)
    full = L[(L.date_sold >= cut) & (L.date_sold < end)].reset_index(drop=True)
    return tr, full[cols].copy(), full, cut

def window_months(cut):
    cut = pd.Timestamp(cut); return np.arange(cut.year * 12 + cut.month - 1, cut.year * 12 + cut.month + 3)

def index_and_addback(tr, te, cut):
    """Detrending: I[seg, month] from training sales only; add-back = damped-trend + seasonal forecast averaged
    over the 4 window months (de-seasonalised last-3-month level + 0.8-damped 6-month slope + seasonal factor).
    """
    I = build_index(tr)
    st = I.stack().rename("v").reset_index(); st.columns = ["m", "s", "v"]
    key = pd.Series(st.v.values, index=pd.MultiIndex.from_arrays([st.s, st.m]))
    itr = key.reindex(list(zip(seg(tr), mperiod(tr.date_sold)))).values
    assert np.isfinite(itr).all()
    fw = forecast(I, window_months(cut), "seas_trend")
    add = np.where(seg(te) == "condo", fw["condo"].mean(), fw["free"].mean())
    return itr, add

# ---------------- model ----------------
LGB = dict(objective="regression", learning_rate=0.05, n_estimators=1500, num_leaves=127, min_child_samples=40,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2, random_state=0)

def align(Xtr, Xte):
    for c in CATS:
        cats = pd.api.types.union_categoricals([Xtr[c], Xte[c]]).categories
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
    return Xtr, Xte

def lgb_fit_predict(Xtr, y, Xte, seed=0, params=None):
    import lightgbm as lgb
    p = dict(LGB, random_state=seed); p.update(params or {})
    return lgb.LGBMRegressor(**p).fit(Xtr, y).predict(Xte)

# ---------------- metrics ----------------
def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); e = (p - y) / y
    return dict(MdAPE=float(np.median(np.abs(e)) * 100), MAPE=float(np.mean(np.abs(e)) * 100),
                MdSE=float(np.median(e) * 100), n=len(y))
