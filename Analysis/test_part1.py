"""test_part1.py - correctness tests for the Part 1 modules (python test_part1.py)."""
import numpy as np, pandas as pd
import common as C
from comps import build_comp_features, BLD_N, COMM_DAYS
from features import build_frames, fold_queries, OFF_MARKET_FEATS
from trend import (get_vintage_index, month_id, index_at, seasonal_factors, forecast_index, cal_month, MIN_VINTAGE)

train = C.load_train()
V = get_vintage_index(train)

# 1. comps engine == brute force (strict date_sold < date_listed; deflation uses only the as-of vintage)
rng = np.random.default_rng(0)
tr = train[train.date_sold < C.FOLDS["S25"]].reset_index(drop=True)
rows = rng.choice(len(tr), 300, replace=False)
q = tr.iloc[rows].reset_index(drop=True)
feat = build_comp_features(tr, q, q["date_listed"], V)
bad = 0
for i, r in q.iterrows():
    h = tr[tr.date_sold < r.date_listed]
    v = month_id(r.date_listed)
    def defl(d):
        return np.log(d.sold_price.values) - V[d.seg_id.values, v, np.minimum(month_id(d.date_sold), v - 1)]
    b = h[h.coord_key == r.coord_key].sort_values("date_sold").tail(BLD_N)
    u = h[h.unit_key == r.unit_key].sort_values("date_sold").tail(1)
    m = h[(h.house_type_name == r.house_type_name) & (h.community == r.community) & (h.date_sold >= r.date_listed - pd.Timedelta(days=COMM_DAYS))]
    exp = {"bld_n": len(b) if len(b) else np.nan, "comm_n": len(m) if len(m) else np.nan,
           "bld_lp_raw": np.median(np.log(b.sold_price)) if len(b) else np.nan,
           "bld_lp": np.median(defl(b)) if len(b) and v >= MIN_VINTAGE else np.nan,
           "comm_lp": np.median(defl(m)) if len(m) and v >= MIN_VINTAGE else np.nan,
           "rep_lp_raw": np.log(u.sold_price.iloc[0]) if len(u) else np.nan,
           "rep_lp": defl(u)[0] if len(u) and v >= MIN_VINTAGE else np.nan}
    for k, e in exp.items():
        g = feat.loc[i, k]
        if not ((np.isnan(e) and np.isnan(g)) or np.isclose(e, g, atol=1e-9)):
            bad += 1; print("MISMATCH", i, k, e, g)
print(f"[1] comps vs brute force on 300 rows x 7 features: {bad} mismatches")
assert bad == 0

# 2. vintage index uses no future data: vintage v must not change when later sales are removed
v0 = 40
cut = train[train.date_sold < pd.Timestamp("2021-01-01") + pd.DateOffset(months=v0)]
from trend import fit_vintage_index
V2 = fit_vintage_index(cut, last_vintage=v0, verbose=False)
assert np.allclose(V2[:, v0, :v0], V[:, v0, :v0], equal_nan=True, atol=1e-6)
print("[2] vintage 40 identical when computed from sales < month 40 only: OK")

# 3. forecast recovers a known trend + seasonality
t = np.arange(61)
S_true = 0.04 * np.sin(2 * np.pi * (cal_month(t) - 2) / 12)
x = 0.006 * t + S_true
S_hat = seasonal_factors(x)
fc = forecast_index(x)
truth = 0.006 * np.arange(61, 65) + 0.04 * np.sin(2 * np.pi * (cal_month(np.arange(61, 65)) - 2) / 12)
print(f"[3] seasonal factor max abs error {np.abs(S_hat - (0.04*np.sin(2*np.pi*(np.arange(12)-2)/12))).max():.4f}; "
      f"forecast error (log pts) {np.round(fc - truth, 4)}")
assert np.abs(fc - truth).max() < 0.02

# 4. frames: strict OOT, no leakage columns, shapes, test-file path
cutoff = C.FOLDS["S25"]
fr = build_frames(train, fold_queries(train, cutoff), cutoff, V, "off_market", verbose=False)
assert fr.meta_tr.date_sold.max() < cutoff and (fr.meta_q.date_sold >= cutoff).all()
assert list(fr.X_tr.columns) == OFF_MARKET_FEATS == list(fr.X_q.columns)
off = C.load_test_off_market()
fo = build_frames(train, off, C.FINAL_CUTOFF, V, "off_market", verbose=False)
assert len(fo.X_q) == len(off) and fo.addback_q.shape == (len(off),)
fl = build_frames(train, C.load_test_list_aware(), C.FINAL_CUTOFF, V, "list_aware", verbose=False)
print(f"[4] fold frame {fr.X_tr.shape}/{fr.X_q.shape}; final off-market query {fo.X_q.shape}; list-aware query {fl.X_q.shape}")
try:
    C.assert_no_leak(OFF_MARKET_FEATS + ["list_price"], "off_market"); raise SystemExit("guard failed")
except AssertionError as e:
    print("[5] leakage guard raises on list_price in off-market:", str(e)[:70])
# 6. training-row comps do not depend on the cutoff (cached all-history pass == cutoff-restricted pass)
from features import train_comps
allc = train_comps(train, V)
sub = tr.iloc[rows[:150]]
restricted = build_comp_features(tr, sub, sub["date_listed"], V)
pos = train.index[train.ml_num.isin(sub.ml_num)]
full = allc.loc[pos].set_index(train.loc[pos, "ml_num"]).loc[sub.ml_num.values].reset_index(drop=True)
assert np.allclose(full[restricted.columns].to_numpy(), restricted.reset_index(drop=True).to_numpy(), equal_nan=True)
print("[6] comps of training rows identical with all-history vs cutoff-restricted history: OK")
# 7. market ratio == brute force, strictly before the reference date
from comps import market_ratio
sample = tr.sample(120, random_state=3)
got = market_ratio(tr, sample["date_listed"], sample["seg_id"])
exp = []
for _, r in sample.iterrows():
    h = tr[(tr.seg_id == r.seg_id) & (tr.date_sold < r.date_listed) & (tr.date_sold >= r.date_listed - pd.Timedelta(days=C.RATIO_DAYS))]
    exp.append(np.median(np.log(h.sold_price / h.list_price)) if len(h) else np.nan)
assert np.allclose(got, np.array(exp), equal_nan=True)
print("[7] trailing sold/list market ratio equals brute force (strictly before the reference date): OK")
print("ALL PART 1 TESTS PASSED")