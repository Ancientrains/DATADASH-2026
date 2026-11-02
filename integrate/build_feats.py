"""Builds and caches all comps features: train-role (once) + test-role for every 4-month block start used by
the folds / stage-1 out-of-time blocks, and for the two real test files (cut 2026-06-01)."""
from common import *
t0 = time.time()
L = load()
comps_train_role(); print(f"train-role done {time.time()-t0:.0f}s", flush=True)
starts = list(pd.date_range("2022-02-01", "2026-02-01", freq="4MS"))
for s in starts:
    hist = L[L.date_sold < s]; q = L[(L.date_sold >= s) & (L.date_sold < s + pd.DateOffset(months=4))]
    comps_test_role(hist, q, s, "blk")
    print(f"block {s.date()} n={len(q)} {time.time()-t0:.0f}s", flush=True)
for name in ("test_off_market", "test_list_aware"):
    comps_test_role(L, load(name), "2026-06-01", name)
print(f"all done {time.time()-t0:.0f}s", flush=True)

# sanity: train-role features equal the per-fold version built by ideas_comps
old = pd.read_parquet(f"{HERE}/../ideas_comps/cache_listed_2026-02-01.parquet")
tr = L[L.date_sold < pd.Timestamp("2026-02-01")].reset_index(drop=True)
new = comps_train_role().reindex(tr.ml_num.values).reset_index(drop=True)
for c in ["rep_lp", "bld_lp", "bld_ppsf", "comm_lp", "knn_lp", "lvl"]:
    a, b = old[c].values, new[c].values
    print(c, "max|diff|", np.nanmax(np.abs(a - b)), "nan mismatch", int((np.isnan(a) != np.isnan(b)).sum()))
