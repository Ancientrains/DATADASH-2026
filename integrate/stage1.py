"""Stage 1 of the list-aware stack: out-of-time off-market log-price estimate.
For every 4-month block [s, s+4m) of sale dates, s = 2022-02-01 ... 2026-02-01 (block starts include the fold cuts
2025-06-01, 2025-10-01, 2026-02-01), the off-market pipeline is trained on sales sold < s and predicts the block,
with comps referenced to s and the index forecast for the block's months. Rows sold before 2022-02 get NaN.
REAL: trained on all listings (cut 2026-06-01), predicts test_list_aware rows.
Usage: python stage1.py <name>   (name in STAGE1)"""
from pipeline import *
STAGE1 = {"kw": dict(kw=True), "best": dict(kw=True, comps="lf", detrend=True)}

def run_stage1(name):
    cfg = STAGE1[name]; L = load(); t0 = time.time(); rows = []
    for s in pd.date_range("2022-02-01", "2026-02-01", freq="4MS"):
        tr = L[L.date_sold < s].reset_index(drop=True)
        full = L[(L.date_sold >= s) & (L.date_sold < s + pd.DateOffset(months=4))].reset_index(drop=True)
        te = full[OFF_COLS].copy()
        p = predict(tr, te, s, None, "off_market", cfg, te_comps=comps_test_role(None, None, s, "blk"))
        rows.append(pd.DataFrame({"ml_num": te.ml_num.values, "om_log": np.log(p)}))
        print(s.date(), len(tr), len(te), "MdAPE %.2f" % metrics(full.sold_price, p)["MdAPE"], f"{time.time()-t0:.0f}s", flush=True)
    cut = pd.Timestamp(CUTS["REAL"])
    te = load("test_list_aware")[OFF_COLS].reset_index(drop=True)
    p = predict(L.reset_index(drop=True), te, cut, "REAL", "off_market", cfg,
                te_comps=comps_test_role(None, None, cut, "test_list_aware"))
    rows.append(pd.DataFrame({"ml_num": te.ml_num.values, "om_log": np.log(p)}))
    S = pd.concat(rows); assert S.ml_num.is_unique
    S.to_parquet(f"{CACHE}/stage1_{name}.parquet"); print("saved", len(S), f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    run_stage1(sys.argv[1])
