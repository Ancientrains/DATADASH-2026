"""End-to-end: features -> stage-1 out-of-time off-market estimates -> final models trained on ALL listings
(cut 2026-06-01, index forecast for Jun-Sep 2026) -> submission CSVs.
From scratch: INTEGRATE_WORK=<empty dir> python make_submission.py   (timings printed per step)"""
from pipeline import *
from stage1 import run_stage1
from run_off import OFF
from run_la import LA
OFF_FINAL = "+kw+comps_lf+dt+3s"
LA_FINAL = "+kw+comps_raw+stack(best)+3s"
T = {}; t0 = time.time()
L = load()
comps_train_role()
for s in pd.date_range("2022-02-01", "2026-02-01", freq="4MS"):
    comps_test_role(L[L.date_sold < s], L[(L.date_sold >= s) & (L.date_sold < s + pd.DateOffset(months=4))], s, "blk")
for name in ("test_off_market", "test_list_aware"):
    comps_test_role(L, load(name), CUTS["REAL"], name)
T["features"] = time.time() - t0; t = time.time()
if not os.path.exists(f"{CACHE}/stage1_best.parquet"): run_stage1("best")
T["stage1"] = time.time() - t; t = time.time()
for track, cfg, out in (("off_market", OFF[OFF_FINAL], "submission_off_market.csv"),
                        ("list_aware", LA[LA_FINAL], "submission_list_aware.csv")):
    d = run_cfg(out, track, cfg, folds=("REAL",))["REAL"]
    ref = load("test_off_market" if track == "off_market" else "test_list_aware")
    assert len(d) == len(ref) and (d.ml_num.values == ref.ml_num.values).all()
    assert np.isfinite(d.p).all() and (d.p > 0).all()
    d[["ml_num", "p"]].rename(columns={"p": "predicted_price"}).to_csv(os.path.join(HERE, out), index=False)
    print(out, len(d), "rows; median pred %.0f" % d.p.median(), flush=True)
    if track == "list_aware":
        r = d.p / ref.list_price.values
        print("list-aware median predicted/list %.4f  (p10 %.3f p90 %.3f)" % (r.median(), r.quantile(.1), r.quantile(.9)))
    T[track] = time.time() - t; t = time.time()
T["total"] = time.time() - t0
print({k: round(v) for k, v in T.items()})
