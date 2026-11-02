"""Per-type breakdown of the final pipelines + test-mix-reweighted validation error."""
from pipeline import *
from run_off import OFF
from run_la import LA
pd.set_option("display.width", 200)
for track, name, cfg, testfile in (("off_market", "+kw+comps_lf+dt+3s", OFF["+kw+comps_lf+dt+3s"], "test_off_market"),
                                  ("list_aware", "+kw+comps_raw+stack(best)+3s", LA["+kw+comps_raw+stack(best)+3s"], "test_list_aware")):
    R = run_cfg(name, track, cfg)
    mix = load(testfile).house_type_name.value_counts(normalize=True)
    print(f"\n=== {track} {name}")
    rows = []
    for f, d in R.items():
        ape = np.abs(d.p / d.y - 1) * 100; se = (d.p / d.y - 1) * 100
        g = pd.DataFrame({"type": d.type, "ape": ape, "se": se})
        by = g.groupby("type").agg(n=("ape", "size"), MdAPE=("ape", "median"), MdSE=("se", "median"))
        # reweight rows so the fold's house-type mix equals the hidden test file's mix
        w = g.type.map(mix / g.type.value_counts(normalize=True)).fillna(0).values
        o = np.argsort(ape.values); cw = np.cumsum(w[o]) / w.sum()
        rw = ape.values[o][np.searchsorted(cw, 0.5)]
        rows.append(dict(fold=f, MdAPE=round(np.median(ape), 2), MdAPE_testmix=round(rw, 2), MAPE=round(ape.mean(), 2)))
        if f == "S25": print("S25 by house_type:\n", by.round(2).to_string())
    print(pd.DataFrame(rows).to_string(index=False))
    print("test file type mix:", mix.round(3).to_dict())
    # per-month MdAPE within S25 (does error grow with horizon?)
    _, _, full, _ = fold_data("S25", track)
    d = R["S25"]; mo = full.date_sold.dt.month.values
    print("S25 MdAPE by sale month:", {m: round(float(np.median(np.abs(d.p[mo == m] / d.y[mo == m] - 1)) * 100), 2) for m in sorted(set(mo))})
