"""List-aware cumulative ablation. Usage: python run_la.py [config names...]"""
from pipeline import *
S3 = (0, 1, 2)
LA = {
    "ref (log + log_list)":                 dict(),
    "ratio target":                         dict(target="ratio"),
    "+kw":                                  dict(target="ratio", kw=True),
    "+kw+comps_raw":                        dict(target="ratio", kw=True, comps="raw"),
    "+kw+comps_lf":                         dict(target="ratio", kw=True, comps="lf"),
    "+kw+comps+stack(kw)":                  dict(target="ratio", kw=True, comps="lf", stack="kw"),
    "+kw+comps+stack(best)":                dict(target="ratio", kw=True, comps="lf", stack="best"),
    "+kw+comps+stack(best)+3s":             dict(target="ratio", kw=True, comps="lf", stack="best", seeds=S3),
    "+kw+comps_raw+stack(best)":            dict(target="ratio", kw=True, comps="raw", stack="best"),
    "+kw+comps_raw+stack(best)+3s":         dict(target="ratio", kw=True, comps="raw", stack="best", seeds=S3),
    "ratio+kw+stack(best) [no comps]":       dict(target="ratio", kw=True, stack="best"),
}
if __name__ == "__main__":
    names = sys.argv[1:] or list(LA)
    rows = []
    for n in names:
        t = time.time(); r = run_cfg(n, "list_aware", LA[n]); rows.append(summarize(n, r))
        print(pd.DataFrame([rows[-1]]).to_string(index=False), f"{time.time()-t:.0f}s", flush=True)
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).to_string(index=False))
