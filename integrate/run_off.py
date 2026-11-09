"""Off-market cumulative ablation. Usage: python run_off.py [config names...]"""
from pipeline import *
S3 = (0, 1, 2)
OFF = {
    "ref":                         dict(),
    "+kw":                         dict(kw=True),
    "+kw+comps_raw":               dict(kw=True, comps="raw"),
    "+kw+comps_lf":                dict(kw=True, comps="lf"),
    "+kw+detrend":                 dict(kw=True, detrend=True),
    "+kw+comps_raw+dt":            dict(kw=True, comps="raw", detrend=True),
    "+kw+comps_lf+dt":             dict(kw=True, comps="lf", detrend=True),
    "+kw+comps_lf+dt+3s":          dict(kw=True, comps="lf", detrend=True, seeds=S3),
    "+...+ff0.5 (1s)":             dict(kw=True, comps="lf", detrend=True, ff=0.5),
    "+...+ff0.5+3s":               dict(kw=True, comps="lf", detrend=True, ff=0.5, seeds=S3),
    "+...+ff0.5+age (1s)":          dict(kw=True, comps="lf", detrend=True, ff=0.5, age=True),
}
if __name__ == "__main__":
    names = sys.argv[1:] or list(OFF)
    rows = []
    for n in names:
        t = time.time(); r = run_cfg(n, "off_market", OFF[n]); rows.append(summarize(n, r))
        print(pd.DataFrame([rows[-1]]).to_string(index=False), f"{time.time()-t:.0f}s", flush=True)
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).to_string(index=False))
