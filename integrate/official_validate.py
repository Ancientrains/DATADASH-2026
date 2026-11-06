"""Monthly walk-forward validation used by the root run command."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

from common import CACHE, LA_COLS, OFF_COLS, comps_test_role, load
from pipeline import predict
from run_off import OFF
from run_la import LA
from stage1 import STAGE1, run_stage1

MONTHS = pd.period_range("2025-12", "2026-05", freq="M")

def ensemble(tr, te, cut, track, cfg, comps):
    values = [np.log(predict(tr, te, cut, "REAL", track, cfg, seed=s, te_comps=comps)) for s in cfg.get("seeds", (0,))]
    return np.exp(np.mean(values, axis=0))

def run(output):
    listings = load()
    base = Path(CACHE) / "stage1_best.parquet"
    if not base.exists(): run_stage1("best")
    stage = pd.read_parquet(base)
    rows = []
    for month in MONTHS:
        cut = month.start_time
        tr = listings[listings.date_sold < cut].reset_index(drop=True)
        full = listings[listings.date_sold.dt.to_period("M") == month].reset_index(drop=True)
        off_te = full[OFF_COLS].copy()
        off_comps = comps_test_role(tr, off_te, cut, f"official_off_{month}")
        p_off = ensemble(tr, off_te, cut, "off_market", OFF["+kw+comps_lf+dt+3s"], off_comps)
        rows.append(pd.DataFrame({"ml_num": full.ml_num, "track": "off_market", "month": str(month), "sold_price": full.sold_price, "predicted_price": p_off, "house_type_name": full.house_type_name}))

        la_te = full[LA_COLS].copy()
        la_comps = comps_test_role(tr, la_te, cut, f"official_la_{month}")
        # A current-cut off-market estimate for list-aware test rows; training
        # rows retain their earlier-block stage-one estimates from `stage`.
        stage_name = f"official_{month}"
        # Match stage1.py: stack features are single-seed off-market estimates.
        p_stage = predict(tr, off_te, cut, "REAL", "off_market", STAGE1["best"], te_comps=off_comps)
        current = pd.DataFrame({"ml_num": la_te.ml_num, "om_log": np.log(p_stage)})
        pd.concat([stage, current]).drop_duplicates("ml_num", keep="last").to_parquet(Path(CACHE) / f"stage1_{stage_name}.parquet")
        cfg = dict(LA["+kw+comps_raw+stack(best)+3s"], stack=stage_name)
        p_la = ensemble(tr, la_te, cut, "list_aware", cfg, la_comps)
        rows.append(pd.DataFrame({"ml_num": full.ml_num, "track": "list_aware", "month": str(month), "sold_price": full.sold_price, "predicted_price": p_la, "house_type_name": full.house_type_name}))
        print(f"{month}: trained on {len(tr):,} sales; predicted {len(full):,} homes", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(output, index=False)
