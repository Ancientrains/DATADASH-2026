"""Check the two required submission files."""
from pathlib import Path
import numpy as np
import pandas as pd
import data

def check_all(folder="submissions"):
    ok = True
    for track in ("off_market", "list_aware"):
        path = Path(folder) / f"submission_{track}.csv"
        problems = []
        try: d = pd.read_csv(path, dtype={"ml_num": str})
        except Exception as exc: d = None; problems.append(str(exc))
        if d is not None:
            if list(d.columns) != ["ml_num", "predicted_price"]: problems.append("columns must be ml_num,predicted_price")
            expected = set(data.load(track).ml_num.astype(str))
            p = pd.to_numeric(d.predicted_price, errors="coerce") if "predicted_price" in d else pd.Series(dtype=float)
            if len(d) != len(expected) or set(d.ml_num.astype(str)) != expected or d.ml_num.duplicated().any(): problems.append("IDs do not match test data")
            if p.isna().any() or np.isinf(p).any() or (p <= 0).any(): problems.append("prices must be positive numbers")
        print(f"{path.name}: " + ("OK" if not problems else "; ".join(problems)))
        ok &= not problems
    return ok
