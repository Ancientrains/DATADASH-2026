"""Run validation, reports, predictions, or submission checks.

Usage:
    python run.py validate
    python run.py report
    python run.py predict
    python run.py check
"""
import os
import shutil
import subprocess
import sys
import json
from pathlib import Path

import pandas as pd

from check_submission import check_all

ROOT = Path(__file__).parent

def validate():
    env = dict(os.environ)
    env.setdefault("DATADASH_DATA", str(ROOT / "data"))
    # Keep official validation separate from prediction/experiment caches.
    work = ROOT / ".work" / "official_validation"
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    env["INTEGRATE_WORK"] = str(work)
    subprocess.run([sys.executable, "-c", "from official_validate import run; from pathlib import Path; run(Path('../outputs/validation_predictions.csv'))"], cwd=ROOT / "integrate", env=env, check=True)

def report():
    p = pd.read_csv(ROOT / "outputs" / "validation_predictions.csv")
    out = {}
    for track, d in p.groupby("track"):
        e = (d.predicted_price - d.sold_price) / d.sold_price * 100
        out[track] = {"n": len(d), "MdAPE": round(float(e.abs().median()), 2), "MAPE": round(float(e.abs().mean()), 2), "MdSE": round(float(e.median()), 2)}
    (ROOT / "outputs" / "validation_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def predict():
    work = Path(os.environ.get("INTEGRATE_WORK", ROOT / ".work"))
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("DATADASH_DATA", str(ROOT / "data"))
    env["INTEGRATE_WORK"] = str(work)
    subprocess.run([sys.executable, "make_submission.py"], cwd=ROOT / "integrate", env=env, check=True)
    out = ROOT / "submissions"
    out.mkdir(exist_ok=True)
    for name in ("submission_off_market.csv", "submission_list_aware.csv"):
        shutil.copy2(ROOT / "integrate" / name, out / name)


def check():
    return check_all(ROOT / "submissions")


COMMANDS = {"validate": validate, "report": report, "predict": predict, "check": check}
if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    result = COMMANDS[sys.argv[1]]()
    raise SystemExit(1 if result is False else 0)
