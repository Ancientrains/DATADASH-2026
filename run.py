"""Run final predictions or check submission files.

Usage:
    python run.py predict
    python run.py check
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from check_submission import check_all

ROOT = Path(__file__).parent


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


COMMANDS = {"predict": predict, "check": check}
if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    result = COMMANDS[sys.argv[1]]()
    raise SystemExit(1 if result is False else 0)
