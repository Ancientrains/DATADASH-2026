import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "integrate"), str(ROOT / "shared")]

import harness
from common import comps_test_role


def test_off_market_columns_never_include_list_price():
    assert "list_price" not in harness.OFF_COLS
    assert "list_price" in harness.LA_COLS


def test_comparables_reject_sales_on_or_after_cutoff():
    hist = pd.DataFrame({"date_sold": pd.to_datetime(["2026-02-01"])})
    with pytest.raises(AssertionError):
        comps_test_role(hist, pd.DataFrame(), "2026-02-01", "test")


def test_e1_training_references_completed_prior_months():
    source = (ROOT / "integrate" / "common.py").read_text()
    assert 'date_listed.dt.to_period("M").dt.to_timestamp()' in source
