import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "integrate"), str(ROOT / "shared")]

import harness
import common
import pipeline


def test_off_market_feature_path_drops_list_price(monkeypatch):
    monkeypatch.setattr(pipeline, "base_features", lambda df: pd.DataFrame(index=df.index))
    monkeypatch.setattr(pipeline, "align", lambda train, test: (train, test))
    train = pd.DataFrame({"ml_num": ["A"], "list_price": [100]})
    test = pd.DataFrame({"ml_num": ["B"], "list_price": [200]})
    x_train, x_test = pipeline.build_X(train, test, pd.Timestamp("2026-02-01"), "F1", "off_market", {})
    assert "list_price" not in x_train and "list_price" not in x_test


def test_comparables_reject_sales_on_or_after_cutoff():
    hist = pd.DataFrame({"date_sold": pd.to_datetime(["2026-02-01"])})
    with pytest.raises(AssertionError):
        common.comps_test_role(hist, pd.DataFrame(), "2026-02-01", "test")


def test_e1_training_references_completed_prior_months(tmp_path, monkeypatch):
    listings = pd.DataFrame({"ml_num": ["A"], "date_listed": pd.to_datetime(["2026-02-17"])})
    seen = {}
    monkeypatch.setattr(common, "CACHE", str(tmp_path))
    monkeypatch.setattr(common, "load", lambda: listings)
    monkeypatch.setattr(common, "CompsIndex", lambda _: object())
    def fake_build(hist, query, reference, index):
        seen["reference"] = reference.copy()
        return pd.DataFrame(index=query.index)
    monkeypatch.setattr(common, "comps_build", fake_build)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *_args, **_kwargs: None)
    common.comps_train_role()
    assert seen["reference"].iloc[0] == pd.Timestamp("2026-02-01")
