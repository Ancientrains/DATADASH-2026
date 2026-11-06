import pandas as pd
import pytest
import check_submission


def test_submission_checker_accepts_exact_test_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(check_submission.data, "load", lambda _: pd.DataFrame({"ml_num": ["A", "B"]}))
    for track in ("off_market", "list_aware"):
        pd.DataFrame({"ml_num": ["A", "B"], "predicted_price": [1.0, 2.0]}).to_csv(tmp_path / f"submission_{track}.csv", index=False)
    assert check_submission.check_all(tmp_path)


def write_valid_files(path):
    for track in ("off_market", "list_aware"):
        pd.DataFrame({"ml_num": ["A", "B"], "predicted_price": [1.0, 2.0]}).to_csv(path / f"submission_{track}.csv", index=False)


@pytest.mark.parametrize("bad", ["duplicate", "missing", "bad_columns", "zero", "non_numeric"])
def test_submission_checker_rejects_each_bad_case(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(check_submission.data, "load", lambda _: pd.DataFrame({"ml_num": ["A", "B"]}))
    write_valid_files(tmp_path)
    if bad == "duplicate":
        frame = pd.DataFrame({"ml_num": ["A", "A"], "predicted_price": [1.0, 2.0]})
    elif bad == "missing":
        frame = pd.DataFrame({"ml_num": ["A"], "predicted_price": [1.0]})
    elif bad == "bad_columns":
        frame = pd.DataFrame({"ml_num": ["A", "B"], "wrong": [1.0, 2.0]})
    elif bad == "zero":
        frame = pd.DataFrame({"ml_num": ["A", "B"], "predicted_price": [0, 2.0]})
    else:
        frame = pd.DataFrame({"ml_num": ["A", "B"], "predicted_price": ["oops", 2.0]})
    frame.to_csv(tmp_path / "submission_off_market.csv", index=False)
    assert not check_submission.check_all(tmp_path)
