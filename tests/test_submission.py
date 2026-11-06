import pandas as pd
import check_submission


def test_submission_checker_accepts_exact_test_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(check_submission.data, "load", lambda _: pd.DataFrame({"ml_num": ["A", "B"]}))
    for track in ("off_market", "list_aware"):
        pd.DataFrame({"ml_num": ["A", "B"], "predicted_price": [1.0, 2.0]}).to_csv(tmp_path / f"submission_{track}.csv", index=False)
    assert check_submission.check_all(tmp_path)
