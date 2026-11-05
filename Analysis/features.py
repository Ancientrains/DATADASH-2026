"""features.py - assemble model-ready frames for a given cutoff.

build_frames(train_all, queries, cutoff, V, track) returns a Frames object with
  X_tr, y_tr        training features and target (off-market: ln price - index(sale month); list-aware: not built here)
  X_q               features for the rows to predict (comps referenced to the cutoff)
  addback_q         per-row forecast index to add back (off-market)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import (TEST_MONTHS, TEXT_FEATS, assert_no_leak)
from comps import COMP_FEATS, LEVEL_FREE_FEATS, RAW_FEATS, build_comp_features
from trend import addback, detrend_target, month_id

PHYS_NUM = ["sqft", "sqft_is_range", "bedrooms_n", "washrooms_n", "parking_n", "garage_n",
            "lot_front", "lot_depth", "lot_area", "latitude", "longitude", "remarks_len", "seg_id"]
CAT_COLS = ["house_type_name", "house_style", "municipality", "community"]

OFF_MARKET_FEATS = PHYS_NUM + TEXT_FEATS + CAT_COLS + LEVEL_FREE_FEATS
LIST_AWARE_BASE_FEATS = PHYS_NUM + TEXT_FEATS + CAT_COLS + RAW_FEATS + ["bld_n", "bld_age", "rep_age", "comm_n"]


@dataclass
class Frames:
    cutoff: pd.Timestamp
    X_tr: pd.DataFrame
    y_tr: np.ndarray
    X_q: pd.DataFrame
    addback_q: np.ndarray                      # per-row index forecast (log points); zeros for list-aware use
    meta_tr: pd.DataFrame = field(default_factory=pd.DataFrame)     # ml_num, sold_price, list_price, seg_id, date_*
    meta_q: pd.DataFrame = field(default_factory=pd.DataFrame)


def categorical_levels(train: pd.DataFrame) -> dict:
    return {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}


def _apply_cats(X: pd.DataFrame, levels: dict) -> pd.DataFrame:
    X = X.copy()
    for c, lv in levels.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=lv)         # unseen level -> NaN
    return X


def build_frames(train_all: pd.DataFrame, queries: pd.DataFrame, cutoff: pd.Timestamp, V: np.ndarray,
                 track: str = "off_market", verbose: bool = True) -> Frames:
    """train_all: every cleaned training row.  Rows with date_sold >= cutoff are discarded (strict OOT).
    queries: cleaned rows to predict (held-out fold rows or the competition test file)."""
    tr = train_all[train_all["date_sold"] < cutoff].reset_index(drop=True)
    q = queries.reset_index(drop=True)
    v = month_id(cutoff)
    if verbose:
        print(f"  cutoff {cutoff.date()}: {len(tr):,} training sales (< cutoff), {len(q):,} query rows")

    comps_tr = build_comp_features(tr, tr, tr["date_listed"], V)          # as-of own listing date
    comps_q = build_comp_features(tr, q, cutoff, V)                        # as-of the cutoff
    levels = categorical_levels(tr)

    cols = PHYS_NUM + TEXT_FEATS + CAT_COLS
    X_tr = _apply_cats(pd.concat([tr[cols], comps_tr], axis=1), levels)
    X_q = _apply_cats(pd.concat([q[cols], comps_q], axis=1), levels)

    if track == "off_market":
        feats = OFF_MARKET_FEATS
        assert_no_leak(feats, "off_market")
        y_tr = detrend_target(tr, V, v)
        ab = addback(V, cutoff, TEST_MONTHS)                              # [free, condo]
        addback_q = ab[q["seg_id"].to_numpy()]
    else:
        feats = LIST_AWARE_BASE_FEATS
        assert_no_leak(feats, "list_aware")
        y_tr = np.log(tr["sold_price"].to_numpy(dtype=float))              # list-aware target is built in the model
        addback_q = np.zeros(len(q))
    mt = [c for c in ("ml_num", "sold_price", "list_price", "date_listed", "date_sold", "seg_id") if c in tr.columns]
    mq = [c for c in ("ml_num", "sold_price", "list_price", "date_listed", "date_sold", "seg_id") if c in q.columns]
    return Frames(cutoff=cutoff, X_tr=X_tr[feats], y_tr=y_tr, X_q=X_q[feats], addback_q=addback_q,
                  meta_tr=tr[mt], meta_q=q[mq])


def fold_queries(train_all: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Held-out rows of a validation fold: sold in [cutoff, cutoff + 4 months)."""
    end = cutoff + pd.DateOffset(months=TEST_MONTHS)
    return train_all[(train_all["date_sold"] >= cutoff) & (train_all["date_sold"] < end)].reset_index(drop=True)
