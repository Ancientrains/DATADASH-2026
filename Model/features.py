"""features.py - assemble model-ready frames for a given cutoff (both tracks share one comps pass).

* Training-row comps depend only on each row's own listing date (all eligible comps sold before it), so they are
  computed once for every training row and re-used for every cutoff (train_comps, cached on disk).
* Held-out / test rows get comps referenced to the cutoff, from the history sold before it.
* Frames carry every column; models select OFF_MARKET_FEATS or the list-aware list (assert_no_leak guards both).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import (CACHE_DIR, CACHE_VERSION, COMPS_VERSION, LIST_MARKET_RATIO, TEST_MONTHS, TEXT_FEATS,
                    assert_no_leak)
from comps import COMP_FEATS, LEVEL_FREE_FEATS, RAW_FEATS, build_comp_features, market_ratio
from trend import addback, detrend_target, month_id

PHYS_NUM = ["sqft", "sqft_is_range", "bedrooms_n", "washrooms_n", "parking_n", "garage_n",
            "lot_front", "lot_depth", "lot_area", "latitude", "longitude", "remarks_len", "seg_id"]
CAT_COLS = ["house_type_name", "house_style", "municipality", "community"]

OFF_MARKET_FEATS = PHYS_NUM + TEXT_FEATS + CAT_COLS + LEVEL_FREE_FEATS
LIST_AWARE_BASE_FEATS = (PHYS_NUM + TEXT_FEATS + CAT_COLS + RAW_FEATS + ["bld_n", "bld_age", "rep_age", "comm_n"]
                         + (["mkt_lr"] if LIST_MARKET_RATIO else []))
BASE_COLS = PHYS_NUM + TEXT_FEATS + CAT_COLS
META_COLS = ["ml_num", "sold_price", "list_price", "date_listed", "date_sold", "seg_id"]


@dataclass
class Frames:
    cutoff: pd.Timestamp
    X_tr: pd.DataFrame                  # all columns
    y_tr: np.ndarray                    # off-market target: ln(price) - index(sale month)
    pos_tr: np.ndarray                  # positions of the training rows inside train_all
    X_q: pd.DataFrame
    addback_q: np.ndarray               # per-row forecast index to add back
    meta_tr: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta_q: pd.DataFrame = field(default_factory=pd.DataFrame)


def data_key(train_all: pd.DataFrame) -> str:
    return hashlib.md5(f"{len(train_all)}|{train_all['sold_price'].sum():.0f}|{train_all['date_sold'].max()}|{CACHE_VERSION}".encode()).hexdigest()[:12]


def train_comps(train_all: pd.DataFrame, V: np.ndarray) -> pd.DataFrame:
    """Comps of every training row as of its own listing date (cached).  Valid for any cutoff, because the eligible
    history of a row sold before the cutoff lies entirely before the cutoff."""
    path = CACHE_DIR / f"train_comps_{data_key(train_all)}_{COMPS_VERSION}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    CACHE_DIR.mkdir(exist_ok=True)
    comps = build_comp_features(train_all, train_all, train_all["date_listed"], V)
    comps["mkt_lr"] = market_ratio(train_all, train_all["date_listed"], train_all["seg_id"])
    comps.to_pickle(path)
    return comps


def categorical_levels(train: pd.DataFrame) -> dict:
    return {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}


def _apply_cats(X: pd.DataFrame, levels: dict) -> pd.DataFrame:
    X = X.copy()
    for c, lv in levels.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=lv)         # unseen level -> NaN
    return X


def window_positions(train_all: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """Positions of the rows sold in [cutoff, cutoff + 4 months)."""
    end = cutoff + pd.DateOffset(months=TEST_MONTHS)
    d = train_all["date_sold"]
    return np.where((d >= cutoff) & (d < end))[0]


def fold_queries(train_all: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return train_all.iloc[window_positions(train_all, cutoff)].reset_index(drop=True)


def build_frames(train_all: pd.DataFrame, queries: pd.DataFrame, cutoff: pd.Timestamp, V: np.ndarray,
                 track=None, verbose: bool = True, comps_all: pd.DataFrame | None = None) -> Frames:
    """Strict out-of-time frames: training = rows with date_sold < cutoff; queries are referenced to the cutoff."""
    pos = np.where(train_all["date_sold"] < cutoff)[0]
    tr = train_all.iloc[pos].reset_index(drop=True)
    q = queries.reset_index(drop=True)
    if verbose:
        print(f"  cutoff {cutoff.date()}: {len(tr):,} training sales (< cutoff), {len(q):,} query rows")
    if comps_all is None:
        comps_all = train_comps(train_all, V)
    comps_tr = comps_all.iloc[pos].reset_index(drop=True)
    comps_q = build_comp_features(tr, q, cutoff, V)
    comps_q["mkt_lr"] = market_ratio(tr, cutoff, q["seg_id"])
    levels = categorical_levels(tr)
    X_tr = _apply_cats(pd.concat([tr[BASE_COLS], comps_tr], axis=1), levels)
    X_q = _apply_cats(pd.concat([q[BASE_COLS], comps_q], axis=1), levels)
    ab = addback(V, cutoff, TEST_MONTHS)                                   # [free, condo]
    mt = [c for c in META_COLS if c in tr.columns]
    mq = [c for c in META_COLS if c in q.columns]
    return Frames(cutoff=cutoff, X_tr=X_tr, y_tr=detrend_target(tr, V, month_id(cutoff)), pos_tr=pos, X_q=X_q,
                  addback_q=ab[q["seg_id"].to_numpy()], meta_tr=tr[mt], meta_q=q[mq])