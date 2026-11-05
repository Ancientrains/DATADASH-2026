"""common.py - shared configuration, ingestion/cleaning and remark (text) features.

Everything that is a *decision* of the competition protocol lives here so that every other module imports it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------
# Paths / protocol constants
# ------------------------------------------------------------------------------------------------
DATA_DIR = Path("../data")
TRAIN_CSV = DATA_DIR / "listings.csv"
TEST_LIST_AWARE_CSV = DATA_DIR / "test_list_aware.csv"
TEST_OFF_MARKET_CSV = DATA_DIR / "test_off_market.csv"
CACHE_DIR = Path("cache")

SEEDS = (0, 1, 2)
BASE_MONTH = pd.Timestamp("2021-01-01")          # month id 0 of every index in this project
TEST_MONTHS = 4                                   # every fold / the final test is a 4-month window
FINAL_CUTOFF = pd.Timestamp("2026-06-01")         # test window: Jun - Sep 2026
FOLDS = {                                         # name -> training cutoff (train: date_sold < cutoff)
    "F1": pd.Timestamp("2026-02-01"),             # test window [cutoff, cutoff + 4 months), by date_sold
    "F2": pd.Timestamp("2025-10-01"),
    "S25": pd.Timestamp("2025-06-01"),            # headline fold
}
CONDO = {"Condo Apt", "Co-Op Apt", "Condo Townhouse"}


def seg(df):
    """Market segment label, exactly as specified."""
    return np.where(df.house_type_name.isin(CONDO), "condo", "free")


# Columns that may never reach a model (post-sale information or the leaked AVM).
POST_SALE_COLS = {"sold_price", "date_sold", "days_on_market"}
AVM_COLS = {"external_avm_estimate", "external_avm_estimate_date"}

REQUIRED_BASE = ["ml_num", "house_type_name", "house_style", "bedrooms", "washrooms", "parking_total", "garage",
                 "house_area", "lot_front", "lot_depth", "municipality", "community", "latitude", "longitude", "remarks"]
REQUIRED_TRAIN = REQUIRED_BASE + ["date_listed", "date_sold", "sold_price", "list_price"]
REQUIRED_LIST_AWARE = REQUIRED_BASE + ["list_price", "date_listed"]
REQUIRED_OFF_MARKET = REQUIRED_BASE

# ------------------------------------------------------------------------------------------------
# Remark flags: the 22 regexes as supplied (applied to the lower-cased text)
# ------------------------------------------------------------------------------------------------
KW = {
    "asis": r"as[- ]is\b",
    "needs_work": r"needs? (?:work|replacing|paint|tlc|updat|new|a full|some)|cosmetic overhaul|renovator|contractor|handyman|fixer|worn|bring your tools|tired",
    "original": r"original|older|bygone|dated|nostalgic|vintage",
    "potential": r"potential|great bones|blank (?:slate|canvas)|personal touch|your vision|make it your own",
    "clean": r"\bclean\b|well[- ]kept|well[- ]maintained|cared for|good (?:shape|condition)|sound and liv",
    "updated": r"updated|refreshed|newer|new countertops|freshly painted|refinished|renewed",
    "renovated": r"renovat|redone|rebuilt|gut",
    "luxury": r"luxur|designer|custom|high-end|spa-like|heated (?:bathroom )?floors|frameless|quartz|chef|built-in appliances|smooth ceilings|premium|upscale|exquisite|wide-plank|pot lights",
    "stunning": r"stunning|exceptional|expansive|truly special|rare (?:find|combination)",
    "turnkey": r"turnkey|move-in ready|nothing (?:but|more)",
    "motivated": r"motivated|best offer|bring (?:your )?offers|quick (?:or flexible )?closing|flexible closing",
    "act_fast": r"won.t last|act quickly|don.t miss|don.t wait",
    "investor": r"investor|investment|income|rental",
    "view": r"unobstructed view|\bviews?\b(?!ing)|skyline|treetops",
    "f2c": r"floor-to-ceiling",
    "amen": r"concierge|gym|rooftop|party room|guest suite",
    "ravine": r"ravine|trails",
    "south": r"south-facing",
    "cozy": r"cozy|compact|efficient|starter|first[- ]time",
    "spacious": r"spacious|room to grow|room for everyone|space for everyone",
    "locker": r"locker",
    "deep_lot": r"deep \d|deep lot|mature trees",
}
KW_COLS = [f"kw_{k}" for k in KW]

# kw_cond_score = sum(weight * flag). Weights are the "controlled price impact" column of the handoff table
# (log-point approximation of the % effect). `motivated` is deliberately absent: its documented effect is on the
# sold/list ratio, not on price level (it stays available as a raw flag for the list-aware model).
COND_WEIGHTS = {"luxury": 0.070, "updated": 0.028, "stunning": 0.022, "renovated": 0.013, "turnkey": 0.009,
                "potential": -0.018, "investor": -0.026, "cozy": -0.035, "needs_work": -0.036, "asis": -0.043}

TEXT_FEATS = KW_COLS + ["kw_cond_score", "kw_n", "kw_cond_x_logsqft", "kw_cond_x_condo", "kw_lux_x_logsqft"]


# ------------------------------------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------------------------------------
def to_num(series: pd.Series) -> pd.Series:
    """Numeric as is; strings such as '3+1' are summed to 4."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    def _f(v):
        if pd.isna(v):
            return np.nan
        nums = re.findall(r"\d+(?:\.\d+)?", str(v))
        return float(sum(float(n) for n in nums)) if nums else np.nan

    return series.map(_f)


def parse_area(series: pd.Series):
    """house_area is either an integer or 'x-y' (|x-y| = 199).  Returns (midpoint sqft, is_range, width)."""
    t = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    m = t.str.extract(r"^(\d+)(?:\s*-\s*(\d+))?$")
    lo, hi = pd.to_numeric(m[0]), pd.to_numeric(m[1])
    sqft = pd.Series(np.where(hi.notna(), (lo + hi) / 2.0, lo), index=series.index, dtype=float)
    return sqft, hi.notna().astype(np.int8), (hi - lo)


def _cat(series: pd.Series) -> np.ndarray:
    return series.astype("object").where(series.notna(), "Missing").astype(str).values


# ------------------------------------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------------------------------------
def clean(df: pd.DataFrame, required) -> pd.DataFrame:
    """Raw csv -> tidy frame (RangeIndex).  Post-sale columns that a file happens to contain and the AVM columns are
    never copied, so they cannot reach a model by accident."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    df = df.reset_index(drop=True)
    o = pd.DataFrame({"ml_num": df["ml_num"].astype(str)})
    for c in ("date_listed", "date_sold"):
        if c in df.columns:
            o[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("sold_price", "list_price"):
        if c in df.columns:
            o[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("house_type_name", "house_style", "municipality", "community"):
        o[c] = _cat(df[c])
    o["bedrooms_n"] = to_num(df["bedrooms"])
    o["washrooms_n"] = to_num(df["washrooms"])
    o["parking_n"] = pd.to_numeric(df["parking_total"], errors="coerce")
    o["garage_n"] = to_num(df["garage"])
    o["sqft"], o["sqft_is_range"], o["sqft_width"] = parse_area(df["house_area"])
    o["lot_front"] = pd.to_numeric(df["lot_front"], errors="coerce")
    o["lot_depth"] = pd.to_numeric(df["lot_depth"], errors="coerce")
    o["lot_area"] = o["lot_front"] * o["lot_depth"]
    o["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    o["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # entity keys: ml_num is a transaction id; building = exact coordinate, unit = coordinate + physical identity
    ck = o["latitude"].round(6).astype(str) + "|" + o["longitude"].round(6).astype(str)
    nocoord = o["latitude"].isna() | o["longitude"].isna()
    o["coord_key"] = np.where(nocoord, "NOCOORD_" + o["ml_num"], ck)
    o["unit_key"] = (o["coord_key"] + "|" + o["house_type_name"] + "|" + o["house_style"] + "|"
                     + df["bedrooms"].astype(str) + "|" + df["washrooms"].astype(str) + "|" + df["house_area"].astype(str))
    o["seg_id"] = o["house_type_name"].isin(CONDO).astype(np.int8)      # 1 = condo, 0 = free(hold)

    # text
    txt = df["remarks"].fillna("").astype(str)
    o["remarks_len"] = txt.str.len().astype(float)
    low = txt.str.lower()
    for k, pat in KW.items():
        o[f"kw_{k}"] = low.str.contains(pat, regex=True).astype(np.int8)
    o["kw_cond_score"] = sum(w * o[f"kw_{k}"] for k, w in COND_WEIGHTS.items()).astype(float)
    o["kw_n"] = o[KW_COLS].sum(axis=1).astype(float)
    logsq = np.log(o["sqft"])
    o["kw_cond_x_logsqft"] = o["kw_cond_score"] * logsq          # interactions with physical attributes
    o["kw_cond_x_condo"] = o["kw_cond_score"] * o["seg_id"]
    o["kw_lux_x_logsqft"] = o["kw_luxury"] * logsq
    return o


def load_train(path=TRAIN_CSV) -> pd.DataFrame:
    o = clean(pd.read_csv(path), REQUIRED_TRAIN)
    keep = o["sold_price"].gt(0) & o["date_sold"].notna() & o["date_listed"].notna()
    o.attrs["dropped_rows"] = int((~keep).sum())
    return o[keep].reset_index(drop=True)


def load_test_off_market(path=TEST_OFF_MARKET_CSV) -> pd.DataFrame:
    return clean(pd.read_csv(path), REQUIRED_OFF_MARKET)


def load_test_list_aware(path=TEST_LIST_AWARE_CSV) -> pd.DataFrame:
    return clean(pd.read_csv(path), REQUIRED_LIST_AWARE)


def assert_no_leak(columns, track: str):
    """Guardrail used before every fit/predict.  track in {'off_market', 'list_aware'}."""
    bad = [c for c in columns if c in POST_SALE_COLS or c in AVM_COLS or "avm" in c]
    if track == "off_market":
        bad += [c for c in columns if "list" in c]
    if bad:
        raise AssertionError(f"leakage guard ({track}): forbidden feature(s) {sorted(set(bad))}")
