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
DATA_DIR = Path("data")
TRAIN_CSV = DATA_DIR / "listings.csv"
TEST_LIST_AWARE_CSV = DATA_DIR / "test_list_aware.csv"
TEST_OFF_MARKET_CSV = DATA_DIR / "test_off_market.csv"
CACHE_DIR = Path("cache")
PARAMS_DIR = Path("params")                       # tuned LightGBM parameters (commit these for reproducibility)
REPORT_DIR = Path("reports")
OUT_DIR = Path(".")
SUBMISSION_OFF_MARKET = OUT_DIR / "submission_off_market.csv"
SUBMISSION_LIST_AWARE = OUT_DIR / "submission_list_aware.csv"
ID_COL = "ml_num"                                 # ASSUMPTION - confirm against the competition's submission template
PRED_COL = "sold_price"                           # ASSUMPTION - confirm against the competition's submission template

CACHE_VERSION = "p2-hybrid-1"                     # bump to invalidate every cached index / comps / block file
COMPS_VERSION = "3"                               # invalidates cached comps to include spatial k-NN and bedroom matches
RATIO_DAYS = 90                                   # window of the trailing sold/list market ratio
# Opt-in list-aware feature: trailing median ln(sold/list) of the same segment over the RATIO_DAYS before the reference date
LIST_MARKET_RATIO = True
SEEDS = (0, 1, 2)
BASE_MONTH = pd.Timestamp("2021-01-01")          # month id 0 of every index in this project
TEST_MONTHS = 4                                   # every fold / the final test is a 4-month window
FINAL_CUTOFF = pd.Timestamp("2026-06-01")         # test window: Jun - Sep 2026
FOLDS = {                                         # name -> training cutoff (train: date_sold < cutoff)
    "F1": pd.Timestamp("2026-02-01"),             # test window [cutoff, cutoff + 4 months), by date_sold
    "F2": pd.Timestamp("2025-10-01"),
    "S25": pd.Timestamp("2025-06-01"),            # headline fold
}
# Reference numbers from the handoff document (final 3-seed rows): (MdAPE, MAPE) in %
HANDOFF_REF = {
    "off_market": {"F1": (7.01, 9.30), "F2": (6.78, 9.31), "S25": (7.04, 9.26)},
    "list_aware": {"F1": (3.78, 4.88), "F2": (3.68, 4.68), "S25": (3.91, 4.94)},
}
CONDO = {"Condo Apt", "Co-Op Apt", "Condo Townhouse"}

# --- Model/common.py additions ---
MONTHLY_FOLDS = {
    "2025-12": pd.Timestamp("2025-12-01"),
    "2026-01": pd.Timestamp("2026-01-01"),
    "2026-02": pd.Timestamp("2026-02-01"),
    "2026-03": pd.Timestamp("2026-03-01"),
    "2026-04": pd.Timestamp("2026-04-01"),
    "2026-05": pd.Timestamp("2026-05-01"),
}
MONTHLY_PREDICTIONS_CSV = REPORT_DIR / "monthly_predictions.csv"
MONTHLY_SUMMARY_CSV = REPORT_DIR / "monthly_summary.csv"
MONTHLY_METRICS_JSON = REPORT_DIR / "monthly_metrics.json"


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
# Remark flags: the 22 regexes with regex lookbehind fix for 'dated'
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
# Negative lookbehind prevents 'dated' from firing inside 'updated'
FIX_ORIGINAL_PATTERN = True
if FIX_ORIGINAL_PATTERN:
    KW["original"] = r"original|older|bygone|(?<!up)dated|nostalgic|vintage"
KW_COLS = [f"kw_{k}" for k in KW]

# Calibrated weights for condition score
COND_WEIGHTS = {"luxury": 0.070, "updated": 0.028, "stunning": 0.022, "renovated": 0.013, "turnkey": 0.009,
                "potential": -0.018, "investor": -0.026, "cozy": -0.035, "needs_work": -0.036, "asis": -0.043}

TEXT_FEATS = KW_COLS + ["kw_cond_score", "kw_n", "kw_cond_x_logsqft", "kw_cond_x_condo", "kw_lux_x_logsqft"]

# List-aware-only text options
LIST_TEXT_MODES = ("none", "a", "b", "ab")
LA_TEXT_RULES = {
    "urgency": r"motivated seller|bring all offers|quick closing|must sell|priced to sell",
    "fixer": r"as[- ]is|handyman|fixer|contractor special|needs (?:work|tlc)|renovator",
    "premium": r"sub[- ]zero|wolf appliance|custom millwork|marble|quartz|smart home",
    "scarcity": r"offer date|rarely available|one of a kind|architectural masterpiece",
    "tenant": r"tenanted|tenant occupied|rental income|currently leased",
}
LIST_TEXT_A_COLS = [f"la_text_{k}" for k in LA_TEXT_RULES]


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
    """house_area is either an integer or 'x-y'. Returns (midpoint sqft, is_range, width)."""
    t = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    m = t.str.extract(r"^(\d+)(?:\s*-\s*(\d+))?$")
    lo, hi = pd.to_numeric(m[0]), pd.to_numeric(m[1])
    sqft = pd.Series(np.where(hi.notna(), (lo + hi) / 2.0, lo), index=series.index, dtype=float)
    return sqft, hi.notna().astype(np.int8), (hi - lo)


def _cat(series: pd.Series) -> np.ndarray:
    return series.astype("object").where(series.notna(), "Missing").astype(str).values


def _key_part(series: pd.Series) -> pd.Series:
    return pd.Series(series.astype("object").where(series.notna(), "NA").astype(str).values, index=series.index)


# ------------------------------------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------------------------------------
def clean(df: pd.DataFrame, required) -> pd.DataFrame:
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

    ck = _key_part(o["latitude"].round(6)) + "|" + _key_part(o["longitude"].round(6))
    nocoord = o["latitude"].isna() | o["longitude"].isna()
    o["coord_key"] = np.where(nocoord, "NOCOORD_" + o["ml_num"], ck)
    o["unit_key"] = (o["coord_key"] + "|" + o["house_type_name"] + "|" + o["house_style"] + "|"
                     + _key_part(df["bedrooms"]) + "|" + _key_part(df["washrooms"]) + "|" + _key_part(df["house_area"]))
    assert not o["coord_key"].isna().any() and not o["unit_key"].isna().any(), "entity keys must never be missing"
    o["seg_id"] = o["house_type_name"].isin(CONDO).astype(np.int8)

    txt = df["remarks"].fillna("").astype(str)
    o["remarks"] = txt
    o["remarks_len"] = txt.str.len().astype(float)
    low = txt.str.lower()
    for k, pat in KW.items():
        o[f"kw_{k}"] = low.str.contains(pat, regex=True).astype(np.int8)
    o["kw_cond_score"] = sum(w * o[f"kw_{k}"] for k, w in COND_WEIGHTS.items()).astype(float)
    o["kw_n"] = o[KW_COLS].sum(axis=1).astype(float)
    logsq = np.log(np.where(o["sqft"] > 0, o["sqft"], np.nan))
    o["kw_cond_x_logsqft"] = (o["kw_cond_score"] * logsq).fillna(0.0)
    o["kw_cond_x_condo"] = o["kw_cond_score"] * o["seg_id"]
    o["kw_lux_x_logsqft"] = (o["kw_luxury"] * logsq).fillna(0.0)
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
    """Guardrail used before every fit/predict."""
    bad = [c for c in columns if c in POST_SALE_COLS or c in AVM_COLS or "avm" in c]
    if track == "off_market":
        bad += [c for c in columns if "list" in c]
    if bad:
        raise AssertionError(f"leakage guard ({track}): forbidden feature(s) {sorted(set(bad))}")