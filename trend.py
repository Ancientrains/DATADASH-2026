"""trend.py - segment price index (hedonic, re-estimated as-of every month), seasonality and damped-trend forecast.

Index definition
----------------
For each segment (free / condo) and each *vintage* v (= the first day of month v) a hedonic regression
    ln(sold_price) ~ mix controls + month dummies
is fitted on the sales with date_sold < start of month v (strictly completed months).  The month coefficients,
anchored at month 0 (= BASE_MONTH, index 0), are the vintage's log price index for months 0..v-1.
V[seg, v, m] therefore never contains information from month v or later (this is the "prior completed month"
fix, item E1).  Vintages of validation folds are a prefix of the full-data vintages, so one fit serves all folds.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

from common import BASE_MONTH, CACHE_DIR, TEST_MONTHS

MIN_VINTAGE = 12        # first vintage with a usable index (12 completed months)
MIN_MONTH_N = 20        # months with fewer sales (within a segment/vintage) are interpolated
RARE_LEVEL_N = 30       # categorical control levels with fewer rows are pooled into "other"
PHI = 0.8               # damping of the 6-month slope
RIDGE = 1e-6            # tiny ridge for the normal equations


def month_id(x):
    """Months since BASE_MONTH for a Timestamp / Series / DatetimeIndex."""
    if isinstance(x, pd.Timestamp):
        return (x.year - BASE_MONTH.year) * 12 + x.month - BASE_MONTH.month
    ix = pd.DatetimeIndex(x)
    return ((ix.year - BASE_MONTH.year) * 12 + ix.month - BASE_MONTH.month).to_numpy()


def cal_month(m):
    """Calendar month (0 = January) of month id(s) m."""
    return (np.asarray(m) + BASE_MONTH.month - 1) % 12


# ------------------------------------------------------------------------------------------------
# Mix controls
# ------------------------------------------------------------------------------------------------
def hedonic_controls(df: pd.DataFrame):
    """Sparse design of mix controls (no month terms) built from the rows passed in.  Only structural attributes
    are used, never prices.  For the as-of index it is called on exactly the rows a vintage may see."""
    num = pd.DataFrame({"log_sqft": np.log(df["sqft"]), "bed": df["bedrooms_n"], "bath": df["washrooms_n"],
                        "park": df["parking_n"], "garage": df["garage_n"], "log_lot": np.log1p(df["lot_area"])})
    blocks = []
    na = num.isna()
    na = na.loc[:, na.any()].astype(float).add_suffix("_na")
    num = num.fillna(num.groupby(df["house_type_name"].values).transform("median")).fillna(num.median()).fillna(0.0)
    num = (num - num.mean()) / num.std().replace(0, 1)
    blocks.append(sp.csr_matrix(pd.concat([num, na], axis=1).to_numpy(dtype=float)))
    for c in ("house_type_name", "house_style", "community"):
        s = df[c].astype(str)
        vc = s.value_counts()
        s = s.where(s.map(vc) >= RARE_LEVEL_N, "other")
        blocks.append(OneHotEncoder(sparse_output=True, handle_unknown="ignore").fit_transform(s.to_frame()).tocsr())
    return sp.hstack(blocks, format="csr")


def _month_dummies(m: np.ndarray, n_months: int) -> sp.csr_matrix:
    """Dummies for months 1..n_months-1 (month 0 is the base)."""
    rows = np.arange(len(m))
    ok = m >= 1
    return sp.csr_matrix((np.ones(ok.sum()), (rows[ok], m[ok] - 1)), shape=(len(m), n_months - 1))


def _solve(X: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
    A = (X.T @ X).toarray()
    A[np.diag_indices_from(A)] += RIDGE * np.mean(np.diag(A))
    return np.linalg.solve(A, X.T @ y)


def _month_effects(sub: pd.DataFrame, y: np.ndarray, m: np.ndarray, v: int) -> np.ndarray:
    """Log index for months 0..v-1 from the sales `sub` (all in months < v).  The mix controls are built from these
    rows only, so imputation medians, scaling and the set of categorical levels never see later sales."""
    cnt = np.bincount(m, minlength=v)
    ok = cnt[m] >= MIN_MONTH_N
    if cnt[0] < MIN_MONTH_N:
        raise ValueError("base month has too few sales in this segment")
    sub, y, m = sub[ok], y[ok], m[ok]
    D = hedonic_controls(sub)
    X = sp.hstack([sp.csr_matrix(np.ones((len(y), 1))), D, _month_dummies(m, v)], format="csr")
    beta = _solve(X, y)
    eff = np.r_[0.0, beta[-(v - 1):]]
    eff[cnt < MIN_MONTH_N] = np.nan
    return pd.Series(eff).interpolate(limit_direction="both").to_numpy()


# ------------------------------------------------------------------------------------------------
# Vintage index
# ------------------------------------------------------------------------------------------------
def fit_vintage_index(train: pd.DataFrame, last_vintage: int | None = None, verbose: bool = True) -> np.ndarray:
    """V[seg_id, v, m]: log index (anchored at month 0) for month m as known at the start of month v.
    NaN for v < MIN_VINTAGE and for m >= v."""
    m = month_id(train["date_sold"])
    if last_vintage is None:
        last_vintage = int(m.max()) + 1
    y = np.log(train["sold_price"].to_numpy(dtype=float))
    s = train["seg_id"].to_numpy()
    V = np.full((2, last_vintage + 1, last_vintage + 1), np.nan)
    for sg in (0, 1):
        rows = np.where(s == sg)[0]
        for v in range(MIN_VINTAGE, last_vintage + 1):
            r = rows[m[rows] < v]
            V[sg, v, :v] = _month_effects(train.iloc[r], y[r], m[r], v)
        if verbose:
            print(f"  index fitted for segment {'condo' if sg else 'free'}: vintages {MIN_VINTAGE}..{last_vintage}")
    return V


def get_vintage_index(train: pd.DataFrame, verbose: bool = True) -> np.ndarray:
    """Cached wrapper (cache key = size, price checksum and last sale month of the training file)."""
    key = hashlib.md5(f"{len(train)}|{train['sold_price'].sum():.0f}|{train['date_sold'].max()}|{MIN_VINTAGE}|{MIN_MONTH_N}".encode()).hexdigest()[:12]
    path = CACHE_DIR / f"vintage_index_{key}.npy"
    if path.exists():
        return np.load(path)
    CACHE_DIR.mkdir(exist_ok=True)
    V = fit_vintage_index(train, verbose=verbose)
    np.save(path, V)
    return V


def index_at(V: np.ndarray, seg_id, v: int, m):
    """Index values for rows with segment seg_id and month ids m, using vintage v (months >= v are carried back to the
    last completed month v-1)."""
    return V[np.asarray(seg_id), v, np.minimum(np.asarray(m), v - 1)]


def detrend_target(train: pd.DataFrame, V: np.ndarray, v: int) -> np.ndarray:
    """ln(price) - Index_seg(sale month) with the vintage-v index (v = month id of the fold cutoff)."""
    return (np.log(train["sold_price"].to_numpy(dtype=float))
            - index_at(V, train["seg_id"].to_numpy(), v, month_id(train["date_sold"])))


# ------------------------------------------------------------------------------------------------
# Seasonality and damped-trend forecast
# ------------------------------------------------------------------------------------------------
def seasonal_factors(index_vec: np.ndarray) -> np.ndarray:
    """Classical log-additive seasonal factors (12, January first, sum 0) from a 2x12 centred moving average.
    All complete months of index_vec are used; fewer than 36 months -> no seasonality."""
    x = np.asarray(index_vec, dtype=float)
    n = len(x)
    if n < 36:
        return np.zeros(12)
    w = np.r_[0.5, np.ones(11), 0.5] / 12.0
    ma = np.convolve(x, w, mode="valid")                  # centred on t = 6 .. n-7
    t = np.arange(6, n - 6)
    dev = x[t] - ma
    S = np.array([dev[cal_month(t) == k].mean() if (cal_month(t) == k).any() else 0.0 for k in range(12)])
    return S - S.mean()


def forecast_index(index_vec: np.ndarray, horizon: int = TEST_MONTHS, phi: float = PHI) -> np.ndarray:
    """Forecast of the log index for the `horizon` months after the last month of index_vec (month T).
    de-seasonalised level = mean of the last 3 months (centred on T-1); slope = OLS slope over the last 6
    de-seasonalised months; damped by phi**j; seasonal factors added back."""
    x = np.asarray(index_vec, dtype=float)
    T = len(x) - 1
    S = seasonal_factors(x)
    D = x - S[cal_month(np.arange(len(x)))]
    level = D[-3:].mean()
    slope = np.polyfit(np.arange(6), D[-6:], 1)[0]
    out = []
    for h in range(1, horizon + 1):
        damp = sum(phi ** j for j in range(1, h + 2))      # level sits at T-1, so h+1 steps to reach T+h
        out.append(level + slope * damp + S[cal_month(T + h)])
    return np.array(out)


def addback(V: np.ndarray, cutoff: pd.Timestamp, horizon: int = TEST_MONTHS) -> np.ndarray:
    """Average forecast log index over the 4-month test window, one value per segment: array [free, condo]."""
    v = month_id(cutoff)
    return np.array([forecast_index(V[sg, v, :v], horizon).mean() for sg in (0, 1)])


def realised_window_index(V: np.ndarray, cutoff: pd.Timestamp, horizon: int = TEST_MONTHS):
    """Realised mean index over the test window (needs a later vintage); [free, condo] or NaN if not yet observed."""
    v0 = month_id(cutoff)
    vl = V.shape[1] - 1
    if v0 + horizon > vl:
        return np.array([np.nan, np.nan])
    return np.array([V[sg, vl, v0:v0 + horizon].mean() for sg in (0, 1)])\
    