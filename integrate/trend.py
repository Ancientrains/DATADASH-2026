"""Monthly hedonic-ish price index per segment, built ONLY from the training rows passed in."""
import numpy as np, pandas as pd
CONDO = {'Condo Apt','Co-Op Apt','Condo Townhouse'}
def seg(df): return np.where(df.house_type_name.isin(CONDO), 'condo', 'free')
def mperiod(d): return d.dt.year*12 + d.dt.month - 1  # integer month id

def build_index(tr, iters=4):
    d = pd.DataFrame({'lp': np.log(tr.sold_price.values), 'm': mperiod(tr.date_sold).values, 's': seg(tr),
                      'cell': (tr.house_type_name.astype(str)+'|'+tr.community.astype(str)+'|'+tr.bedrooms.astype(str)+'|'+tr.washrooms.astype(str)).values})
    d['me'] = 0.0
    for _ in range(iters):  # alternating medians: lp = cell_fe + month_fe(seg)
        d['ce'] = (d.lp - d.me).groupby(d.cell).transform('median')
        me = (d.lp - d.ce).groupby([d.s, d.m]).median(); d['me'] = me.loc[list(zip(d.s, d.m))].values
    I = me.unstack(0)                     # index: month id, cols: condo/free
    I = I - I.iloc[-12:].mean()           # normalise
    return I

def seasonal(I):
    S = {}
    for s in I.columns:
        x = I[s]; ma = x.rolling(12, center=True).mean(); r = (x - ma).dropna()
        f = r.groupby(r.index % 12).mean().reindex(range(12)).fillna(0.0); S[s] = f - f.mean()
    return pd.DataFrame(S)

def forecast(I, months, method):
    """months: array of integer month ids (> last). returns DataFrame-like dict seg -> array"""
    T = I.index.max(); S = seasonal(I); out = {}
    months = np.asarray(months)
    for s in I.columns:
        x = I[s]
        if method == 'lvcf': f = np.full(len(months), x.loc[T])
        elif method == 'last3': f = np.full(len(months), x.loc[T-2:T].mean())
        elif method == 'yoy':
            g = np.mean([x.loc[T-k] - x.loc[T-k-12] for k in range(3)])
            f = np.array([x.loc[m-12] if m-12 <= T else x.loc[m-24] + g for m in months]) + g
        elif method.startswith('seas'):
            ds = x - S[s].reindex(x.index % 12).values
            lvl = ds.loc[T-2:T].mean()
            slope = 0.0
            if method == 'seas_trend':
                yy = ds.loc[T-5:T].values; slope = np.polyfit(np.arange(6), yy, 1)[0]
            h = months - (T - 1)        # horizon from centre of last-3 window
            damp = np.array([sum(0.8**j for j in range(1, hh+1)) for hh in h])
            f = lvl + slope*damp + S[s].reindex(months % 12).values
        out[s] = f
    return out
