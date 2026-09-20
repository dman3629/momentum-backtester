"""Signal generation. Every function only reads prices up to the signal date."""
from __future__ import annotations

import pandas as pd


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in the index.

    The final (possibly incomplete) month is dropped so a signal is never
    generated from a month that has not finished.
    """
    s = pd.Series(index, index=index)
    last = s.groupby([index.year, index.month]).max()
    return pd.DatetimeIndex(last.to_numpy()[:-1])


def dual_momentum_weights(prices: pd.DataFrame, risky: list[str], cash: str,
                          lookback: int = 12, skip: int = 1, top_n: int = 3) -> pd.DataFrame:
    """Dual momentum on month-end prices.

    Momentum at month-end t = P[t - skip] / P[t - skip - lookback] - 1.
    Relative leg: rank the risky assets, keep the top_n.
    Absolute leg: a kept asset only gets its slot if its momentum beats the
    cash asset's momentum over the same window, otherwise that slot goes to cash.
    Each slot is 1 / top_n.
    """
    if top_n < 1 or top_n > len(risky):
        raise ValueError("top_n must be between 1 and the number of risky assets")
    me = month_end_dates(prices.index)
    pm = prices.loc[me]
    mom = pm.shift(skip) / pm.shift(skip + lookback) - 1.0
    rows = {}
    for d in me:
        m = mom.loc[d]
        if m[risky + [cash]].isna().any():
            continue
        ranked = m[risky].sort_values(ascending=False)
        w = pd.Series(0.0, index=prices.columns)
        for a in ranked.index[:top_n]:
            if ranked[a] > m[cash]:
                w[a] += 1.0 / top_n
            else:
                w[cash] += 1.0 / top_n
        rows[d] = w
    return pd.DataFrame(rows).T


def fixed_weights(prices: pd.DataFrame, weights: dict[str, float],
                  start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Constant-mix benchmark rebalanced every month end."""
    me = month_end_dates(prices.index)
    if start is not None:
        me = me[me >= start]
    w = pd.Series(0.0, index=prices.columns)
    for k, v in weights.items():
        w[k] = v
    return pd.DataFrame({d: w for d in me}).T
