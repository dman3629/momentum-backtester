"""Performance statistics. All inputs are daily simple returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return float("nan")
    total = float((1.0 + returns).prod())
    return total ** (TRADING_DAYS / n) - 1.0


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = returns - rf
    sd = ex.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = returns - rf
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    if downside < 1e-12:
        return float("nan")
    return float(ex.mean() / downside * np.sqrt(TRADING_DAYS))


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = (1.0 + returns).cumprod()
    peak = eq.cummax().clip(lower=1.0)   # starting capital counts as a peak
    return eq / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def monthly_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1.0


def summary_stats(returns: pd.Series, rf: pd.Series | float = 0.0,
                  turnover: pd.Series | None = None) -> dict:
    m = monthly_returns(returns)
    years = len(returns) / TRADING_DAYS
    mdd = max_drawdown(returns)
    c = cagr(returns)
    out = {
        "cagr": c,
        "vol": ann_vol(returns),
        "sharpe": sharpe(returns, rf),
        "sortino": sortino(returns, rf),
        "max_drawdown": mdd,
        "calmar": c / abs(mdd) if mdd < 0 else float("nan"),
        "worst_month": float(m.min()),
        "best_month": float(m.max()),
        "pct_positive_months": float((m > 0).mean()),
        "years": years,
    }
    if turnover is not None and years > 0:
        out["turnover_pa"] = float(turnover.sum() / years)
    return out


def block_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, rf: pd.Series | float = 0.0,
                                block: int = 21, n_boot: int = 2000, seed: int = 7) -> dict:
    """Circular block bootstrap of Sharpe(a) - Sharpe(b) on paired daily returns.

    Blocks keep the autocorrelation and the pairing between the two series.
    Returns the point estimate, a 95 percent interval and P(diff <= 0).
    """
    df = pd.concat([a - rf, b - rf], axis=1).dropna()
    x = df.to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    ann = np.sqrt(TRADING_DAYS)
    for j in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        ix = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        s = x[ix]
        sd = s.std(axis=0, ddof=1)
        diffs[j] = (s[:, 0].mean() / sd[0] - s[:, 1].mean() / sd[1]) * ann
    sd_full = x.std(axis=0, ddof=1)
    point = float((x[:, 0].mean() / sd_full[0] - x[:, 1].mean() / sd_full[1]) * ann)
    return {"diff": point, "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p_le_zero": float((diffs <= 0).mean())}
