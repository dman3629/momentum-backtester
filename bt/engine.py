"""Backtest engine.

Timing convention (the part most backtests get wrong):

* A signal is computed with prices up to and including the close of day t.
* The trade happens at the close of day t + lag (lag=1 by default), because
  you cannot observe a close and trade at that same close.
* The new weights earn returns from day t + lag + 1 onward.
* Between rebalances weights drift with prices. They are not reset daily.
* Costs are charged on traded notional: cost = turnover * cost_bps / 10,000,
  where turnover = sum(|target weight - drifted weight|). A full switch from
  one asset to another is turnover 2.0 (sell 100 percent, buy 100 percent).
* Any weight not allocated sits in cash earning zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series          # portfolio value, starts at 1.0
    returns: pd.Series         # daily net returns
    weights: pd.DataFrame      # end-of-day weights actually held
    turnover: pd.Series        # turnover on each execution date
    costs: pd.Series           # cost paid (fraction of equity) on each execution date


def run_backtest(prices: pd.DataFrame, target_weights: pd.DataFrame,
                 cost_bps: float = 10.0, lag: int = 1) -> BacktestResult:
    """Simulate a portfolio.

    prices          daily adjusted closes, one column per asset, no gaps
    target_weights  rows indexed by SIGNAL date (must be in prices.index),
                    columns a subset of prices.columns, each row sums to <= 1
    cost_bps        cost per unit of traded notional, in basis points
    lag             trading days between signal and execution (>= 0)
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if prices.isna().any().any():
        raise ValueError("prices contain NaN; clean the data first")
    if not target_weights.index.isin(prices.index).all():
        raise ValueError("every signal date must be a trading day in prices")
    if (target_weights < -1e-12).any().any():
        raise ValueError("long-only engine: negative weights not supported")
    if (target_weights.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("target weights sum to more than 1 (no leverage)")

    assets = list(prices.columns)
    tw = target_weights.reindex(columns=assets).fillna(0.0)
    idx = prices.index
    pos = {d: i for i, d in enumerate(idx)}

    # map execution day index -> target vector
    exec_map: dict[int, np.ndarray] = {}
    for sig_date, row in tw.iterrows():
        e = pos[sig_date] + lag
        if e < len(idx):
            exec_map[e] = row.to_numpy(dtype=float)

    rets = prices.pct_change().fillna(0.0).to_numpy()
    n, k = rets.shape
    w = np.zeros(k)
    equity = np.ones(n)
    net = np.zeros(n)
    held = np.zeros((n, k))
    turn = {}
    cost = {}
    value = 1.0
    for i in range(n):
        # 1. earn today's return on weights carried in from yesterday
        port_ret = float(w @ rets[i])
        if w.any():
            w = w * (1.0 + rets[i]) / (1.0 + port_ret)   # drift
        # 2. rebalance at today's close if an order is due
        c = 0.0
        if i in exec_map:
            target = exec_map[i]
            t = float(np.abs(target - w).sum())
            c = t * cost_bps / 1e4
            turn[idx[i]] = t
            cost[idx[i]] = c
            w = target.copy()
        day_ret = (1.0 + port_ret) * (1.0 - c) - 1.0
        value *= 1.0 + day_ret
        net[i] = day_ret
        equity[i] = value
        held[i] = w
    return BacktestResult(
        equity=pd.Series(equity, index=idx, name="equity"),
        returns=pd.Series(net, index=idx, name="ret"),
        weights=pd.DataFrame(held, index=idx, columns=assets),
        turnover=pd.Series(turn, dtype=float, name="turnover"),
        costs=pd.Series(cost, dtype=float, name="cost"),
    )
