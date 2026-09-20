import numpy as np
import pandas as pd
import pytest

from bt.engine import run_backtest


def days(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def test_buy_and_hold_matches_price_ratio():
    idx = days(50)
    rng = np.random.default_rng(0)
    p = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 50))}, index=idx)
    tw = pd.DataFrame({"A": [1.0]}, index=[idx[0]])
    res = run_backtest(p, tw, cost_bps=0, lag=1)
    # signal day 0, trade at close of day 1, so equity tracks price from day 1
    expected = p["A"].iloc[-1] / p["A"].iloc[1]
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_lag_means_jump_before_execution_is_not_captured():
    idx = days(6)
    p = pd.DataFrame({"A": [100, 100, 150, 150, 150, 150.0]}, index=idx)
    tw = pd.DataFrame({"A": [1.0]}, index=[idx[1]])          # signal at close of day 1
    lagged = run_backtest(p, tw, cost_bps=0, lag=1)          # trades at close of day 2
    assert lagged.equity.iloc[-1] == pytest.approx(1.0)      # missed the jump, correctly
    same_close = run_backtest(p, tw, cost_bps=0, lag=0)      # trades at close of day 1
    assert same_close.equity.iloc[-1] == pytest.approx(1.5)


def test_cost_of_full_switch_is_two_sided():
    idx = days(10)
    p = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)     # flat prices isolate costs
    tw = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[idx[0], idx[4]])
    res = run_backtest(p, tw, cost_bps=10, lag=1)
    assert res.turnover.tolist() == pytest.approx([1.0, 2.0])
    assert res.equity.iloc[-1] == pytest.approx((1 - 0.001) * (1 - 0.002), rel=1e-12)


def test_weights_drift_between_rebalances():
    idx = days(4)
    p = pd.DataFrame({"A": [100, 100, 200, 200.0], "B": [100, 100, 100, 100.0]}, index=idx)
    tw = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[idx[0]])
    res = run_backtest(p, tw, cost_bps=0, lag=1)
    assert res.equity.iloc[-1] == pytest.approx(1.5)
    assert res.weights["A"].iloc[-1] == pytest.approx(2 / 3)   # drifted, not reset to 0.5


def test_unallocated_weight_is_cash_at_zero():
    idx = days(5)
    p = pd.DataFrame({"A": [100, 100, 110, 121, 121.0]}, index=idx)
    tw = pd.DataFrame({"A": [0.5]}, index=[idx[0]])
    res = run_backtest(p, tw, cost_bps=0, lag=1)
    assert res.equity.iloc[-1] == pytest.approx(0.5 + 0.5 * 1.21)


def test_no_lookahead_future_prices_do_not_change_the_past():
    idx = days(120)
    rng = np.random.default_rng(1)
    p = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (120, 2)), axis=0),
                     index=idx, columns=["A", "B"])
    tw = pd.DataFrame({"A": [1.0, 0.0, 0.5], "B": [0.0, 1.0, 0.5]}, index=[idx[5], idx[40], idx[80]])
    base = run_backtest(p, tw)
    corrupted = p.copy()
    corrupted.iloc[60:] *= 3.7                                   # rewrite the future
    alt = run_backtest(corrupted, tw)
    pd.testing.assert_series_equal(base.equity.iloc[:60], alt.equity.iloc[:60])


def test_rejects_bad_inputs():
    idx = days(5)
    p = pd.DataFrame({"A": [1, 2, 3, 4, 5.0]}, index=idx)
    with pytest.raises(ValueError):
        run_backtest(p, pd.DataFrame({"A": [1.5]}, index=[idx[0]]))          # leverage
    with pytest.raises(ValueError):
        run_backtest(p, pd.DataFrame({"A": [-0.5]}, index=[idx[0]]))         # short
    with pytest.raises(ValueError):
        run_backtest(p, pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("1999-01-01")]))
    bad = p.copy(); bad.iloc[2, 0] = np.nan
    with pytest.raises(ValueError):
        run_backtest(bad, pd.DataFrame({"A": [1.0]}, index=[idx[0]]))
