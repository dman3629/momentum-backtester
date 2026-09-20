import numpy as np
import pandas as pd
import pytest

from bt.strategies import dual_momentum_weights, fixed_weights, month_end_dates


def synthetic(n=900, seed=2):
    idx = pd.bdate_range("2015-01-01", periods=n)
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "UP": 100 * np.exp(0.0010 * t),                 # steady winner
        "FLAT": 100 * np.exp(0.0001 * t),
        "DOWN": 100 * np.exp(-0.0008 * t),              # steady loser
        "NOISE": 100 * np.cumprod(1 + rng.normal(0, 0.002, n)),
        "CASH": 100 * np.exp(0.00012 * t),
    }, index=idx)


def test_month_end_dates_are_last_trading_days_and_drop_open_month():
    idx = pd.bdate_range("2022-01-03", "2022-04-14")
    me = month_end_dates(idx)
    assert list(me) == [pd.Timestamp("2022-01-31"), pd.Timestamp("2022-02-28"), pd.Timestamp("2022-03-31")]


def test_picks_the_winner_and_weights_sum_to_one():
    p = synthetic()
    w = dual_momentum_weights(p, ["UP", "FLAT", "DOWN", "NOISE"], "CASH", lookback=6, skip=1, top_n=1)
    assert (w["UP"] == 1.0).all()
    assert np.allclose(w.sum(axis=1), 1.0)


def test_absolute_filter_sends_losers_to_cash():
    p = synthetic()[["DOWN", "CASH"]].assign(DOWN2=lambda d: d["DOWN"] * 0.999)
    w = dual_momentum_weights(p, ["DOWN", "DOWN2"], "CASH", lookback=6, skip=0, top_n=2)
    assert (w["CASH"] == 1.0).all()


def test_first_signal_waits_for_full_lookback():
    p = synthetic()
    w = dual_momentum_weights(p, ["UP", "FLAT"], "CASH", lookback=12, skip=1, top_n=1)
    me = month_end_dates(p.index)
    assert w.index[0] == me[13]           # needs 12 + 1 earlier month-ends


def test_signals_do_not_use_future_data():
    p = synthetic()
    risky = ["UP", "FLAT", "DOWN", "NOISE"]
    full = dual_momentum_weights(p, risky, "CASH", lookback=6, skip=1, top_n=2)
    cut = p.index[500]
    early = dual_momentum_weights(p.loc[:cut], risky, "CASH", lookback=6, skip=1, top_n=2)
    pd.testing.assert_frame_equal(full.loc[early.index], early)


def test_fixed_weights_and_bad_top_n():
    p = synthetic()
    w = fixed_weights(p, {"UP": 0.6, "CASH": 0.4})
    assert np.allclose(w.sum(axis=1), 1.0) and (w["UP"] == 0.6).all()
    with pytest.raises(ValueError):
        dual_momentum_weights(p, ["UP"], "CASH", top_n=3)
