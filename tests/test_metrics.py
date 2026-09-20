import numpy as np
import pandas as pd
import pytest

from bt import metrics as m


def test_cagr_known_answer():
    r = pd.Series([0.0] * 251 + [0.10], index=pd.bdate_range("2020-01-01", periods=252))
    assert m.cagr(r) == pytest.approx(0.10)


def test_max_drawdown_known_answer():
    eq = pd.Series([1.0, 1.2, 0.9, 1.0, 1.5, 1.2])
    r = eq.pct_change().fillna(0.0)
    assert m.max_drawdown(r) == pytest.approx(0.9 / 1.2 - 1)


def test_drawdown_counts_loss_from_starting_capital():
    r = pd.Series([-0.10, 0.0, 0.0])
    assert m.max_drawdown(r) == pytest.approx(-0.10)


def test_sharpe_matches_manual_and_handles_zero_vol():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    manual = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert m.sharpe(r) == pytest.approx(manual)
    assert np.isnan(m.sharpe(pd.Series([0.001] * 100)))


def test_monthly_returns_compound():
    idx = pd.bdate_range("2021-01-01", "2021-02-26")
    r = pd.Series(0.001, index=idx)
    mr = m.monthly_returns(r)
    n_jan = (idx.month == 1).sum()
    assert mr.iloc[0] == pytest.approx(1.001 ** n_jan - 1)


def test_bootstrap_is_reproducible_and_centred():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    a = pd.Series(rng.normal(0.0006, 0.01, 1500), index=idx)
    b = pd.Series(rng.normal(0.0002, 0.01, 1500), index=idx)
    x = m.block_bootstrap_sharpe_diff(a, b, n_boot=300)
    y = m.block_bootstrap_sharpe_diff(a, b, n_boot=300)
    assert x == y
    assert x["ci_low"] < x["diff"] < x["ci_high"]
