"""Small, tested portfolio backtesting engine."""
from .engine import run_backtest, BacktestResult
from .metrics import summary_stats
from .strategies import dual_momentum_weights, fixed_weights, month_end_dates

__all__ = ["run_backtest", "BacktestResult", "summary_stats",
           "dual_momentum_weights", "fixed_weights", "month_end_dates"]
