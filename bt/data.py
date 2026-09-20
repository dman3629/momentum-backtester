"""Price download with an on-disk cache so results are reproducible."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prices(tickers: list[str], start: str, end: str | None, cache: str | Path) -> pd.DataFrame:
    cache = Path(cache)
    if cache.exists():
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers).issubset(px.columns):
            return px[tickers]
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = raw[tickers].dropna(how="any")     # common history only, no forward fill
    cache.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(cache)
    return px
