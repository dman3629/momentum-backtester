"""Walk-forward test of dual momentum across asset-class ETFs.

Protocol, fixed before looking at out-of-sample numbers:
  1. Universe is asset-class ETFs, not single stocks, so there is no survivorship bias
     from picking today's index members.
  2. In-sample 2007-2016: grid over lookback, skip and top_n. Pick the best Sharpe.
  3. Freeze those parameters. Out-of-sample 2017 onward is run once.
  4. Every OOS grid cell is also reported so the chosen cell can be judged against
     its neighbours, and a block bootstrap puts an interval on the Sharpe gap vs 60/40.
"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bt.data import load_prices
from bt.engine import run_backtest
from bt.metrics import block_bootstrap_sharpe_diff, drawdown_series, summary_stats
from bt.strategies import dual_momentum_weights, fixed_weights

RISKY = ["SPY", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "GLD", "DBC"]
CASH = "SHY"
COST_BPS = 10.0
LAG = 1
IS_START, IS_END = "2007-06-01", "2016-12-31"
OOS_START = "2017-01-01"
OUT = Path("results")


def sliced(res, start, end=None):
    r = res.returns.loc[start:end]
    t = res.turnover.loc[start:end]
    return r, t


def main():
    OUT.mkdir(exist_ok=True)
    px = load_prices(RISKY + [CASH], "2006-02-01", None, "data/prices.csv")
    rf = px[CASH].pct_change().fillna(0.0)
    print(f"prices {px.index[0].date()} to {px.index[-1].date()}, {len(px)} days, {px.shape[1]} assets")

    grid = list(itertools.product([3, 6, 9, 12], [0, 1], [2, 3, 4]))
    rows = []
    runs = {}
    for lb, sk, n in grid:
        tw = dual_momentum_weights(px, RISKY, CASH, lookback=lb, skip=sk, top_n=n)
        res = run_backtest(px, tw, cost_bps=COST_BPS, lag=LAG)
        runs[(lb, sk, n)] = res
        r_is, t_is = sliced(res, IS_START, IS_END)
        r_oos, t_oos = sliced(res, OOS_START)
        s_is = summary_stats(r_is, rf.loc[r_is.index], t_is)
        s_oos = summary_stats(r_oos, rf.loc[r_oos.index], t_oos)
        rows.append({"lookback": lb, "skip": sk, "top_n": n,
                     "is_sharpe": s_is["sharpe"], "is_cagr": s_is["cagr"], "is_mdd": s_is["max_drawdown"],
                     "oos_sharpe": s_oos["sharpe"], "oos_cagr": s_oos["cagr"], "oos_mdd": s_oos["max_drawdown"]})
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "grid.csv", index=False)
    best = g.sort_values("is_sharpe", ascending=False).iloc[0]
    key = (int(best.lookback), int(best.skip), int(best.top_n))
    print("chosen in-sample:", key)
    strat = runs[key]

    first_sig = dual_momentum_weights(px, RISKY, CASH, lookback=12, skip=1, top_n=3).index[0]
    bench = {
        "SPY buy and hold": fixed_weights(px, {"SPY": 1.0}, first_sig),
        "60/40 SPY/IEF": fixed_weights(px, {"SPY": 0.6, "IEF": 0.4}, first_sig),
        "Equal weight universe": fixed_weights(px, {a: 1 / len(RISKY) for a in RISKY}, first_sig),
    }
    bres = {k: run_backtest(px, v, cost_bps=COST_BPS, lag=LAG) for k, v in bench.items()}

    table = {}
    for period, (a, b) in {"in_sample": (IS_START, IS_END), "out_of_sample": (OOS_START, None)}.items():
        table[period] = {}
        r, t = sliced(strat, a, b)
        table[period]["Dual momentum"] = summary_stats(r, rf.loc[r.index], t)
        for name, res in bres.items():
            r, t = sliced(res, a, b)
            table[period][name] = summary_stats(r, rf.loc[r.index], t)

    r_s, _ = sliced(strat, OOS_START)
    r_b, _ = sliced(bres["60/40 SPY/IEF"], OOS_START)
    boot = block_bootstrap_sharpe_diff(r_s, r_b, rf.loc[r_s.index])

    # cost and lag sensitivity on the frozen parameters, OOS only
    tw = dual_momentum_weights(px, RISKY, CASH, lookback=key[0], skip=key[1], top_n=key[2])
    sens = []
    for c, l in itertools.product([0, 10, 25, 50], [0, 1, 2]):
        rr, tt = sliced(run_backtest(px, tw, cost_bps=c, lag=l), OOS_START)
        s = summary_stats(rr, rf.loc[rr.index], tt)
        sens.append({"cost_bps": c, "lag": l, "oos_sharpe": s["sharpe"], "oos_cagr": s["cagr"]})
    pd.DataFrame(sens).to_csv(OUT / "cost_lag_sensitivity.csv", index=False)

    oos_rank = int((g["oos_sharpe"] > float(best.oos_sharpe)).sum()) + 1
    results = {
        "data": {"start": str(px.index[0].date()), "end": str(px.index[-1].date()), "days": len(px),
                 "universe": RISKY, "cash": CASH},
        "protocol": {"in_sample": [IS_START, IS_END], "out_of_sample_start": OOS_START,
                     "cost_bps": COST_BPS, "lag_days": LAG, "grid_cells": len(grid)},
        "chosen_params": {"lookback": key[0], "skip": key[1], "top_n": key[2]},
        "chosen_oos_rank_in_grid": oos_rank,
        "is_vs_oos_sharpe_rank_correlation": float(g.is_sharpe.rank().corr(g.oos_sharpe.rank())),
        "grid_oos_sharpe": {"min": float(g.oos_sharpe.min()), "median": float(g.oos_sharpe.median()),
                            "max": float(g.oos_sharpe.max())},
        "stats": table,
        "bootstrap_sharpe_diff_vs_6040_oos": boot,
        "cost_lag_sensitivity": sens,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # ---- charts
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    start = first_sig
    series = {"Dual momentum": strat.returns, **{k: v.returns for k, v in bres.items()}}
    colors = {"Dual momentum": "#0b3d91", "SPY buy and hold": "#999999",
              "60/40 SPY/IEF": "#d9822b", "Equal weight universe": "#5aa469"}
    for name, r in series.items():
        r = r.loc[start:]
        ax[0].plot((1 + r).cumprod(), label=name, color=colors[name], lw=1.6 if name == "Dual momentum" else 1.0)
    ax[0].set_yscale("log"); ax[0].axvline(pd.Timestamp(OOS_START), color="k", ls="--", lw=0.8)
    ax[0].text(pd.Timestamp(OOS_START), ax[0].get_ylim()[1] * 0.9, "  out-of-sample starts", fontsize=8, va="top")
    ax[0].set_title(f"Growth of 1.0, net of {COST_BPS:.0f} bps costs, signals traded one day late (log scale)")
    ax[0].legend(frameon=False, fontsize=8)
    for name in ["Dual momentum", "SPY buy and hold", "60/40 SPY/IEF"]:
        ax[1].plot(drawdown_series(series[name].loc[start:]), color=colors[name], lw=1.0)
    ax[1].set_title("Drawdown"); ax[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout(); fig.savefig(OUT / "equity_drawdown.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(g.is_sharpe, g.oos_sharpe, color="#999999", s=28)
    ax.scatter([best.is_sharpe], [best.oos_sharpe], color="#0b3d91", s=70, label="chosen in-sample")
    ax.axhline(table["out_of_sample"]["60/40 SPY/IEF"]["sharpe"], color="#d9822b", ls="--", lw=1, label="60/40 OOS Sharpe")
    ax.set_xlabel("in-sample Sharpe 2007-2016"); ax.set_ylabel("out-of-sample Sharpe 2017 onward")
    ax.set_title("All 24 parameter sets: does in-sample rank predict out-of-sample?")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "grid_is_vs_oos.png", dpi=160); plt.close(fig)

    w = strat.weights.loc[start:].resample("ME").last()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.stackplot(w.index, w.T.to_numpy(), labels=w.columns, linewidth=0)
    ax.set_ylim(0, 1); ax.set_title("Dual momentum: month-end holdings")
    ax.legend(ncol=6, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(OUT / "holdings.png", dpi=160); plt.close(fig)

    print(json.dumps({k: results[k] for k in ["chosen_params", "chosen_oos_rank_in_grid", "is_vs_oos_sharpe_rank_correlation", "grid_oos_sharpe",
                                               "bootstrap_sharpe_diff_vs_6040_oos"]}, indent=1))
    for period in table:
        print(period)
        print(pd.DataFrame(table[period]).T[["cagr", "vol", "sharpe", "max_drawdown", "calmar", "turnover_pa"]].round(3))


if __name__ == "__main__":
    main()
