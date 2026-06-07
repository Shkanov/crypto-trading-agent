"""PIT weekly log-return series + autoregressive windowing.

Ports ForecastAGLT's `preprocess_data(scaling_strategy="average_prices")` and
`create_dataset`, cleaned and decoupled from yfinance/Streamlit:

  hourly klines  ->  daily closes  ->  N-day "average-price" blocks
                 ->  log-return between consecutive block means
                 ->  sliding window (X = L lagged returns, y = next return)

"average_prices" (ForecastAGLT's frozen-prod aggregation) means: bin into N-day
blocks, take the MEAN close within each block, then the log-return between
consecutive block means. The averaging smooths a single noisy weekly close; the
trade-off is a one-block MA lag, which is fine for a same-horizon AR forecast.

Everything here is causal: block k's return is realised at block k's end, and a
window predicting it uses only strictly-earlier returns. Blocks are anchored at
the series' last bar and walk backward, matching ForecastAGLT's now()-anchored
binning (for a fixed-end historical study the bin edges are fixed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    step_days: int = 7        # block width — weekly horizon (ForecastAGLT NSS=7)
    lookback: int = 15        # L lagged returns per sample (ForecastAGLT TSB=15)
    agg: str = "average"      # "average" = ForecastAGLT mean-price; "last" = c2c


def weekly_log_returns(close: pd.Series, step_days: int = 7,
                       agg: str = "average") -> pd.Series:
    """Hourly/any-frequency close → N-day-block log returns.

    Returns a chronological Series indexed by each block's END timestamp (the
    instant the block's return is realised). The most-recent block may be partial
    — harmless, since walk-forward only ever trains on blocks strictly before a
    test period.

    agg:
      "average" — ForecastAGLT's frozen "average_prices": block return is the
        log-ratio of consecutive block MEAN prices. WARNING: returns of a
        block-mean series are positively autocorrelated by construction (a
        Working/moving-average artifact) AND are not tradable (you can't transact
        at a week's average). Faithful to ForecastAGLT, but momentum measured on
        it is inflated. Use it only to reproduce their number.
      "last" — true weekly CLOSE-TO-CLOSE: block return is the log-ratio of the
        last daily close in each block. No smoothing artifact, tradable. This is
        the honest setting for any economic claim.
    """
    close = close.dropna().sort_index()
    if close.empty:
        return pd.Series(dtype=float)
    # Collapse to one close per UTC day first (ForecastAGLT operated on daily
    # bars), then aggregate daily closes within each N-day block.
    daily = close.resample("1D").last().dropna()
    if len(daily) < 2:
        return pd.Series(dtype=float)

    last = daily.index.max().normalize()
    blk = ((last - daily.index.normalize()) // pd.Timedelta(days=step_days)).astype(int)
    df = pd.DataFrame({"close": daily.to_numpy(), "blk": blk.to_numpy()},
                      index=daily.index)
    price_agg = "mean" if agg == "average" else "last"
    g = df.groupby("blk").agg(close=("close", price_agg),
                              end=("close", lambda s: s.index.max()))
    g = g.sort_values("end")                       # chronological (newest block last)
    ret = np.log(g["close"] / g["close"].shift(1))
    out = pd.Series(ret.to_numpy(), index=pd.DatetimeIndex(g["end"].to_numpy(),
                                                           name="t"))
    return out.dropna()


def window_samples(returns: pd.Series, lookback: int) -> pd.DataFrame:
    """Slide an L-wide window over a return series → AR samples.

    Sample i (for i >= L): features = returns[i-L : i] (oldest→newest), target =
    returns[i], timestamp = the target's index (when it is realised). Columns:
    f0..f{L-1}, y, t (target timestamp).
    """
    r = returns.to_numpy()
    idx = returns.index
    if len(r) <= lookback:
        return pd.DataFrame()
    rows = []
    ts = []
    for i in range(lookback, len(r)):
        rows.append(np.concatenate([r[i - lookback:i], [r[i]]]))
        ts.append(idx[i])
    cols = [f"f{j}" for j in range(lookback)] + ["y"]
    out = pd.DataFrame(rows, columns=cols)
    out["t"] = pd.DatetimeIndex(ts)
    return out


def build_pooled_samples(price_panel: dict[str, pd.DataFrame],
                         spec: WindowSpec) -> pd.DataFrame:
    """All coins' AR samples stacked into one tidy frame.

    Columns: coin, t (target timestamp), f0..f{L-1}, y. The pooled frame is what
    the walk-forward trains/scores on — pooling across coins is how a weekly
    horizon (≈100 blocks/coin over 2y) reaches an honest sample count.
    """
    frames = []
    for coin, df in price_panel.items():
        if "close" not in df.columns or df.empty:
            continue
        r = weekly_log_returns(df["close"], spec.step_days, spec.agg)
        w = window_samples(r, spec.lookback)
        if w.empty:
            continue
        w.insert(0, "coin", coin)
        frames.append(w)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)


def feature_cols(spec: WindowSpec) -> list[str]:
    return [f"f{j}" for j in range(spec.lookback)]
