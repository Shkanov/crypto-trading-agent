"""Native price-return predictor — ForecastAGLT capability ported onto Binance perps.

ForecastAGLT's forecasters (LSTM/GMDH/SARIMA) are all *univariate autoregressive*
on log-returns: each takes a coin's past N (weekly-aggregated) log-returns and
predicts the next one. A direct directional probe of that capability (their own
`_directional_accuracy`, weekly horizon, 2026-06-01) showed a FAINT but non-zero
pulse (LSTM 0.564, GMDH 0.551, pooled 0.526) on liquid majors — the first
non-zero signal in the whole funding investigation. But that read was a single
train/test split on yfinance majors, with two structural problems for our use:
the universe was near-disjoint from the funding book, and the yfinance data layer
silently resolves funding alts to dead/wrong namesakes.

This package re-tests that capability HONESTLY:
  * data-source-agnostic: fed clean Binance perp klines (fixes identity),
  * model-robust: the nonlinear-AR capability is exercised by TWO learners
    (lightgbm GBM + sklearn MLP) — if the pulse is real it survives across model
    classes; if it only lived in one Keras config it was overfit,
  * validated WALK-FORWARD (expanding-origin, calendar-anchored, embargoed),
    pooled across coins for honest n, scored after costs.

Two readouts:
  (1) standalone — does the ~55% weekly directional survive proper walk-forward
      n and beat costs?
  (2) overlay — does conditioning a Δfunding leg on predictor sign-agreement
      improve that leg's after-cost economics, on the coins with measurable skill?

Offline research only. Read-only public data. NOT wired into the live runtime.
"""
