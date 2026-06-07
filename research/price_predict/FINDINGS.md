# Native price predictor — findings (2026-06-06)

**Verdict: NEGATIVE on both readouts.** The ForecastAGLT return-prediction
capability, ported honestly onto tradable Binance perp close-to-close weekly
returns and validated walk-forward, has **no genuine directional skill**, and as
a Δfunding side-filter it produces **no significant improvement**. The "faint
~55% weekly directional pulse" previously flagged was two stacked artifacts.

## What was built

`research/price_predict/` — the ForecastAGLT capability (univariate nonlinear AR
on weekly log-returns) decoupled from yfinance/Streamlit and re-tested cleanly:

- **series.py** — Binance hourly klines → daily closes → N-day blocks → weekly
  log-returns → sliding AR windows (X = 15 lagged returns, y = next return).
  `agg="average"` reproduces ForecastAGLT's block-mean aggregation;
  `agg="last"` is true close-to-close (the honest, tradable setting).
- **models.py** — the nonlinear-AR capability across two model classes:
  `gbm` (lightgbm trees) + `mlp` (sklearn neural, the LSTM's stand-in) — gmdh's
  C++ binding segfaults under numpy-2/py-3.13, so it was dropped. Baselines:
  `momentum`, `always_long`, `always_short`.
- **walkforward.py** — expanding-origin, calendar-anchored, embargoed walk-forward
  pooled across coins (~19k OOS predictions). Forward-only, PIT, train-fit scaling.
- **evaluate.py** — directional accuracy + after-cost economics + per-coin +
  temporal halves + **de-drift** diagnostic.
- **overlay.py** — gate each Δfunding leg on predictor sign-agreement; raw vs
  filtered per-leg economics on the SAME OOS legs.
- **run.py** — driver for both readouts. `uv run --extra research python -m research.price_predict.run`

Universe: 50 liquid perps (majors + established alts) over 4y, plus the validated
funding universe for the overlay. Cost 10 bps/leg.

## Readout 1 — standalone weekly directional (the artifact, exposed)

| model | agg="average" (ForecastAGLT) | agg="last" (tradable c2c) |
|---|---|---|
| momentum | 0.576  (z 10.6, +240bps, t 15.2) | 0.492  (z −1.1, +40bps, t 2.0) |
| gbm | 0.559  (z 8.2, +164bps, t 10.3) | **0.503  (z 0.4, +0.5bps, t 0.03)** |
| mlp | 0.549  (z 6.8, +126bps, t 7.8) | **0.540  (z 5.6, +82bps, t 4.2)** |
| always_long | 0.436 | 0.439 |

**Artifact 1 — the average-price moving-average effect.** Log-returns of block
*mean* prices are positively autocorrelated by construction (Working/MA effect)
and are not tradable (you can't transact at a week's average). On the block-mean
series everything looks skilled (momentum 0.576, t 15); switch to tradable
close-to-close and momentum/gbm collapse to a coin flip. ForecastAGLT's measured
0.564 directional accuracy lived entirely on this aggregation.

**Artifact 2 — mlp's residual 0.540 is market-drift harvesting, not skill.**
On c2c, mlp alone still shows 0.540 (z 5.6), stable across both time-halves
(z 5.2 / 2.7) and broad (81% of 42 coins > 0.50). It looks real — but:

- `always_short` (a trivial constant rule) scores **0.561 / +91bps / t 4.62** —
  it BEATS mlp on every metric. The alt universe *fell* in ~56% of weeks
  (always_long 0.439), so any net-short tilt scores >0.50 with zero forecasting.
  mlp predicts "down" 46% of the time — a noisy short tilt, worse than pure short.
- **De-drift test** (score on each week's cross-sectionally demeaned returns —
  "can it pick which coin beats the market?"): mlp → **0.485, −4.7bps, t −0.34.**
  The entire pulse was the common 2024-26 downward drift; cross-sectional skill
  is nil. Regime-dependent (flips when alts rise), not a tradable edge.

## Readout 2 — predictor as Δfunding side-filter

Honest (c2c) predictor, on the ~263/330 funding legs that have an OOS prediction
(canonical run):

| book | n | net bps/leg | t | win% |
|---|---|---|---|---|
| raw funding legs | 263 | +154.9 | 1.17 | 0.53 |
| gbm sign-agreement | 124 | +204.6 | 1.20 | 0.49 |
| mlp sign-agreement | 134 | +331.2 | 1.73 | 0.52 |

The raw funding leg is already noise — and *unstable*: across two runs (funding
cache refetched) the raw book swung +84.6bps (t 0.59) → +154.9bps (t 1.17) with
no code change, which is itself the tell that it's a tiny-n noise estimate.
Filtering nudges bps up but **t stays < 2 throughout**, win% does not improve,
and the lift is inconsistent across models and runs. A predictor with no real
directional skill (Readout 1) cannot rescue a noise primary — no robust lift.

## Why this is the right conclusion

This closes the price-side thread of the funding investigation. Three
independent "no-edge" verdicts now stand: (1) Δfunding price-leg ≈ 0
(t 0.12, [[project-ml-meta-labeling]]); (2) meta-labeling can't manufacture edge
from a flat primary; (3) the ForecastAGLT capability has no tradable directional
skill once the average-price artifact and market drift are removed. The
falsification-first lesson holds: the only "non-zero" signal in the whole hunt
was an aggregation artifact, caught by switching to tradable returns and
benchmarking against the drift baselines + de-drift.
