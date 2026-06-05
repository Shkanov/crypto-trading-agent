# Meta-labeling research — design

**Goal.** Learn a non-linear filter that, given a *base strategy's* trade signal,
predicts whether that specific trade will be profitable after costs — then trade
only the signals the filter believes in (and optionally size by its confidence).
This raises precision (cuts false positives) without ever touching the base
strategy's side/timing decision. López de Prado, *Advances in Financial Machine
Learning* (AFML), ch. 3.

**Why meta-labeling first (vs. predicting returns directly).** The base signals
(Δfunding, mean-rev) are already CPCV-validated to have *some* edge. A meta-model
that only decides take/skip on top of them has a tiny hypothesis space relative to
predicting raw returns, so it is far harder to overfit and far easier to falsify.
It can only improve on the primary by abstaining — a strictly bounded claim.

**The non-linearity / interaction thesis.** The meta-model's features span
(base-signal strength × instrument × regime): cross-sectional funding dispersion,
BTC realized-vol & trend, correlation regime, basis, OI, time-of-day. The bet is
that *when* a base signal pays is conditional on these in ways a linear allocator
can't express (e.g. "Δfunding pays only when dispersion is high and BTC vol is
low"). Gradient-boosted trees capture those interactions natively.

---

## The only thing that matters: no leakage

Every control is enforced in code + asserted in tests, not by intention.

| Leak vector | Control |
|---|---|
| Look-ahead in features | Each feature uses bars with `close_time ≤ t_decision`; entry price = `close[t_decision]`; barriers/labels read only bars **after** `t_decision`. |
| Target leakage | Label window `[t0, t1]` is strictly forward of `t_decision`. One decision cut per sample. |
| Overlapping labels (forward-horizon returns overlap) | **Average-uniqueness sample weights** (AFML 4.2) downweight concurrent labels. |
| CV leakage from autocorrelation | **Purged + embargoed CPCV** keyed on each event's label span `[t0,t1]` (AFML 7). Reuse `src/services/cpcv.py`; extend purge to event spans. |
| Scaler / feature-selection / hyperparam leakage | Fit **inside each train fold only**. Never `.fit` on full data. |
| Survivorship | PIT universe + `binance_delistings.json` (already in repo). |
| Non-stationarity | Fractional differentiation where a feature needs stationarity but must keep memory (AFML 5). |
| Selection bias (trying N configs) | **PBO** (already in repo) + **Deflated Sharpe** (already in repo). Report the haircut, never the raw best. |
| Costs | Almgren–Chriss `costs.py` applied to the label (a trade is "win" only if positive **after** round-trip cost) and to OOS PnL. |

**Hard rule:** any result is reported as the *purged-CPCV OOS distribution* with
its PBO and deflated Sharpe — never a single train/test split, never in-sample.

---

## Pipeline (modules under `research/ml_meta/`)

1. `data.py` — PIT panel builder. Cache klines (1h floor) + funding + (≤30d) OI
   to parquet, keyed by PIT-universe membership. Deterministic, re-runnable.
2. `labeling.py` *(this commit)* — triple-barrier meta-labels (cost-aware binary:
   did the primary's bet profit?) + average-uniqueness weights. Pure, tested.
3. `events.py` — replay a base strategy PIT to emit `(t_decision, symbol, side)`.
4. `features.py` — PIT-safe feature matrix at each event (signal strength + regime
   + cross-instrument). Explicit per-feature lag.
5. `cv.py` — purged+embargoed CPCV over event spans (wraps `services/cpcv.py`).
6. `model.py` — LightGBM binary classifier; scaler/selection fit per-fold.
7. `evaluate.py` — meta-filtered vs raw-primary OOS Sharpe; PBO; deflated Sharpe;
   per-fold turnover & costs.

## Success criteria (falsification-first)
The meta-filtered strategy must, on **purged-CPCV OOS**:
- beat the **raw primary** OOS Sharpe (the only honest baseline), AND
- PBO < 0.5, AND
- positive **deflated** Sharpe (after the multiple-trials haircut), AND
- do it with fewer trades but higher per-trade edge (precision gain), AND
- survive on ≥2 base strategies / instrument groups (not a single lucky cell).

If it doesn't clear all five, the honest output is "no edge" — and we report that.

## Model class
LightGBM first (small, noisy, tabular → trees beat NNs and overfit less). NNs /
Chronos (from ForecastAGLT) only if trees show real, stable OOS signal.

## Compute
Build + iterate locally (LightGBM trains in seconds). Offload large CPCV
hyperparameter sweeps to **Modal** (parallel, scale-to-zero) once the search
grid is big enough to matter. Deploy any winner as a **testnet** sleeve first.

## Phases
- **P1 (now):** labeling + uniqueness (this commit) → events → features → one base
  strategy (mean-rev: most discrete trades) → LightGBM → purged-CPCV eval. Falsify.
- **P2:** add Δfunding leg-events; cross-instrument/regime features; Modal sweep.
- **P3:** if P1/P2 survive, the regime-aware allocator; only then consider direct
  return prediction.
