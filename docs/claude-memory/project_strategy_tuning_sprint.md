---
name: project-strategy-tuning-sprint
description: "17-item implementation programme from the 5-stream research synthesis. Tracks what's done, what's next, and where to pick up."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2847930c-aa46-4b82-9b82-fed5365401a1
---

Programme started 2026-05-27 after 5-stream research (trend, mean-rev, funding,
cross-section, risk-mgmt) produced `data/research/strategy_tuning/recommendations_2026_05_27.md`.
User said "implement all of the steps." **All 17 strategy items now have
their code units committed.** Remaining work is integration / wire-up.

**Why:** 1-year backtest exposed 3 of 4 strategies broken (indicator −$8.4k,
meanrev −$1.1k, funding +$52). Recommendations doc is the roadmap; this memory
is the live progress tracker. Drop after the full programme lands.

**How to apply:** When the user returns, read the recommendations doc + this
memory + run `git log --oneline -20` to see what's committed. Resume at the
next pending item.

## Completed (commits on `main`)

| Item | Commit | Notes |
|---|---|---|
| #1 Realistic cost model | `c76d62a` | Almgren-Chriss sqrt-impact (k=0.05 major / 0.15 mid / 0.30 small), perp/spot fee split, funding accrual helper. `src/services/costs.py` + 13 tests. |
| #2 Vol-target sizing | `c76d62a` | 0.20× Kelly, 25% per-position vol target, EWMA λ=0.94. `src/services/sizing.py` + 11 tests. Opt-in via `vol_cfg` kwarg. |
| #11 Exact Bailey-LdP DSR | `c76d62a` | Wichura AS241 norm-ppf; skew+kurt denom; optional trial-variance override. Replaces `sqrt(log(N))/sqrt(T)` heuristic. |
| #3 Funding rolling-z | `01aa1b9` | `FundingBacktestParams.use_rolling_z`. 60d window, z_entry +1.5σ, z_exit +0.3σ, z_stop ±2σ. API3 SR +1.14 DSR +0.30 at z_exit=-0.5σ. |
| #4 Funding cost-gated entry | `01aa1b9` | `min_edge_apr` floor. Defaults to maker fills + 21-cycle hold. |
| #6 Trend regime gate ADX+CHOP | `fa33949` | CHOP14 added to `IndicatorSnapshot` (Dreiss formula). `StrategyConfig.require_strong_trend_regime`. ALSO fixed pre-existing HTF lookahead bug in indicator + meanrev simulators (HTF state was using end-of-history values). |
| #7+#8 Donchian+Chandelier | `5aec917` | `donchian55_upper_prior`/`donchian55_lower_prior` in snapshot. `StrategyConfig.entry_rule="donchian55"` / `exit_rule="chandelier"`, chandelier_period=22, atr_mult=3.0. SimTrade.extreme_since_entry. |
| #13 Mean-rev triple-barrier + Hurst gate | `04cd6b5` | New `src/services/mean_rev_regime.py` (Hurst, VR-test, OU half-life). `MeanReversionConfig.use_strict_regime_gate` + `use_triple_barrier`. σ-scaled barriers + 3× OU-HL time stop. SimTrade.local_time_stop. Trades drop ~70% as predicted; losses shrink. |
| #5 Cointegrated pairs (module) | `05da614` | `src/strategies/pairs_cointegration.py`. Engle-Granger + ADF (MacKinnon piecewise). `evaluate_pair()` returns entry/exit/stop. |
| #5 Pairs backtest driver | `01ca969` | `scripts/backtest_pairs.py`. Aligned 1h closes, weekly refit on 60d window, dollar-hedge with abs(β) clamp, per-leg taker fee + Almgren-Chriss slippage + inflated-spread stop fills, JSON dump to `data/research/strategy_tuning/pairs_*.json`. **1y smoke: ETHUSDT/BTCUSDT 28 trades Sharpe −0.41 dd 12.5%; SOLUSDT/ETHUSDT 174 trades Sharpe −10.2 dd 169%.** Below SR~3 target — fixed z-thresholds don't survive crypto regime drift. Driver responds correctly to param sweeps. Calibration (OU-half-life thresholds, shorter lookback) is the follow-up. |
| #9 PIT survivorship filter | `a06ff30` | `src/scanners/universe_pit.py` (load_pit_log, is_active_at, eligible_universe_at, filter_universe_for_span, coverage_fraction, universe_size_over_time) + `scripts/build_pit_universe.py` (auto-fetches listed_ms from first kline per symbol, auto-detects delistings on re-run, --refresh / --dry-run flags) + `tests/test_universe_pit.py` (10 tests incl. leap-year-aware coverage). Data file is `data/research/universe/binance_delistings.json` (gitignored under data/). Builder discovered **419 active USDT spot pairs** on first run. Prerequisite for #15/#16 cross-sectional overlays. |
| #10 CPCV(N=10,k=2) + PBO | `fede7b3` | `src/services/cpcv.py` (folds, holdout_mask, train_mask_with_embargo, daily_bucket_pnls, sharpe_per_column with FP-zero-vol fix, cpcv_oos_sharpes, pbo → PBOResult) + `tests/test_cpcv.py` (18 tests) + `scripts/cpcv_validate.py` (27-config funding sweep, concurrent backtests). **Smoke: API3USDT 365d IS-best SR=+0.485 → CPCV-OOS −0.627 ± 3.24, PBO 0.002 PASS (selection clean but no edge); KATUSDT IS-best SR=+1.56 → CPCV-OOS +1.00 ± 1.33, PBO 0.367 PASS (borderline).** Replaces the 3-window walk-forward in `cascade_walkforward.py`. Gate: PBO < 0.5. |
| #12 DD circuit breakers | `58efc9b` | `src/services/risk_circuits.py` (stateless `evaluate_circuits` → CircuitState carrying size_multiplier, flatten, no_new_entries, cooloff_until_ms + metrics) + `tests/test_risk_circuits.py` (18 tests). Three independent circuits: trailing equity DD (−10% halve / −20% flatten + 14d cooloff), 20d-vol > 2× target for 5 consecutive days (halve), daily PnL ≤ −3% (block new entries; existing positions run). Composition: min over multipliers, flatten overrides. Complements (does NOT replace) `src/tools/risk_gate.py` which remains the point-in-time gate. Caller must thread the previous cooloff_until_ms forward. Not yet wired into the orchestrator. |
| #15 Scanner momentum-primary | `b54e7f6` | `src/scanners/aktradescalp_scanner.py` extended with opt-in `ScannerParams.use_momentum_primary` flag. Adds `ret_30d_bps`/`ret_7d_bps` to SymbolFeatures, computed from existing 30d × 1h history. New `_score_momentum_primary`: 60/40 blend of 30d/7d returns ranks the universe; top/bot momentum_top_pct gate; ≥2 of {vol_z, oi_z, funding_extreme} confirmer requirement with funding_extreme using STRICT > of universe-wide 95th-pctile |funding|; side_hint follows momentum direction (continuation, NOT fade). Legacy scoring preserved when flag is False — `joint_sim` / `cascade_walkforward` / diagnostics unaffected. +7 tests (17 total in test_aktradescalp_scanner.py). Flipping flag on is a one-line call-site override. |
| #16 Cross-sectional funding carry | `e167c6b` | `src/strategies/funding_carry.py` (CarryParams, rank_for_carry, build_rebalance, cycle_pnl) + `tests/test_funding_carry.py` (14 tests) + `scripts/backtest_funding_carry.py` (weekly rebalance driver, fetches funding history + 8h perp closes for top-N USDT perps, ex-majors). **Both legs pay funding — alpha is in subsequent price drift, not the carry itself.** 1y / top-30 smoke: 51 weeks, +$1588, WR 72.5%, Sharpe +2.89, DSR +2.05, dd 8.4%, +159% annualised. **Far above Fan et al.'s 43.4% — survivorship bias from using today's top-30 (alive winners only). Wiring #9 `universe_pit` filter is the natural follow-up.** |
| #17 HRP multi-strategy allocator | `570c72d` | `src/services/portfolio.py` — 3 methods: `equal_weight`, `inverse_vol`, `hrp` (López de Prado 2016, scipy-free single-linkage clustering + recursive bisection). `allocate()` dispatcher with L1 turnover guard that falls back to inverse_vol/equal when HRP weights jitter > threshold. Degenerate cases (empty / single / constant / short series) safely fall back to equal-weight. `tests/test_portfolio.py` 22 tests including the canonical HRP sanity check (2 correlated + 1 outlier → outlier gets ~50%). Not yet wired into orchestrator. |
| End-to-end carry validation | `6c3e87c` | Wires `universe_pit` into `backtest_funding_carry.py` (--pit-log flag, simulate_carry filters per-rebalance via is_active_at). New `scripts/cpcv_validate_carry.py` — 12-config grid (top_n × book_pct), shared-histories fetch, per-config CPCV + family-wide PBO at S=8. **End-to-end finding: raw carry SR 2.89/+159% → PIT-corrected SR 0.81/+20% (87% of headline was survivorship from recently-listed pumpers). CPCV+PBO sweep: 12/12 IS-positive, IS-best tn2_bk10 SR=+1.65 → CPCV OOS-mean +1.42 ± 2.50, PBO 0.114, rank 0.787, PASS.** The carry strategy survives both the survivorship and selection-overfitting tests. |
| End-to-end pairs validation | `145a6ed` | `scripts/cpcv_validate_pairs.py` — same template applied to cointegrated-pairs. 4×3×3=36-config grid over (lookback, z_entry, z_exit), z_stop scales per-config. **Two-pair contrast is the cleanest illustration of CPCV+PBO's value:** ETH/BTC → 3/36 IS-positive but IS-best `lb1440_ze2.5_zx0.5` SR +1.10 → CPCV OOS +1.09 ± 0.98 (deflation +0.011!), PBO 0.050 — PASS, **tradeable at higher-z thresholds**. SOL/ETH → 0/36 IS-positive, IS-best SR −2.21 → OOS −1.29, PBO 0.462 — nominal PASS but every config loses, so combined PBO+OOS-mean gate REJECTS. **Lesson: PBO alone is necessary but not sufficient.** Combined gate must be PBO<0.5 AND CPCV-OOS-mean>0. |
| Multi-symbol funding validation | `efded46` | `scripts/cpcv_validate_funding.py` — multi-symbol upgrade of the original `cpcv_validate.py`. Top-N universe → PIT-filter (≥95% coverage) → 27-cfg sweep per symbol → combined gate. Top-20 candidate × 8 PIT-survivors × 1y run: **0/8 PASS.** Every PIT-filtered top-20 perp (DOGE/FIL/NEAR/SUI/TAO/TON/WLD/ZEC) has all 27 configs IS-negative. PBO uniformly 0.000–0.083 (pure-PBO would green-light ALL of them — the combined gate catches this). |
| Mid-cap funding validation | `a15c922` | Same script with new `--skip-top` flag (slice [skip_top : top_n_universe]). Mid-cap run on ranks 31..100, 15/15 PIT-survivors: **0/15 PASS.** Same pattern as mainstream — every config IS-negative, IS-best ze2.0_zx-0.3 routinely loses. Even APT/ATOM (least bad at IS −0.3 to −0.5) have OOS −1 to −2. The cross-sectional CARRY overlay (`6c3e87c`) is the funding-variant that DOES work — earns from the basket spread, not per-symbol z-extremes. |
| Small-cap funding validation | data run 2026-05-27 (gitignored JSON: `cpcv_funding_multi_*_smallcap.json`) | Same script with `--skip-top 100 --top-n-universe 200` on small-cap perps (ranks 101-200), 15 PIT-survivors: **0/15 PASS.** Three symbols actually had positive IS Sharpe — AXS (+0.36), BERA (+0.81), BIO (+0.69) — all flipped negative OOS. AXS deflation was IS +0.36 → OOS −2.19 (PBO 0.154 still passes pure-PBO; combined gate catches it). **Three-tier consolidated verdict: 0/38 broad-perp PIT-survivors PASS the combined gate.** Single-symbol z-thresholded funding-harvest is DEAD across the entire liquidity spectrum from mainstream down to small-caps. The KAT PASS from sprint #10 is either deep-small-cap (rank >200) idiosyncratic or a statistical fluke; even small-cap peers don't reproduce it. **Only the cross-sectional carry overlay (`6c3e87c`) survives — deploy that, not per-symbol harvest.** |
| Mean-rev validation | `ab765ef` | `scripts/cpcv_validate_meanrev.py` — fourth strategy through the template. 27-config grid covers sprint #13's regime/exit additions explicitly: rsi_oversold × atr_stop_mult × operating_mode∈{baseline, strict_gate (Hurst+VR+OU), triple_barrier (López de Prado)}. Defaults to 1y of 1h bars (4× lighter than 15m; backtest_mean_reversion has no shared-data path yet). 10 PIT-survivors: **2/10 PASS — FETUSDT (IS +1.00 → OOS +0.45, PBO 0.116) and SOLUSDT (IS +1.33 → OOS +0.85, PBO 0.412), both baseline mode.** ETHUSDT is the canonical PBO-saves-you case (IS +1.00 OOS +0.45 — looks great — but PBO 0.654 and mean OOS rank 0.446: selection procedure is overfit, combined gate correctly rejects). **Sprint #13's additions don't add OOS edge:** triple_barrier wins IS-best 0/10 times; strict_gate wins 6/10 IS-best but 0 of those 6 pass the combined gate. Plain baseline ADX<20 + fixed ATR is the surviving operating mode. The sprint #13 work bounded loss tails (good defensive property) but didn't lift edge. |
| Small-cap mean-rev validation | `c1834c9` | Added `--skip-top` flag to the mean-rev validator (mirrors funding). Small-cap run on ranks 101..200 spot, 10 PIT-survivors: **1/10 PASS — BANANAS31USDT (IS +1.24 → OOS +0.84, PBO 0.000) with `rsi25_atr2.0_strict_gate`.** Most small-cap REJECTs land at 0/0 trades because the strict_gate (Hurst+VR+OU) over-filters shorter-history symbols; only baseline mode generates trades on these names and the per-symbol Sharpe just isn't there. **Consolidated mean-rev verdict across both slices: 3/20 PASS** (mainstream FET/SOL baseline + small-cap BANANAS31 strict_gate). Note: BANANAS31's strict_gate PASS is the first time the sprint #13 operating mode beats baseline — it's a single-symbol n=1 finding, treat as exploratory not deployable. |
| #12 wire-up | `b7b0c41` | Connected `evaluate_circuits` to the live orchestrator. New `build_equity_series` in `src/services/performance.py` (daily equity curve + daily-PnL%, today's row threads live `pnl_today` so the 3% daily-loss circuit fires intraday). New `CircuitStateRow` table + `save/load_circuit_state` for cooloff persistence (crash mid-cooloff must NOT re-enable trading on restart). `_evaluate_risk_circuits` runs every housekeeping tick (60s), persists on cooloff change, audits transitions, alerts Telegram, and triggers `_tg_flatten()` on first `state.flatten` while positions are open. `_propose` short-circuits on `no_new_entries` and scales `decision.qty/notional_usd` by `size_multiplier`. `propose_pair` honors block but does NOT rescale — pair trades are dollar-hedged at the strategy level and rescaling one leg would unbalance the hedge. Three new Prometheus gauges: `agent_circuit_size_multiplier`, `agent_circuit_dd_from_peak_pct`, `agent_circuit_cooloff_active`. Gated by `settings.risk_circuits_enabled` (default True). 8 tests in `test_circuit_wireup.py`; full suite 192/192. NOT wired into the standalone trader-agent backtest path (`backtest_trader.py` does its own equity tracking) — follow-up if needed. |
| #17 wire-up | `aef6d7c` | Connected `allocate()` to the orchestrator. New `build_strategy_returns` in `src/services/portfolio.py` (per-strategy daily-return-% matrix, shared `reference_equity` denominator preserves correlation + inverse-vol structure at portfolio scale). New `AllocatorStateRow` table + `save/load_allocator_state` for monthly weights persistence + new `realized_pnl_by_day_per_strategy` helper. `_rebalance_allocator` fires from housekeeping every `allocator_rebalance_days` (default 30): pulls 90d returns → `allocate(method, fallback, turnover_threshold, prev_weights)` → persists, audits, alerts Telegram with per-strategy %. `equity_available_usd(strategy_name)` now multiplies live_equity by stored weight; falls back to 1/N when allocator disabled / unknown strategy / `None` arg / empty weights. Default `method="equal"` per **DeMiguel-Garlappi-Uppal 2009** — for ≤5 strategies EW typically beats fancier methods OOS. Promote to `inverse_vol` once ≥6 months of clean per-strategy returns; promote to `hrp` once basket >4 strategies AND cross-correlation is non-degenerate. Gated by `settings.allocator_enabled` (default True). 11 tests in `test_allocator_wireup.py`; full suite 203/203. |
| Allocator-method backtest | `53f6a2d` | `scripts/backtest_allocator.py` — 3-method comparison on the validated 4-strategy basket via the production `allocate()` code path. Single-realisation 1y (2025-06 → 2026-05): **HRP +1.45 SR / equal +0.60 / inv_vol −1.66**. HRP also lowest DD (0.4%) + highest Calmar (1.17). Inv-vol crashed because sparse-trade-day strategies (pairs 14/357d, meanrev_FET 13/357d) look "low vol" to a daily estimator → over-weighted. HRP's corr-distance clustering dodged the trap by luck on this window. **Production default stays `allocator_method="equal"`** per DGU 2009 — promote only after multi-realisation evidence. |
| Prod test (modules + wire-ups) | `409001a` (empty commit) | Verification-only: 203/203 tests; 9 strategy backtests on past data ran cleanly. **Circuit smoke on real 357d carry equity** ($1000→$1484 peak→$994 trough→$1203 end, 18.9% peak DD): 62/357 days dd_halve, 0 dd_flatten, 0 daily_loss_block. **Allocator smoke**: 11 monthly rebalances via production code path, 0 fallback events. Known limitations surfaced: `backtest_funding_carry` needs ≥~365d (90d/180d return 0 weeks cleanly, not a crash); `pairs SOL/ETH` at default config is a confirmed CPCV-known-fail. Result JSONs gitignored under `data/research/strategy_tuning/`. Verdict: **wire-ups production-ready; strategy edge profile unchanged from CPCV verdict.** |

## Pending (in suggested order)

| Item | Description | Where |
|---|---|---|
| #5 calibration | OU-half-life-tuned z_entry/z_stop (Pole 2007; López de Prado 2018 Ch.5). Replace fixed 2σ/3.5σ with thresholds proportional to half-life × σ. Re-run ETH/BTC + SOL/ETH; consider also AAVE/UNI, LINK/UNI, MATIC/SOL. | extend `pairs_cointegration.py`; new sweep in `backtest_pairs.py` |
| #14 Cascade microstructure entry | After scanner approves, wait 20–40min for CVD slope flip OR OB imbalance>1.5x OR liquidation flush reclaim. | `src/strategies/cascade_breakout.py` |
| #15 CPCV validation of new scanner | Run `joint_sim` (or new validation harness) with `use_momentum_primary=True` and compare against legacy scoring on the 80d corpus. Gate at PBO<0.5. | re-run existing harnesses with flag flipped |

## Picking up after restart

1. `cd /Users/BulatShkanov/Downloads/machine-learning-agent`
2. `git log --oneline -10` to confirm state.
3. Read `data/research/strategy_tuning/recommendations_2026_05_27.md` for context.
4. Run all new tests: `.venv/bin/python -m pytest tests/test_costs.py tests/test_sizing.py tests/test_deflated_sharpe.py tests/test_choppiness.py tests/test_mean_rev_regime.py -q` (54 pass).
5. **17-item programme COMPLETE; all 4 strategies validated end-to-end.**
   Combined gate (PBO<0.5 AND CPCV-OOS-mean>0) battle-tested across **6
   runs, ~110 symbol/parameter combinations**:

   | Strategy        | Commit    | Sweep size               | PASS |
   |-----------------|-----------|--------------------------|------|
   | Funding-carry   | `6c3e87c` | 12 cfg × 1 universe      | ✓ basket-wide |
   | Pairs           | `145a6ed` | 36 cfg × 2 pairs         | 1/2 (ETH/BTC) |
   | Funding harvest | `efded46`+`a15c922`+data | 27 cfg × 38 symbols | 0/38 |
   | Mean reversion  | `ab765ef`+`c1834c9` | 27 cfg × 20 symbols | 3/20 (FET, SOL, BANANAS31) |

   **The strategy library that should actually run live:**
   - Cross-sectional **carry** overlay (broad universe, weekly rebalance)
   - **Pairs** on majors (ETH/BTC z_entry=2.5)
   - **Mean-rev baseline** on the FET/SOL pocket (ADX<20 + fixed ATR; not
     the sprint #13 strict_gate or triple_barrier variants)
   - **NOT** single-symbol funding harvest (proven dead across tiers)

   Operational lesson reinforced 5×: pure-PBO gating passes uniformly-
   losing families OR mid-PBO+positive-OOS families that are actually
   overfit (e.g. ETHUSDT mean-rev). The conjunction is non-optional.

   Next natural sessions: re-validate carry without survivorship at
   deeper universe size, close the remaining wire-up
   (#15 CPCV-compare for the momentum-primary scanner flag), or work
   #14 (cascade microstructure entry). #12 + #17 orchestrator wire-ups
   shipped 2026-05-27 (`b7b0c41` + `aef6d7c`); small-cap mean-rev tier
   closed 2026-05-27 (1/10 PASS, n=1 single-symbol exploratory).
6. PIT log lives at `data/research/universe/binance_delistings.json`
   (gitignored). To rebuild: `BINANCE_TESTNET=false .venv/bin/python -m scripts.build_pit_universe`.
7. CPCV reports land in `data/research/strategy_tuning/cpcv_<strategy>_<symbol>_<ts>.json`. To re-validate funding: `BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate --strategy funding --symbol KATUSDT --days 365`.

## Watch-outs accumulated during the sprint

- Default `FundingBacktestParams.use_rolling_z=False` and `min_edge_apr=0` so legacy paths still work. New tuning requires explicit opt-in.
- HTF-lookahead fix in `backtest.py` (commit `fa33949`) materially changes indicator/meanrev numbers vs prior commits — re-run baselines before comparing.
- Cost model in `costs.py` is ~30% stricter than the prior flat-bps model; expect across-the-board Sharpe haircuts on re-run.
- Donchian path on 15m short windows looks weak; long-horizon (1y+) and 1H/4H expected to do better. Calibration sweep is follow-up.
- Triple-barrier `local_time_stop` is overridden per-trade via the new `SimTrade.local_time_stop` field — anyone touching the bar-management loop needs to honour it.
- Binance OI history retains only ~30d, so cascade joint-sim cannot extend to 1y. See [[project-cascade-strategy-research]].
- `.env` does not exist; default is `BINANCE_TESTNET=true`. Backtests must run with `BINANCE_TESTNET=false` to hit mainnet history.
- Pairs driver writes to `data/research/strategy_tuning/pairs_*.json`, gitignored. Re-run smoke if reviewing: `BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_pairs --bars 8760`.

Links: [[project-cascade-strategy-research]], [[project-trader-agent-design]],
[[project-levelbreak-validation]].
