# Delta-Neutral Basis Book with Funding Gate — Implementation Spec

**Status:** design, not built. Offline-validated (`research/portfolio/funding_gate.py`,
3y backtest). Deploy path: paper → testnet → small mainnet, gated by the
validation checklist at the end. Current gate state (2026-07): **OFF (hold USD)**.

## 1. Objective

Harvest perpetual-futures funding with ~zero price risk (delta-neutral basis:
long spot + short perp of the same asset), but **only when funding clears the
cost hurdle**; otherwise hold USD. Through-cycle target ~7%/yr; the point is
stability + beating cash, not max PnL. This is a yield product, not an edge bet.

## 2. Universe

Fixed basket of liquid names with **funding positive in every year 2023–2026**
(persistence-screened, not single-window):

    BTCUSDT, ETHUSDT, LINKUSDT, UNIUSDT, LTCUSDT, DOGEUSDT, AAVEUSDT

Anchors = BTC/ETH (deepest, lowest tail risk). Explicitly EXCLUDE names that
went funding-negative in any year (SOL, XRP, FIL, DOT, ATOM, INJ, BNB) — on a
basis book a funding flip turns earning into bleeding.

## 3. The funding-gate rule (the trigger)

Economically-anchored thresholds (NOT tuned to the backtest):

    HURDLE = USD_yield (4.5%) + borrow/exec (2.5%) = 7.0 %/yr   # OFF line
    ON     = HURDLE + 2.0 = 9.0 %/yr                            # hysteresis band

Signal, computed daily (00:00 UTC), PIT (trailing data only):

    for each name: ann_funding_i = mean(funding rate over trailing 21 days) * 3 * 365
    S = mean(ann_funding_i over the basket)      # basket-level signal, %/yr

State machine (hysteresis prevents whipsaw near the hurdle):

    if book OFF and S >= ON (9%)  -> turn ON
    if book ON  and S <  HURDLE (7%) -> turn OFF
    else keep current state

Within-book selection when ON:

    include name k  iff  ann_funding_k >= 7%/yr   # drop names gone lean/negative

When OFF: fully in USD (stablecoin / T-bill ~4.5%/yr). Historically ON ~36% of
days, ~3 flips/yr (holding periods of weeks–months).

## 4. Book construction (per included name, when ON)

Delta-neutral leg pair, equal $ notional:

    LONG  spot  k   (notional = book_equity * gate_scale / n_included)
    SHORT perp  k   (same notional)   # reduceOnly-aware, 1x leverage on the perp

- Net delta ≈ 0: spot up = perp short down, they cancel. You keep the funding.
- `gate_scale` starts small (ramp), max 1.0. Never lever the *pair* above the
  equity backing it — the whole point is no liquidation from price moves.
- Perp leg margin: post enough isolated margin that a **40% adverse gap** does
  NOT liquidate the short before the spot gain offsets (see §6). Cross-margin on
  the account is acceptable since the spot leg is the natural hedge, but size the
  perp margin buffer explicitly.

## 5. Rebalance cadence

- **Signal / gate evaluation:** daily at 00:00 UTC (cheap; funding is slow).
- **Book rebalance:** only on (a) a gate ON/OFF transition, or (b) a name
  entering/leaving the included set, or (c) leg drift > 5% of target notional
  (spot vs perp notionals diverge as prices move — top up the smaller leg).
  Do NOT rebalance on every funding tick — that just pays fees.
- On OFF transition: unwind all legs (sell spot, buy-to-close perp), park USD.

## 6. Risk controls (the tail this strategy actually has)

The price legs cancel, so the residual risks are operational, not directional:

1. **Perp-leg liquidation on a violent gap.** Size isolated perp margin for a
   ≥40% adverse instantaneous move without liquidation. Auto-add margin if the
   perp's margin ratio crosses a threshold. This is THE risk (March-2024 style
   short-squeeze wicks); the funding-only backtest does NOT model it.
2. **Funding sign flip mid-hold.** If a name's *live* funding turns negative and
   its trailing signal drops below the per-name hurdle, drop it at the next daily
   eval (you stop paying to hold it). The gate + per-name inclusion already does
   this; make the check daily, not weekly.
3. **Spot-borrow / margin-call on the spot leg** (if the spot long is on margin).
   Prefer unlevered spot (own USD buys spot) to remove this leg of risk.
4. **Depeg / venue risk** if using a yield-bearing stablecoin as the USD parking
   asset — use a plain, liquid stable or actual USD, not a synthetic.
5. **Per-name notional cap** so no single alt leg dominates (e.g., <= 25% of book).
6. Reuse existing `risk_circuits` for account-level drawdown halt.

## 7. Infra requirements (what's missing today)

- **Spot + perp legs together.** The system is perp-only in prod
  (`live_spot_margin_enabled = false`). Need: spot market access on the same
  venue (or cross-venue), spot order execution, and combined spot+perp position
  tracking / reconciliation. This is the main build.
- A `BasisStrategy` sleeve emitting paired (spot, perp) proposals, plus a
  basis-aware executor that opens/closes both legs and keeps them balanced
  (mirror the existing `PairExecutor` pattern used for pairs trades).
- Funding-rate feed (already have `fetch_funding` / the WS mark-price funding).
- USD parking accounting when OFF (so equity/PnL reflect the cash yield).

## 8. Data / PIT notes

- Funding harvest is a **realized cashflow**, not a forecast — no prediction, no
  model. The only assumption is funding **autocorrelation** (persistence), used
  PIT via the trailing signal. Leak-audited: signal window `[i-21, i)` is disjoint
  from the earned day `i` (`funding_gate.py`); decide-on-past, collect-present.
- Backtest simplifications to fix in a live-accurate sim before capital: charge
  per-transition taker fees at each flip/rebalance; model spot borrow if levered;
  add a gap-liquidation stress scenario. All of these REDUCE the result — they
  are conservative to omit, but must be added before trusting the net number.

## 9. Expected performance (offline, 3y, funding-only, gross of tail)

    GATED (basis|USD): +7.4%/yr   (vs always-on basis +5.6%, vs always-USD +4.5%)
    time in basis: 36% of days;  ~3 regime flips/yr

Honest caveats: Sharpe in the backtest (~16) is inflated by funding smoothness
and IGNORES the gap-liquidation tail — treat +7.4%/yr as the *return*, not the
risk figure. Net of realistic costs the through-cycle edge over cash is ~2–3%/yr,
earned lumpily in fat-funding regimes. **Not a max-PnL strategy — a stable,
cash-plus yield you switch on when funding pays for it.**

## 10. Validation checklist before mainnet capital

1. Live-accurate backtest with per-transition costs + spot borrow + a
   gap-liquidation stress → net still beats USD through-cycle.
2. Paper-trade the full spot+perp open/close/rebalance loop (reconciliation
   clean, legs stay balanced, gate transitions unwind correctly).
3. Testnet dry-run of the executor (leg balancing, margin buffer, OFF-unwind).
4. Small mainnet ($ tiny), gate currently OFF so it starts in USD; only scales
   into the basis when the signal clears 9%/yr. Verify reconciliation + margin
   buffer behavior on the first real ON transition.
