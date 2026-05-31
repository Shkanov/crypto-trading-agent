# Strategies — Plain-Language Guide

A jargon-free guide to every trading strategy in this system: what it bets on,
how it tries to make money, and whether it actually works after honest testing.

> **Status key:** ✅ works · ⚠️ weak / works only in narrow cases · ❌ falsified
> (doesn't work) · 🔬 research / not deployable yet.
>
> "Honest testing" here means costs included (fees + slippage) and out-of-sample
> validation (CPCV + PBO) — not just a nice-looking backtest on past data.

---

## 1. Funding Carry ✅ (works)
**The bet:** On crypto futures, traders on one side pay a small fee to the other
side every few hours (the "funding rate"). It's like rent for holding a position.

**How it works:** Find the coins where this rent is highest and collect it — bet
*against* the crowded side on several coins at once, balanced so you don't care
if the whole market goes up or down. You just pocket the rent.

**Does it work?** Yes — the most reliable one we've validated. Steady, boring,
small income. *(Code: `src/strategies/funding_carry.py`)*

## 2. Funding Harvest ⚠️ (mostly doesn't)
**The bet:** Same "rent" idea, but on a single coin at a time, fully hedged so
there's almost no price risk.

**How it works:** When the rent on one coin gets extreme, jump in to collect it
and hedge out the price risk entirely.

**Does it work?** Rarely — it passed only 0 of 38 honest tests. The rent usually
isn't big enough to beat trading costs. *(Code: `src/strategies/funding_harvest.py`)*

## 3. Pairs Trading (e.g. ETH vs BTC) ⚠️ (worked, now faded)
**The bet:** Two coins that normally move together (like Ethereum and Bitcoin)
sometimes drift apart. They usually snap back.

**How it works:** When ETH gets unusually expensive *relative to* BTC, sell ETH
and buy BTC; profit when the gap closes. You're betting on the *relationship*,
not on market direction.

**Does it work?** It worked while the two moved in lockstep (late 2025–early
2026), but that relationship **broke in April 2026** and the edge disappeared.
Re-tested May 2026: currently loses money. *(Code: `src/strategies/pairs_cointegration.py`)*

## 4. Mean Reversion ⚠️ (works in narrow cases)
**The bet:** When a price spikes too far from its recent average, it tends to
bounce back.

**How it works:** Buy sharp dips, sell sharp spikes — but only when the market
is calm and range-bound, not trending.

**Does it work?** Sometimes — only about 3 of 20 versions survived honest
testing, mostly on smaller coins. *(Code: `src/strategies/mean_reversion.py`)*

## 5. Indicator Confluence ⚠️ (weak)
**The bet:** Several popular chart signals (momentum, trend lines, etc.) are
more trustworthy when they all agree.

**How it works:** Only trade when multiple indicators point the same way, to
filter out false signals.

**Does it work?** Weakly. Standard chart indicators on a single coin are so
widely used that any edge is mostly traded away. *(Code: `src/strategies/indicator_confluence.py`)*

## 6. Level Breakout ❌ (doesn't work)
**The bet:** Price often "breaks out" and keeps running after it pushes through
a key level (support/resistance or a trend line).

**How it works:** Wait for price to break a level, then jump in the breakout
direction.

**Does it work?** No — falsified across many coins and time periods. Most
breakouts fizzle. *(Code: `src/strategies/level_breakout.py`)*

## 7. Cascade Breakout 🔬 (research)
**The bet:** Based on studying a skilled manual trader. When lots of forced
sell-offs ("liquidations") pile up, price moves violently in a predictable
chain reaction.

**How it works:** Try to mechanically copy the pattern that trader uses around
these cascades.

**Does it work?** Partly understood — but his real skill turned out to be in
*which* coins he picks and *when*, not the mechanical rule itself. Hard to
automate. *(Code: `src/strategies/cascade_breakout.py`)*

## 8. MACD Crossover Bot ❌ (doesn't work)
**The bet:** A classic momentum signal: buy when momentum flips up, sell when it
flips down. *(This is the off-the-shelf BitMEX bot we analysed from a zip file.)*

**How it works:** Flip between betting up and betting down on Bitcoin every time
the momentum indicator crosses zero.

**Does it work?** No — falsified on every coin we tested. It flips so often that
trading fees alone bury it (lost on all 6 tests). *(Code: `scripts/backtest_macd_cross.py`)*

---

## The one-line takeaway
Most strategies that *sound* clever — breakouts, momentum crossovers, chart
indicators — **don't survive honest testing**: fees and randomness eat them. The
ones with a real, durable edge are the unglamorous ones that either collect a
structural "rent" (**Funding Carry**) or bet on a stable *relationship* between
two things (**Pairs Trading**, for as long as the relationship lasts).

A recurring lesson: an edge that worked last year can quietly **decay** (Pairs
broke in April 2026), so strategies need ongoing health checks, not a one-time
backtest.
