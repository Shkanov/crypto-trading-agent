"""Portfolio measurement — the input layer for the stability-gated allocator.

The HRP/inverse-vol allocator (`src/services/portfolio.py`) is already built and
wired. It weights sleeves by their REALIZED daily returns — but it can only
manufacture a stable book if the sleeves it's fed are (a) at least weakly
positive and (b) low-correlated. Combining several dead/near-zero sleeves gives a
stable near-ZERO book, not max PnL.

So before committing testnet capital, this package measures — offline, over a
common window — each candidate sleeve's honest standalone expectancy + stability
+ the pairwise correlation matrix, then runs the SAME `allocate()` the live
system uses and reports the combined book's risk-adjusted PnL and the
diversification benefit over the best single sleeve.

Verdict-oriented: it answers "is there a diversified stable-positive book here at
all, or just one weak sleeve dressed up?" — falsification-first, per the
stable-max-PnL directive.
"""
