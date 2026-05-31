---
name: user-role
description: "User is building an LLM-supervised intra-day crypto trading agent on Binance; thinks in desk-trader terms (R:R, ATR stops, scale-in, news flow, OI/liquidations)."
metadata: 
  node_type: memory
  type: user
  originSessionId: 6a6d183b-f5db-43e5-ae1c-04c3241d690c
---

User is the sole developer of a personal crypto trading system on Binance (spot + perps). Reasons fluently about:
- Indicator-confluence strategies, mean-reversion R:R math, partial TP geometry
- Funding-harvest pair trades, basis blowups
- Desk-trader inputs: orderbook depth, liquidations, OI change, funding, news flow
- Cost-of-edge math (spread + slippage + fees vs expected edge in bps)

Frames problems in terms of what a human trader would do, not abstract ML metrics — e.g. asked for a "trader sitting at desk" regime where the LLM uses tools the same way a human would. Treat the LLM as an *operator*, not a fancy classifier, when collaborating on this codebase.
