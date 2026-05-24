"""Rolling correlation matrix from real price history.

Replaces the hardcoded `0.85` BTC-beta assumption in the risk gate with a
correlation derived from actual recent returns. Run on a daily cron from
the housekeeping loop; the matrix is read by `risk_gate._btc_beta_equiv`.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import structlog

from src.tools.binance_client import BinanceClient

log = structlog.get_logger(__name__)


class CorrelationMatrix:
    """Per-symbol rolling correlation to BTC, computed from N days of closes.

    `update()` fetches `1h` klines and computes pearson(returns_sym, returns_btc).
    `beta_to_btc(symbol)` returns the latest estimate (or `default` if missing).
    """

    def __init__(self, binance: BinanceClient, days: int = 30,
                 default_beta: float = 0.85) -> None:
        self.binance = binance
        self.days = days
        self.default_beta = default_beta
        self._betas: dict[str, float] = {}
        self._updated_at: Optional[datetime] = None

    async def update(self, symbols: list[str]) -> None:
        # 1h klines for `days` days
        limit = max(24 * self.days, 100)
        try:
            btc_raw = await self.binance.fetch_klines("BTCUSDT", "1h", limit=limit,
                                                       market="spot")
        except Exception as e:
            log.warning("correlation.btc_fetch_failed", err=str(e))
            return
        btc_closes = np.array([float(r[4]) for r in btc_raw])
        if btc_closes.size < 50:
            return
        btc_ret = np.diff(np.log(btc_closes))

        for sym in symbols:
            if sym == "BTCUSDT":
                self._betas[sym] = 1.0
                continue
            try:
                raw = await self.binance.fetch_klines(sym, "1h", limit=limit, market="spot")
            except Exception as e:
                log.warning("correlation.sym_fetch_failed", sym=sym, err=str(e))
                continue
            closes = np.array([float(r[4]) for r in raw])
            if closes.size < 50:
                continue
            # Align lengths from the right (most recent N).
            n = min(closes.size, btc_closes.size)
            sym_ret = np.diff(np.log(closes[-n:]))
            btc_ret_a = btc_ret[-len(sym_ret):]
            if sym_ret.size < 30 or sym_ret.std() == 0 or btc_ret_a.std() == 0:
                continue
            # Beta in the CAPM sense: cov(s, m)/var(m).
            cov = np.cov(sym_ret, btc_ret_a, ddof=1)[0, 1]
            var_btc = np.var(btc_ret_a, ddof=1)
            beta = cov / var_btc if var_btc > 0 else self.default_beta
            # Clip to a sane range; alts rarely have |beta| > 2 to BTC on majors.
            beta = max(-2.0, min(2.0, float(beta)))
            self._betas[sym] = beta

        self._updated_at = datetime.utcnow()
        log.info("correlation.updated", n_symbols=len(self._betas),
                 betas=self._betas)

    def beta_to_btc(self, symbol: str) -> float:
        return self._betas.get(symbol, self.default_beta)

    def is_stale(self, max_age: timedelta = timedelta(hours=12)) -> bool:
        if self._updated_at is None:
            return True
        return (datetime.utcnow() - self._updated_at) > max_age
