"""News + macro sentiment ingestion.

Polled (not webhook-driven) so it works without inbound networking. Sources:
- CryptoPanic (best free aggregator; token optional)
- alternative.me Fear & Greed (free, daily)
- DefiLlama stablecoin liquidity (free)

Each source returns a list of NewsItem; the NewsSentimentAgent (LLM) folds
them into SentimentScore per symbol. We dedupe by (source, id) within memory.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Optional

import httpx
import structlog

from src.config.settings import get_settings
from src.models.types import NewsItem

log = structlog.get_logger(__name__)

CRYPTOPANIC_URL = "https://cryptopanic.com/api/developer/v2/posts/"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
DEFI_LLAMA_STABLES = "https://stablecoins.llama.fi/stablecoins"


class NewsService:
    def __init__(self) -> None:
        self.s = get_settings()
        self.seen_ids: set[str] = set()
        self.fear_greed: Optional[int] = None
        self.fear_greed_classification: str = "unknown"
        self.stablecoin_marketcap_usd: Optional[float] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=15.0)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def fetch_cryptopanic(self, coins: Optional[list[str]] = None) -> list[NewsItem]:
        if not self._client:
            return []
        params: dict[str, Any] = {"public": "true"}
        if self.s.cryptopanic_token:
            params["auth_token"] = self.s.cryptopanic_token
        if coins:
            params["currencies"] = ",".join(coins)
        try:
            r = await self._client.get(CRYPTOPANIC_URL, params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("cryptopanic.fetch_failed", err=str(e))
            return []

        out: list[NewsItem] = []
        for post in data.get("results", []):
            pid = str(post.get("id"))
            key = f"cryptopanic:{pid}"
            if key in self.seen_ids:
                continue
            self.seen_ids.add(key)
            currencies = [c.get("code", "") for c in post.get("currencies", []) or []]
            out.append(NewsItem(
                id=key, source="cryptopanic",
                title=post.get("title", ""),
                url=post.get("url", post.get("source", {}).get("domain", "")),
                published_ms=int(time.time() * 1000),
                coins=currencies, raw=post,
            ))
        return out

    async def fetch_fear_greed(self) -> Optional[int]:
        if not self._client:
            return None
        try:
            r = await self._client.get(FEAR_GREED_URL)
            r.raise_for_status()
            data = r.json()
            self.fear_greed = int(data["data"][0]["value"])
            self.fear_greed_classification = data["data"][0]["value_classification"]
            return self.fear_greed
        except Exception as e:
            log.warning("fear_greed.fetch_failed", err=str(e))
            return None

    async def fetch_stablecoin_mcap(self) -> Optional[float]:
        if not self._client:
            return None
        try:
            r = await self._client.get(DEFI_LLAMA_STABLES, params={"includePrices": "false"})
            r.raise_for_status()
            data = r.json()
            total = sum(
                (s.get("circulating", {}) or {}).get("peggedUSD", 0)
                for s in data.get("peggedAssets", [])
            )
            self.stablecoin_marketcap_usd = float(total)
            return self.stablecoin_marketcap_usd
        except Exception as e:
            log.warning("defillama.fetch_failed", err=str(e))
            return None

    async def poll_once(self) -> dict:
        items = await self.fetch_cryptopanic()
        await self.fetch_fear_greed()
        await self.fetch_stablecoin_mcap()
        return {
            "news": items,
            "fear_greed": self.fear_greed,
            "fear_greed_classification": self.fear_greed_classification,
            "stablecoin_marketcap_usd": self.stablecoin_marketcap_usd,
        }
