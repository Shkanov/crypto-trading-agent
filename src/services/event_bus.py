"""In-process async pub/sub.

Swap to Redis Streams later by keeping this module's surface: `publish`,
`subscribe(topic) -> AsyncIterator`. Producers should not block on slow
consumers, so each subscriber gets its own bounded queue and we drop on full
(with a warning).
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, AsyncIterator

import structlog

log = structlog.get_logger(__name__)


class EventBus:
    def __init__(self, queue_size: int = 1024) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._queue_size = queue_size

    def publish(self, topic: str, event: Any) -> None:
        for q in self._subs[topic]:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("event_bus.drop", topic=topic, qsize=q.qsize())

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subs[topic].append(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._subs[topic].remove(q)


# Canonical topic names
TOPIC_KLINE = "market.kline"
TOPIC_TRADE = "market.trade"
TOPIC_INDICATOR = "indicators.update"
TOPIC_SIGNAL = "signals.raw"
TOPIC_PROPOSAL = "proposals.new"
TOPIC_APPROVED = "proposals.approved"
TOPIC_FILL = "orders.filled"
TOPIC_NEWS = "news.item"
TOPIC_SENTIMENT = "sentiment.update"
TOPIC_ANOMALY = "events.anomaly"
TOPIC_TELEGRAM_OUT = "telegram.outbound"
TOPIC_STRATEGY_CONFIG = "strategy.config"
