from __future__ import annotations

from src.agents.strategy_agent import validate_and_clamp
from src.models.types import StrategyConfig


def _cur() -> StrategyConfig:
    return StrategyConfig(allowed_symbols=["BTCUSDT", "ETHUSDT"])


def test_validates_subset_and_clamps():
    cur = _cur()
    proposed = {
        "allowed_symbols": ["BTCUSDT", "PEPE1000USDT"],  # PEPE not in universe -> filtered
        "feature_weights": {"trend": 5.0, "momentum": 0.5, "volume": 0.5,
                            "volatility": 0.5, "pattern": 0.5},
        "long_score_threshold": 0.99,
        "short_score_threshold": -0.99,
        "min_confidence": 0.99,
        "atr_stop_mult": 10.0,
        "rr_target": 10.0,
        "enabled_sides": ["long"],
        "notes": "test",
    }
    out = validate_and_clamp(proposed, cur, ["BTCUSDT", "ETHUSDT"])
    assert out is not None
    assert out.allowed_symbols == ["BTCUSDT"]
    assert abs(sum(out.feature_weights.values()) - 1.0) < 1e-6
    assert 0.20 <= out.long_score_threshold <= 0.65
    assert -0.65 <= out.short_score_threshold <= -0.20
    assert 0.30 <= out.min_confidence <= 0.75
    assert 1.2 <= out.atr_stop_mult <= 3.5
    assert 1.2 <= out.rr_target <= 3.0
    assert out.version == cur.version + 1


def test_rejects_empty_symbol_set():
    cur = _cur()
    proposed = {"allowed_symbols": ["UNKNOWN"]}
    out = validate_and_clamp(proposed, cur, ["BTCUSDT", "ETHUSDT"])
    assert out is None
