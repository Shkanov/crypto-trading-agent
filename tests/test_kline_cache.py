import time

from src.services import kline_cache


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(kline_cache, "CACHE_DIR", tmp_path / "klines")
    raw = [[1, "2", "3", "1", "2", "10", 99, "100", 5, "6", "7", "0"]]
    kline_cache.save("spot", "BTCUSDT", "1h", raw)
    got = kline_cache.load("spot", "BTCUSDT", "1h", ttl_s=600)
    assert got == raw


def test_load_miss_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(kline_cache, "CACHE_DIR", tmp_path / "klines")
    assert kline_cache.load("spot", "ETHUSDT", "5m", ttl_s=600) is None


def test_load_miss_when_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(kline_cache, "CACHE_DIR", tmp_path / "klines")
    kline_cache.save("spot", "SOLUSDT", "15m", [[1, "1", "1", "1", "1", "1", 2, "1", 1, "1", "1", "0"]])
    # Backdate the file so it reads as stale.
    p = kline_cache._path("spot", "SOLUSDT", "15m")
    old = time.time() - 10_000
    import os
    os.utime(p, (old, old))
    assert kline_cache.load("spot", "SOLUSDT", "15m", ttl_s=600) is None


def test_ttl_zero_always_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(kline_cache, "CACHE_DIR", tmp_path / "klines")
    kline_cache.save("spot", "BTCUSDT", "1m", [[1, "1", "1", "1", "1", "1", 2, "1", 1, "1", "1", "0"]])
    assert kline_cache.load("spot", "BTCUSDT", "1m", ttl_s=0) is None


def test_corrupt_cache_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(kline_cache, "CACHE_DIR", tmp_path / "klines")
    p = kline_cache._path("spot", "BTCUSDT", "1h")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json")
    assert kline_cache.load("spot", "BTCUSDT", "1h", ttl_s=600) is None
