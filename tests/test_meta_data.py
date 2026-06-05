"""Tests for research/ml_meta/data.py — the pure kline parser only.

Network fetching is exercised by the one-off cache build, not by pytest (we
don't hit Binance in unit tests).
"""
import pandas as pd

from research.ml_meta.data import _klines_to_df


def test_parser_types_index_and_sort():
    rows = [
        # open_time, o, h, l, c, vol, close_time, qv, trades, tbb, tbq, ignore
        [1_700_000_000_000, "10", "12", "9", "11", "100", 1_700_003_599_999, "1100", 5, "60", "660", "0"],
        [1_700_003_600_000, "11", "13", "10", "12", "200", 1_700_007_199_999, "2400", 8, "120", "1440", "0"],
    ]
    df = _klines_to_df(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "quote_volume", "close_time"]
    assert df["close"].tolist() == [11.0, 12.0]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df["close"].dtype == float


def test_parser_dedups_and_sorts_unordered_input():
    rows = [
        [1_700_003_600_000, "11", "13", "10", "12", "200", 1_700_007_199_999, "2400", 8, "120", "1440", "0"],
        [1_700_000_000_000, "10", "12", "9", "11", "100", 1_700_003_599_999, "1100", 5, "60", "660", "0"],
        # duplicate open_time of the first row, keep=last
        [1_700_003_600_000, "11", "14", "10", "12.5", "201", 1_700_007_199_999, "2410", 9, "121", "1450", "0"],
    ]
    df = _klines_to_df(rows)
    assert len(df) == 2
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 12.5  # kept the last duplicate


def test_parser_empty():
    df = _klines_to_df([])
    assert df.empty
