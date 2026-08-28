from pathlib import Path

import pytest

from trading_system.data import generate_sample_candles, load_candles, write_candles_csv
from trading_system.scanner import InitialScanner
from trading_system.types import MarketSnapshot


def test_csv_round_trip(tmp_path: Path):
    path = tmp_path / "sample.csv"
    write_candles_csv(generate_sample_candles(150), path)
    candles = load_candles(path)
    assert len(candles) == 150
    assert candles[0].timestamp <= candles[-1].timestamp


def test_loader_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,open,high\n2026-01-01T00:00:00+00:00,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing CSV columns"):
        load_candles(path)


def test_scanner_limits_results():
    snapshots = [MarketSnapshot(f"PAIR_{index}", generate_sample_candles(150, seed=index)) for index in range(12)]
    opportunities = InitialScanner(top_n=5, min_history=120).scan(snapshots)
    assert 0 < len(opportunities) <= 5
    assert all(0 <= item.score <= 1 for item in opportunities)
