from datetime import datetime, timedelta, timezone

from trading_system.public_api_sources import TwelveDataClient
from trading_system.types import Candle


def test_twelve_data_download_range_chunks_and_deduplicates(monkeypatch):
    calls = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def fake_time_series(self, symbol, interval, outputsize, start_date=None, end_date=None):
        calls.append((start_date, end_date))
        start = datetime.fromisoformat(start_date)
        return [Candle(start, 1, 1.1, 0.9, 1.05, 10)]

    monkeypatch.setattr(TwelveDataClient, "time_series", fake_time_series)
    candles = TwelveDataClient(api_key="test").download_range(
        "EUR/USD", base.isoformat(), (base + timedelta(days=30)).isoformat(), chunk_days=14, sleep_seconds=0
    )
    assert len(calls) == 3
    assert len(candles) == 3
    assert candles[0].timestamp < candles[-1].timestamp
