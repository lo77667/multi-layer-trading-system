from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .types import Candle, MarketSnapshot

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def load_candles(path: str | Path) -> list[Candle]:
    """Load and validate OHLCV candles from a CSV file."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        candles: list[Candle] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                candle = Candle(
                    timestamp=timestamp,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid candle on CSV row {row_number}: {exc}") from exc
            if candle.low <= 0 or candle.high < candle.low or candle.open <= 0 or candle.close <= 0:
                raise ValueError(f"Invalid OHLC values on CSV row {row_number}")
            if candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close):
                raise ValueError(f"OHLC consistency error on CSV row {row_number}")
            if candle.volume < 0:
                raise ValueError(f"Negative volume on CSV row {row_number}")
            candles.append(candle)
    candles.sort(key=lambda item: item.timestamp)
    if len(candles) < 2:
        raise ValueError("At least two candles are required")
    return candles


def snapshot_from_csv(symbol: str, path: str | Path) -> MarketSnapshot:
    return MarketSnapshot(symbol=symbol, candles=load_candles(path))


def generate_sample_candles(rows: int = 500, seed: int = 7, start_price: float = 1.08) -> list[Candle]:
    """Generate deterministic synthetic candles for tests and local demonstrations."""
    rng = random.Random(seed)
    timestamp = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=rows)
    price = start_price
    candles: list[Candle] = []
    for index in range(rows):
        drift = 0.00002 * math.sin(index / 25.0) + (0.00001 if index % 120 < 60 else -0.00001)
        shock = rng.gauss(0, 0.00035)
        open_price = price
        close = max(0.0001, open_price * (1 + drift + shock))
        high = max(open_price, close) * (1 + abs(rng.gauss(0, 0.00015)))
        low = min(open_price, close) * (1 - abs(rng.gauss(0, 0.00015)))
        volume = 1000 + abs(rng.gauss(0, 250)) * (1.8 if index % 37 == 0 else 1.0)
        candles.append(Candle(timestamp=timestamp, open=open_price, high=high, low=low, close=close, volume=volume))
        price = close
        timestamp += timedelta(minutes=1)
    return candles


def write_candles_csv(candles: list[Candle], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for candle in candles:
            writer.writerow([
                candle.timestamp.isoformat(), candle.open, candle.high, candle.low, candle.close, candle.volume
            ])
