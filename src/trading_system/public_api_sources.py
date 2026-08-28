from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .types import Candle


@dataclass(frozen=True)
class SourceDescriptor:
    name: str
    category: str
    url: str
    documentation: str
    auth: str
    project_role: str


PUBLIC_API_SOURCES = (
    SourceDescriptor("Twelve Data", "Finance", "https://twelvedata.com/", "https://twelvedata.com/docs", "apiKey", "intraday OHLCV"),
    SourceDescriptor("Frankfurter", "Currency Exchange", "https://frankfurter.dev/", "https://frankfurter.dev/v1/", "none", "daily reference rates"),
    SourceDescriptor("GNews", "News", "https://gnews.io/", "https://docs.gnews.io/endpoints/search-endpoint", "apiKey", "news context"),
)


class _JsonClient:
    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies to use external API adapters") from exc
        response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("status") == "error":
            raise RuntimeError(payload.get("message", "External API returned an error"))
        return payload


@dataclass
class TwelveDataClient:
    api_key: str | None = None
    timeout: int = 20

    def time_series(self, symbol: str, interval: str = "5min", outputsize: int = 5000, start_date: str | None = None, end_date: str | None = None) -> list[Candle]:
        key = self.api_key or os.getenv("TWELVE_DATA_API_KEY")
        if not key:
            raise RuntimeError("Set TWELVE_DATA_API_KEY to download intraday market data")
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "outputsize": str(outputsize), "timezone": "UTC"}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        payload = _JsonClient(self.timeout).get(
            "https://api.twelvedata.com/time_series",
            params=params,
            headers={"Authorization": f"apikey {key}"},
        )
        values = payload.get("values", [])
        candles: list[Candle] = []
        for row in values:
            candles.append(Candle(
                timestamp=datetime.fromisoformat(row["datetime"].replace("Z", "+00:00")),
                open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
                close=float(row["close"]), volume=float(row.get("volume") or 0),
            ))
        return sorted(candles, key=lambda item: item.timestamp)

    def download_range(self, symbol: str, start_date: str, end_date: str, interval: str = "5min", chunk_days: int = 14, sleep_seconds: float = 1.0) -> list[Candle]:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        if start >= end or chunk_days <= 0:
            raise ValueError("Invalid date range or chunk_days")
        merged: dict[datetime, Candle] = {}
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days), end)
            chunk = self.time_series(symbol, interval, 5000, cursor.isoformat(), chunk_end.isoformat())
            merged.update({candle.timestamp: candle for candle in chunk})
            cursor = chunk_end
            if cursor < end:
                time.sleep(max(0.0, sleep_seconds))
        return [merged[key] for key in sorted(merged)]


@dataclass
class FrankfurterClient:
    timeout: int = 20

    def time_series(self, start: str, end: str, base: str = "EUR", symbols: list[str] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"base": base}
        if symbols:
            params["symbols"] = ",".join(symbols)
        return _JsonClient(self.timeout).get(f"https://api.frankfurter.dev/v1/{start}..{end}", params=params)


@dataclass
class GNewsClient:
    api_key: str | None = None
    timeout: int = 20

    def search(self, query: str, language: str = "en", max_articles: int = 10, page: int = 1) -> list[dict[str, Any]]:
        key = self.api_key or os.getenv("GNEWS_API_KEY")
        if not key:
            raise RuntimeError("Set GNEWS_API_KEY to fetch news context")
        payload = _JsonClient(self.timeout).get(
            "https://gnews.io/api/v4/search",
            params={"q": query, "lang": language, "max": str(min(max_articles, 100)), "page": str(page), "sortby": "publishedAt", "apikey": key},
        )
        return payload.get("articles", [])
