from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .types import Candle


class HistoricalDataProvider(Protocol):
    def candles(self, symbol: str, granularity: str = "M5", count: int = 500) -> list[Candle]: ...


@dataclass
class OandaHistoricalProvider:
    """Read-only OANDA candle adapter; credentials are read only from environment variables."""

    api_token: str | None = None
    account_environment: str = "practice"

    def candles(self, symbol: str, granularity: str = "M5", count: int = 500) -> list[Candle]:
        token = self.api_token or os.getenv("OANDA_API_TOKEN")
        if not token:
            raise RuntimeError("Set OANDA_API_TOKEN to use the read-only historical adapter")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install the 'broker' extra to use OANDA") from exc
        host = "api-fxpractice.oanda.com" if self.account_environment == "practice" else "api-fxtrade.oanda.com"
        response = requests.get(
            f"https://{host}/v3/instruments/{symbol}/candles",
            headers={"Authorization": f"Bearer {token}"},
            params={"granularity": granularity, "count": count, "price": "M"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        candles: list[Candle] = []
        for item in payload.get("candles", []):
            if not item.get("complete", True):
                continue
            mid = item["mid"]
            candles.append(Candle(
                timestamp=datetime.fromisoformat(item["time"].replace("Z", "+00:00")),
                open=float(mid["o"]), high=float(mid["h"]), low=float(mid["l"]), close=float(mid["c"]),
                volume=float(item.get("volume", 0)),
            ))
        return candles


@dataclass
class MT5HistoricalProvider:
    """Optional read-only MetaTrader 5 adapter. Live order methods are deliberately absent."""

    def candles(self, symbol: str, granularity: str = "M5", count: int = 500) -> list[Candle]:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:
            raise RuntimeError("MT5 adapter is available on supported installations with the 'broker' extra") from exc
        timeframe = getattr(mt5, f"TIMEFRAME_{granularity}", None)
        if timeframe is None or not mt5.initialize():
            raise RuntimeError("MT5 initialization or timeframe mapping failed")
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                raise RuntimeError(f"MT5 returned no candles for {symbol}")
            return [Candle(
                timestamp=datetime.fromtimestamp(int(row["time"])),
                open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
                close=float(row["close"]), volume=float(row["tick_volume"]),
            ) for row in rates]
        finally:
            mt5.shutdown()
