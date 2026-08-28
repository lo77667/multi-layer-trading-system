from __future__ import annotations

import math

from .types import Candle


def closes(candles: list[Candle]) -> list[float]:
    return [candle.close for candle in candles]


def returns(candles: list[Candle]) -> list[float]:
    prices = closes(candles)
    return [prices[index] / prices[index - 1] - 1.0 for index in range(1, len(prices))]


def sma(values: list[float], window: int) -> float:
    if window <= 0 or len(values) < window:
        raise ValueError("Not enough values for SMA")
    return sum(values[-window:]) / window


def stddev(values: list[float], window: int) -> float:
    if window <= 1 or len(values) < window:
        raise ValueError("Not enough values for standard deviation")
    sample = values[-window:]
    mean = sum(sample) / window
    return math.sqrt(sum((value - mean) ** 2 for value in sample) / window)


def atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        raise ValueError("Not enough candles for ATR")
    true_ranges: list[float] = []
    for index in range(1, len(candles)):
        current = candles[index]
        previous = candles[index - 1]
        true_ranges.append(max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        ))
    return sum(true_ranges[-period:]) / period


def momentum(candles: list[Candle], lookback: int = 12) -> float:
    if len(candles) <= lookback:
        raise ValueError("Not enough candles for momentum")
    return candles[-1].close / candles[-1 - lookback].close - 1.0


def trend_score(candles: list[Candle], fast: int = 10, slow: int = 30) -> float:
    prices = closes(candles)
    if len(prices) < slow:
        raise ValueError("Not enough candles for trend score")
    fast_mean = sma(prices, fast)
    slow_mean = sma(prices, slow)
    volatility = stddev(returns(candles), min(20, len(candles) - 1))
    return (fast_mean / slow_mean - 1.0) / max(volatility, 1e-8)


def volume_ratio(candles: list[Candle], window: int = 20) -> float:
    if len(candles) < window + 1:
        raise ValueError("Not enough candles for volume ratio")
    baseline = sum(candle.volume for candle in candles[-window - 1:-1]) / window
    return candles[-1].volume / max(baseline, 1e-12)
