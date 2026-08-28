from __future__ import annotations

import numpy as np
import pandas as pd

from .types import Candle


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    if not candles:
        raise ValueError("At least one candle is required")
    frame = pd.DataFrame([{
        "timestamp": candle.timestamp,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    } for candle in candles]).set_index("timestamp").sort_index()
    return frame


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(candles: list[Candle]) -> pd.DataFrame:
    """Create causal features; every feature at t uses only observations <= t."""
    frame = candles_to_frame(candles)
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    frame["return_1"] = close.pct_change()
    frame["return_5"] = close.pct_change(5)
    frame["return_20"] = close.pct_change(20)
    for period in (7, 14, 21):
        frame[f"rsi_{period}"] = _rsi(close, period)
    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    frame["macd_histogram"] = macd - signal
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    frame["bollinger_width"] = (4 * bb_std) / bb_mid.replace(0, np.nan)
    atr_20 = true_range.ewm(span=20, adjust=False, min_periods=20).mean()
    kc_mid = close.ewm(span=20, adjust=False, min_periods=20).mean()
    frame["keltner_width"] = (2 * atr_20) / kc_mid.replace(0, np.nan)
    frame["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    frame["volume_ratio_20"] = frame["volume"] / frame["volume"].rolling(20, min_periods=20).mean().shift(1)
    frame["range_pct"] = (high - low) / close.replace(0, np.nan)
    frame["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    return frame.replace([np.inf, -np.inf], np.nan)


def make_direction_labels(frame: pd.DataFrame, horizon: int = 8, threshold: float = 0.0005) -> pd.Series:
    """Binary label: future return above threshold; shift keeps labels out of features."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    future_return = frame["close"].shift(-horizon) / frame["close"] - 1
    return (future_return > threshold).astype("float").where(future_return.notna())


def model_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    excluded = {"open", "high", "low", "close", "volume"}
    columns = [column for column in frame.columns if column not in excluded]
    clean = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    return clean, columns
