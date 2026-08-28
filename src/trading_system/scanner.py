from __future__ import annotations

from dataclasses import dataclass

from .indicators import atr, momentum, trend_score, volume_ratio
from .types import MarketSnapshot, ScannerOpportunity, Side


@dataclass
class InitialScanner:
    top_n: int = 10
    min_history: int = 120

    def score(self, snapshot: MarketSnapshot) -> ScannerOpportunity | None:
        if len(snapshot.candles) < self.min_history:
            return None
        trend = trend_score(snapshot.candles)
        mom = momentum(snapshot.candles)
        vol = volume_ratio(snapshot.candles)
        volatility = atr(snapshot.candles) / snapshot.candles[-1].close
        raw = 0.50 * abs(trend) + 0.35 * abs(mom) + 0.15 * min(vol, 3.0)
        score = max(0.0, min(1.0, raw / 6.0))
        side = Side.LONG if trend + mom >= 0 else Side.SHORT
        return ScannerOpportunity(
            symbol=snapshot.symbol,
            score=score,
            side=side,
            features={"trend_score": trend, "momentum": mom, "volume_ratio": vol, "atr_pct": volatility},
        )

    def scan(self, snapshots: list[MarketSnapshot]) -> list[ScannerOpportunity]:
        opportunities = [item for snapshot in snapshots if (item := self.score(snapshot)) is not None]
        return sorted(opportunities, key=lambda item: item.score, reverse=True)[: self.top_n]


class XGBoostScanner:
    """Optional XGBoost hook; the project remains usable without the heavy dependency."""

    def __init__(self, fallback: InitialScanner | None = None) -> None:
        self.fallback = fallback or InitialScanner()
        self.model = None

    def fit(self, *_args, **_kwargs) -> None:
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install the optional 'ml' extra to train XGBoost") from exc
        self.model = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=7,
            eval_metric="logloss",
        )
        raise NotImplementedError("Training data preparation is intentionally explicit and time-series aware")

    def scan(self, snapshots: list[MarketSnapshot]) -> list[ScannerOpportunity]:
        return self.fallback.scan(snapshots)
