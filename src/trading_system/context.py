from __future__ import annotations

from dataclasses import dataclass

from .indicators import momentum, trend_score
from .types import AnalystSignal, ContextAssessment, Decision, MarketSnapshot, Strategy


@dataclass
class ContextModulator:
    blackout_minutes: int = 120
    sentiment_extreme_threshold: float = 0.75

    def detect_regime(self, snapshot: MarketSnapshot) -> Strategy:
        score = trend_score(snapshot.candles)
        return Strategy.BREAKOUT if abs(score) >= 0.75 else Strategy.MEAN_REVERSION

    def assess(self, snapshot: MarketSnapshot, signal: AnalystSignal) -> ContextAssessment:
        reasons: list[str] = []
        if snapshot.upcoming_event_minutes is not None and snapshot.upcoming_event_minutes <= self.blackout_minutes:
            return ContextAssessment(
                decision=Decision.WAIT,
                strategy=Strategy.NONE,
                size_multiplier=0.0,
                reasons=["Economic event is inside the blackout window"],
            )
        if snapshot.sentiment_sources < 2:
            return ContextAssessment(
                decision=Decision.REJECT,
                strategy=Strategy.NONE,
                size_multiplier=0.0,
                reasons=["Two independent sentiment sources are required"],
            )

        regime = self.detect_regime(snapshot)
        size_multiplier = 1.0
        if abs(snapshot.sentiment_score) >= self.sentiment_extreme_threshold:
            size_multiplier = 0.5
            reasons.append("Extreme sentiment: position size reduced by 50%")
        if snapshot.sentiment_score * (1 if signal.side.value == "long" else -1) < -0.50:
            reasons.append("Signal is contrarian to extreme sentiment; retained only at reduced size")
            size_multiplier = min(size_multiplier, 0.5)
        if regime != signal.strategy:
            reasons.append(f"Regime mismatch: signal={signal.strategy.value}, regime={regime.value}")
            return ContextAssessment(
                decision=Decision.WAIT,
                strategy=regime,
                size_multiplier=0.0,
                reasons=reasons,
            )
        reasons.append(f"Regime confirmed: {regime.value}")
        if abs(momentum(snapshot.candles)) < 0.00005:
            reasons.append("Momentum is too weak")
            return ContextAssessment(Decision.WAIT, regime, 0.0, reasons)
        return ContextAssessment(Decision.APPROVE, regime, size_multiplier, reasons)
