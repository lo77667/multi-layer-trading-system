from __future__ import annotations

from dataclasses import dataclass

from .indicators import atr, momentum, trend_score, volume_ratio
from .types import AnalystSignal, MarketSnapshot, ScannerOpportunity, Side, Strategy


@dataclass
class DeepAnalyst:
    atr_period: int = 14
    support_window: int = 50

    def analyze(self, snapshot: MarketSnapshot, opportunity: ScannerOpportunity) -> AnalystSignal:
        candles = snapshot.candles
        entry = candles[-1].close
        current_atr = atr(candles, self.atr_period)
        trend = trend_score(candles)
        mom = momentum(candles)
        vol = volume_ratio(candles)
        model_votes = {
            "trend": Side.LONG if trend >= 0 else Side.SHORT,
            "momentum": Side.LONG if mom >= 0 else Side.SHORT,
            "volume": opportunity.side if vol >= 1.0 else Side.SHORT if opportunity.side == Side.LONG else Side.LONG,
        }
        agreement = sum(vote == opportunity.side for vote in model_votes.values())
        confidence = min(0.99, 0.40 + 0.15 * agreement + 0.10 * min(abs(trend), 2.0))
        recent = candles[-self.support_window:]
        support = min(item.low for item in recent)
        resistance = max(item.high for item in recent)
        strategy = Strategy.BREAKOUT if abs(trend) >= 0.75 else Strategy.MEAN_REVERSION
        return AnalystSignal(
            symbol=snapshot.symbol,
            side=opportunity.side,
            confidence=confidence,
            strategy=strategy,
            entry=entry,
            support=support,
            resistance=resistance,
            atr=current_atr,
            model_votes=model_votes,
        )

    def analyze_with_optional_deep_models(self, snapshot: MarketSnapshot, opportunity: ScannerOpportunity) -> AnalystSignal:
        """Explicit extension point for trained LSTM/Transformer models.

        The baseline must remain deterministic and auditable; neural predictions should be
        injected as additional votes rather than silently replacing risk controls.
        """
        return self.analyze(snapshot, opportunity)
