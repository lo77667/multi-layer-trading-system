from __future__ import annotations

from dataclasses import dataclass

from .indicators import sma, volume_ratio
from .types import AnalystSignal, Decision, MarketSnapshot, UncertaintyAssessment


@dataclass
class UncertaintyEngine:
    volume_window: int = 20
    block_trade_threshold_pips: float = 2.0
    related_correlation_window: int = 50

    def _related_alignment(self, snapshot: MarketSnapshot, signal: AnalystSignal) -> bool:
        if not snapshot.related_returns:
            return False
        recent_means = []
        for values in snapshot.related_returns.values():
            if len(values) < min(self.related_correlation_window, 5):
                continue
            recent_means.append(sum(values[-min(self.related_correlation_window, len(values)):]))
        if not recent_means:
            return False
        aligned = sum(value >= 0 for value in recent_means) >= len(recent_means) / 2
        return aligned if signal.side.value == "long" else not aligned

    def assess(self, snapshot: MarketSnapshot, signal: AnalystSignal) -> UncertaintyAssessment:
        reasons: list[str] = []
        model_agreement = sum(vote == signal.side for vote in signal.model_votes.values()) == 3
        if model_agreement:
            reasons.append("3/3 model votes agree")
        else:
            reasons.append("Model disagreement")

        two_sources = snapshot.sentiment_sources >= 2 and len(snapshot.candles) > 0
        if two_sources:
            reasons.append("Price and sentiment sources confirmed")
        else:
            reasons.append("Missing independent data-source confirmation")

        questions_passed = 0
        if len(snapshot.candles) >= self.volume_window + 1 and volume_ratio(snapshot.candles, self.volume_window) > 1.0:
            questions_passed += 1
            reasons.append("Volume is above the prior 20-candle average")
        else:
            reasons.append("Volume is not above the prior 20-candle average")
        if snapshot.block_trade_distance_pips is not None and snapshot.block_trade_distance_pips > self.block_trade_threshold_pips:
            questions_passed += 1
            reasons.append("No nearby block trade detected")
        else:
            reasons.append("Nearby block trade evidence is missing or too close")
        if self._related_alignment(snapshot, signal):
            questions_passed += 1
            reasons.append("Related assets are directionally aligned")
        else:
            reasons.append("Related-asset alignment is not confirmed")

        score = (3 if model_agreement else 0) + (2 if two_sources else 0) + (1 if questions_passed == 3 else 0)
        if score == 6:
            return UncertaintyAssessment(Decision.APPROVE, score, reasons)
        if score >= 3:
            return UncertaintyAssessment(Decision.WAIT, score, reasons)
        return UncertaintyAssessment(Decision.REJECT, score, reasons)
