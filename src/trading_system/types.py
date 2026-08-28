from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class Strategy(str, Enum):
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    NONE = "none"


class Decision(str, Enum):
    APPROVE = "approve"
    WAIT = "wait"
    REJECT = "reject"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MarketSnapshot:
    symbol: str
    candles: list[Candle]
    related_returns: dict[str, list[float]] = field(default_factory=dict)
    sentiment_score: float = 0.0
    sentiment_sources: int = 0
    upcoming_event_minutes: int | None = None
    block_trade_distance_pips: float | None = None


@dataclass
class ScannerOpportunity:
    symbol: str
    score: float
    side: Side
    features: dict[str, float]


@dataclass
class AnalystSignal:
    symbol: str
    side: Side
    confidence: float
    strategy: Strategy
    entry: float
    support: float
    resistance: float
    atr: float
    model_votes: dict[str, Side]


@dataclass
class ContextAssessment:
    decision: Decision
    strategy: Strategy
    size_multiplier: float
    reasons: list[str]


@dataclass
class UncertaintyAssessment:
    decision: Decision
    score: int
    reasons: list[str]


@dataclass
class RiskPlan:
    approved: bool
    units: float
    entry: float
    stop_loss: float
    take_profit: float
    trailing_activation: float
    risk_amount: float
    reward_risk: float
    reason: str = ""


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    side: Side
    units: float
    limit_price: float
    status: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeResult:
    order_id: str
    symbol: str
    pnl: float
    won: bool
    closed_at: datetime


@dataclass
class ReadinessReport:
    ready: bool
    paper_days: int
    trades: int
    win_rate: float
    consecutive_wins: int
    reasons: list[str]
