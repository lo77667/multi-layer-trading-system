from __future__ import annotations

from datetime import datetime, timezone

from .types import ReadinessReport, TradeResult


def evaluate_readiness(
    trades: list[TradeResult],
    paper_started_at: datetime,
    required_paper_days: int = 90,
    required_trades: int = 60,
    minimum_win_rate: float = 0.55,
    require_consecutive_wins: bool = True,
) -> ReadinessReport:
    now = datetime.now(timezone.utc)
    start = paper_started_at if paper_started_at.tzinfo else paper_started_at.replace(tzinfo=timezone.utc)
    paper_days = max(0, (now - start).days)
    wins = sum(trade.won for trade in trades)
    win_rate = wins / len(trades) if trades else 0.0
    consecutive = 0
    for trade in reversed(trades):
        if not trade.won:
            break
        consecutive += 1
    reasons: list[str] = []
    if paper_days < required_paper_days:
        reasons.append(f"Need {required_paper_days - paper_days} more paper-trading days")
    if len(trades) < required_trades:
        reasons.append(f"Need {required_trades - len(trades)} more trades")
    if win_rate < minimum_win_rate:
        reasons.append(f"Win rate {win_rate:.2%} is below {minimum_win_rate:.2%}")
    if require_consecutive_wins and consecutive < required_trades:
        reasons.append("The consecutive-win gate is not satisfied")
    ready = not reasons
    if ready:
        reasons.append("Paper-trading gates are satisfied; live execution still requires separate human approval")
    return ReadinessReport(ready, paper_days, len(trades), win_rate, consecutive, reasons)
