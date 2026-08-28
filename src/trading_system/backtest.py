from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .indicators import atr, momentum, trend_score
from .risk import RiskEngine
from .types import AnalystSignal, Candle, Side, Strategy, TradeResult


@dataclass
class BacktestReport:
    initial_equity: float
    final_equity: float
    trades: list[TradeResult]

    @property
    def win_rate(self) -> float:
        return sum(trade.won for trade in self.trades) / len(self.trades) if self.trades else 0.0

    @property
    def net_pnl(self) -> float:
        return self.final_equity - self.initial_equity


class ConservativeBacktester:
    def __init__(self, risk: RiskEngine | None = None, fee_rate: float = 0.00002) -> None:
        self.risk = risk or RiskEngine()
        self.fee_rate = fee_rate

    def run(self, symbol: str, candles: list[Candle], initial_equity: float = 10_000.0) -> BacktestReport:
        equity = initial_equity
        daily_pnl = 0.0
        trades: list[TradeResult] = []
        index = 60
        trade_number = 0
        while index < len(candles) - 1:
            history = candles[:index]
            trend = trend_score(history)
            mom = momentum(history)
            side = Side.LONG if trend + mom >= 0 else Side.SHORT
            entry = candles[index].close
            current_atr = atr(history)
            signal = AnalystSignal(
                symbol=symbol,
                side=side,
                confidence=0.5,
                strategy=Strategy.BREAKOUT,
                entry=entry,
                support=min(item.low for item in history[-50:]),
                resistance=max(item.high for item in history[-50:]),
                atr=current_atr,
                model_votes={"trend": side, "momentum": side, "volume": side},
            )
            plan = self.risk.build_plan(signal, equity, daily_pnl, 0)
            if not plan.approved:
                index += 1
                continue
            exit_price = None
            exit_time: datetime | None = None
            won = False
            for future_index in range(index + 1, len(candles)):
                future = candles[future_index]
                if side == Side.LONG:
                    if future.low <= plan.stop_loss:
                        exit_price, won = plan.stop_loss, False
                    elif future.high >= plan.take_profit:
                        exit_price, won = plan.take_profit, True
                else:
                    if future.high >= plan.stop_loss:
                        exit_price, won = plan.stop_loss, False
                    elif future.low <= plan.take_profit:
                        exit_price, won = plan.take_profit, True
                if exit_price is not None:
                    exit_time = future.timestamp
                    index = future_index
                    break
            if exit_price is None:
                break
            gross = plan.units * ((exit_price - entry) if side == Side.LONG else (entry - exit_price))
            fees = (entry + exit_price) * plan.units * self.fee_rate
            pnl = gross - fees
            equity += pnl
            daily_pnl += pnl
            trade_number += 1
            trades.append(TradeResult(f"bt-{trade_number:05d}", symbol, pnl, pnl > 0, exit_time))
            index += 1
        return BacktestReport(initial_equity, equity, trades)
