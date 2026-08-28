from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from statistics import mean, pstdev

import matplotlib.pyplot as plt

from .indicators import atr, momentum, trend_score
from .risk import RiskEngine
from .types import AnalystSignal, Candle, Side, Strategy


@dataclass(frozen=True)
class StressWindow:
    name: str
    start: datetime
    end: datetime
    slippage_multiplier: float = 3.0

    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp <= self.end


@dataclass(frozen=True)
class TradingCostModel:
    commission_per_unit: float = 0.00001
    base_slippage_pips: float = 0.2
    pip_size: float = 0.0001
    news_slippage_multiplier: float = 3.0
    stressed_windows: tuple[StressWindow, ...] = ()

    def slippage(self, timestamp: datetime) -> float:
        multiplier = self.news_slippage_multiplier if timestamp.hour in {12, 13, 14} else 1.0
        for window in self.stressed_windows:
            if window.contains(timestamp):
                multiplier = max(multiplier, window.slippage_multiplier)
        return self.base_slippage_pips * self.pip_size * multiplier


@dataclass
class AdvancedTrade:
    trade_id: str
    symbol: str
    side: Side
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    units: float
    pnl: float
    won: bool
    stress_event: str | None = None

    @property
    def duration_minutes(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 60.0


@dataclass
class AdvancedBacktestReport:
    initial_equity: float
    final_equity: float
    equity_curve: list[tuple[datetime, float]]
    trades: list[AdvancedTrade]
    benchmark_final_equity: float
    benchmark_curve: list[tuple[datetime, float]]
    halted_dates: list[date]

    @property
    def returns(self) -> list[float]:
        values = [value for _, value in self.equity_curve]
        return [values[i] / values[i - 1] - 1 for i in range(1, len(values)) if values[i - 1] > 0]

    @property
    def win_rate(self) -> float:
        return mean([trade.won for trade in self.trades]) if self.trades else 0.0

    @property
    def average_duration_minutes(self) -> float:
        return mean([trade.duration_minutes for trade in self.trades]) if self.trades else 0.0

    @property
    def max_drawdown(self) -> float:
        peak = self.initial_equity
        largest = 0.0
        for _, equity in self.equity_curve:
            peak = max(peak, equity)
            largest = max(largest, (peak - equity) / peak if peak else 0.0)
        return largest

    @property
    def sharpe_ratio(self) -> float:
        values = self.returns
        if len(values) < 2 or pstdev(values) == 0:
            return 0.0
        return mean(values) / pstdev(values) * math.sqrt(252)

    @property
    def sortino_ratio(self) -> float:
        values = self.returns
        downside = [value for value in values if value < 0]
        if not values or not downside or pstdev(downside) == 0:
            return 0.0
        return mean(values) / pstdev(downside) * math.sqrt(252)

    def metrics(self) -> dict[str, float | int]:
        return {
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "net_pnl": self.final_equity - self.initial_equity,
            "benchmark_final_equity": self.benchmark_final_equity,
            "trades": len(self.trades),
            "win_rate": self.win_rate,
            "average_duration_minutes": self.average_duration_minutes,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "halted_days": len(self.halted_dates),
        }

    def plot_equity_curve(self, output_path: str) -> None:
        timestamps = [timestamp for timestamp, _ in self.equity_curve]
        values = [value for _, value in self.equity_curve]
        if not timestamps:
            raise ValueError("No equity observations to plot")
        benchmark_timestamps = [timestamp for timestamp, _ in self.benchmark_curve]
        benchmark_values = [value for _, value in self.benchmark_curve]
        fig, axis = plt.subplots(figsize=(10, 5.5))
        axis.plot(timestamps, values, label="Strategy equity", color="#1769aa")
        axis.plot(benchmark_timestamps, benchmark_values, label="Buy & hold benchmark", color="#777777", linestyle="--")
        axis.set_title("Equity Curve vs Buy & Hold")
        axis.set_ylabel("Equity")
        axis.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


class AdvancedBacktester:
    def __init__(self, risk: RiskEngine | None = None, costs: TradingCostModel | None = None) -> None:
        self.risk = risk or RiskEngine()
        self.costs = costs or TradingCostModel()

    def run_sma(self, symbol: str, candles: list[Candle], initial_equity: float = 10_000.0) -> AdvancedBacktestReport:
        if len(candles) < 80:
            raise ValueError("At least 80 candles are required")
        equity = initial_equity
        daily_pnl = 0.0
        current_date: date | None = None
        halted_dates: set[date] = set()
        trades: list[AdvancedTrade] = []
        curve: list[tuple[datetime, float]] = [(candles[0].timestamp, equity)]
        index = 60
        sequence = 0
        while index < len(candles) - 1:
            now = candles[index].timestamp
            if current_date != now.date():
                current_date = now.date()
                daily_pnl = 0.0
            if current_date in halted_dates:
                curve.append((now, equity))
                index += 1
                continue
            history = candles[:index]
            side = Side.LONG if trend_score(history) + momentum(history) >= 0 else Side.SHORT
            signal = AnalystSignal(
                symbol=symbol, side=side, confidence=0.5, strategy=Strategy.BREAKOUT,
                entry=candles[index].close, support=min(c.low for c in history[-50:]),
                resistance=max(c.high for c in history[-50:]), atr=atr(history),
                model_votes={"trend": side, "momentum": side, "volume": side},
            )
            plan = self.risk.build_plan(signal, equity, daily_pnl, 0)
            if not plan.approved:
                index += 1
                continue
            entry_slip = self.costs.slippage(now)
            entry_price = plan.entry + entry_slip if side == Side.LONG else plan.entry - entry_slip
            exit_price = None
            exit_time = None
            won = False
            event_name = None
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
                    for window in self.costs.stressed_windows:
                        if window.contains(future.timestamp) or window.contains(now):
                            event_name = window.name
                    break
            if exit_price is None or exit_time is None:
                break
            exit_slip = self.costs.slippage(exit_time)
            effective_exit = exit_price - exit_slip if side == Side.LONG else exit_price + exit_slip
            gross = plan.units * ((effective_exit - entry_price) if side == Side.LONG else (entry_price - effective_exit))
            costs = plan.units * self.costs.commission_per_unit
            pnl = gross - costs
            equity += pnl
            daily_pnl += pnl
            sequence += 1
            trades.append(AdvancedTrade(
                f"adv-{sequence:05d}", symbol, side, now, exit_time, entry_price, effective_exit, plan.units, pnl, pnl > 0, event_name
            ))
            curve.append((exit_time, equity))
            if daily_pnl <= -(equity * self.risk.daily_loss_limit_pct):
                halted_dates.add(current_date)
            index = candles.index(next(c for c in candles if c.timestamp == exit_time)) + 1
        first = candles[0].close
        last = candles[-1].close
        benchmark_final = initial_equity * last / first
        benchmark_curve = [(candle.timestamp, initial_equity * candle.close / first) for candle in candles]
        return AdvancedBacktestReport(initial_equity, equity, curve, trades, benchmark_final, benchmark_curve, sorted(halted_dates))
