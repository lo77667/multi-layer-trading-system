from __future__ import annotations

import math
from dataclasses import dataclass

from .types import AnalystSignal, RiskPlan, Side


@dataclass
class RiskEngine:
    daily_loss_limit_pct: float = 0.02
    risk_per_trade_pct: float = 0.01
    minimum_reward_risk: float = 3.0
    stop_atr_multiple: float = 1.5
    trailing_activation_profit_pct: float = 0.01
    max_open_positions: int = 3
    max_notional_pct_of_equity: float = 25.0

    def build_plan(
        self,
        signal: AnalystSignal,
        equity: float,
        daily_pnl: float,
        open_positions: int,
        size_multiplier: float = 1.0,
    ) -> RiskPlan:
        if equity <= 0:
            return RiskPlan(False, 0.0, signal.entry, signal.entry, signal.entry, signal.entry, 0.0, 0.0, "Equity must be positive")
        if daily_pnl <= -(equity * self.daily_loss_limit_pct):
            return RiskPlan(False, 0.0, signal.entry, signal.entry, signal.entry, signal.entry, 0.0, 0.0, "Daily loss limit reached")
        if open_positions >= self.max_open_positions:
            return RiskPlan(False, 0.0, signal.entry, signal.entry, signal.entry, signal.entry, 0.0, 0.0, "Maximum open positions reached")
        if size_multiplier <= 0:
            return RiskPlan(False, 0.0, signal.entry, signal.entry, signal.entry, signal.entry, 0.0, 0.0, "Context did not approve sizing")

        distance = signal.atr * self.stop_atr_multiple
        if distance <= 0 or not math.isfinite(distance):
            return RiskPlan(False, 0.0, signal.entry, signal.entry, signal.entry, signal.entry, 0.0, 0.0, "Invalid ATR-derived stop distance")
        risk_amount = equity * self.risk_per_trade_pct * min(size_multiplier, 1.0)
        units = risk_amount / distance
        notional_cap = equity * self.max_notional_pct_of_equity / 100.0
        units = min(units, notional_cap / signal.entry)
        if units <= 0 or not math.isfinite(units):
            return RiskPlan(False, 0.0, signal.entry, signal.entry, signal.entry, signal.entry, 0.0, 0.0, "Calculated units are invalid")

        if signal.side == Side.LONG:
            stop = signal.entry - distance
            target = signal.entry + distance * self.minimum_reward_risk
            trailing = signal.entry * (1 + self.trailing_activation_profit_pct)
        else:
            stop = signal.entry + distance
            target = signal.entry - distance * self.minimum_reward_risk
            trailing = signal.entry * (1 - self.trailing_activation_profit_pct)
        reward_risk = round(abs(target - signal.entry) / abs(signal.entry - stop), 8)
        if reward_risk + 1e-9 < self.minimum_reward_risk:
            return RiskPlan(False, 0.0, signal.entry, stop, target, trailing, risk_amount, reward_risk, "Reward/risk below hard minimum")
        return RiskPlan(True, units, signal.entry, stop, target, trailing, risk_amount, reward_risk)
