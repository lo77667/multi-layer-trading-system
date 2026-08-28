from datetime import datetime, timedelta, timezone

import pytest

from trading_system.execution import LiveExecutionGuard, LiveTradingDisabled, PaperExecutor
from trading_system.readiness import evaluate_readiness
from trading_system.types import AnalystSignal, RiskPlan, Side, Strategy, TradeResult


def test_live_execution_is_hard_disabled():
    with pytest.raises(LiveTradingDisabled):
        LiveExecutionGuard().submit("EUR_USD", 1)


def test_paper_plan_is_split_into_three_orders():
    plan = RiskPlan(True, 900, 1.1, 1.0985, 1.1045, 1.111, 100, 3.0)
    orders = PaperExecutor(slices=3).submit_limit_plan("EUR_USD", Side.LONG, plan)
    assert len(orders) == 3
    assert sum(order.units for order in orders) == plan.units
    assert all(order.status == "simulated_pending" for order in orders)


def test_readiness_requires_all_gates():
    now = datetime.now(timezone.utc)
    trades = [TradeResult(str(i), "EUR_USD", 1.0, True, now) for i in range(60)]
    report = evaluate_readiness(trades, now - timedelta(days=90))
    assert report.ready
    assert report.win_rate == 1.0
    assert report.consecutive_wins == 60
