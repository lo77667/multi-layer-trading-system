from trading_system.risk import RiskEngine
from trading_system.types import AnalystSignal, Side, Strategy


def signal(side=Side.LONG):
    return AnalystSignal(
        symbol="EUR_USD",
        side=side,
        confidence=0.8,
        strategy=Strategy.BREAKOUT,
        entry=1.1000,
        support=1.0900,
        resistance=1.1100,
        atr=0.0010,
        model_votes={"trend": side, "momentum": side, "volume": side},
    )


def test_risk_plan_uses_one_percent_and_three_to_one():
    plan = RiskEngine().build_plan(signal(), equity=10_000, daily_pnl=0, open_positions=0)
    assert plan.approved
    assert plan.risk_amount == 100
    assert plan.reward_risk >= 3
    assert plan.stop_loss < plan.entry < plan.take_profit


def test_daily_loss_limit_blocks_new_trades():
    plan = RiskEngine().build_plan(signal(), equity=10_000, daily_pnl=-200, open_positions=0)
    assert not plan.approved
    assert "Daily loss" in plan.reason


def test_short_stop_and_target_are_on_correct_sides():
    plan = RiskEngine().build_plan(signal(Side.SHORT), equity=10_000, daily_pnl=0, open_positions=0)
    assert plan.approved
    assert plan.take_profit < plan.entry < plan.stop_loss
