from datetime import datetime, timezone

from trading_system.advanced_backtest import AdvancedBacktester, StressWindow, TradingCostModel
from trading_system.data import generate_sample_candles
from trading_system.risk import RiskEngine


def test_cost_model_increases_slippage_inside_stress_window():
    window = StressWindow(
        "shock",
        datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 13, tzinfo=timezone.utc),
        5.0,
    )
    costs = TradingCostModel(stressed_windows=(window,))
    normal = costs.slippage(datetime(2026, 1, 1, 9, tzinfo=timezone.utc))
    stressed = costs.slippage(datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc))
    assert stressed > normal


def test_advanced_backtest_returns_metrics_and_benchmark():
    candles = generate_sample_candles(220)
    report = AdvancedBacktester(RiskEngine()).run_sma("EUR_USD", candles, 10_000)
    metrics = report.metrics()
    assert metrics["initial_equity"] == 10_000
    assert metrics["benchmark_final_equity"] > 0
    assert 0 <= metrics["max_drawdown"] <= 1
    assert len(report.benchmark_curve) == len(candles)
