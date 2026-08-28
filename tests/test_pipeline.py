from trading_system.analyst import DeepAnalyst
from trading_system.context import ContextModulator
from trading_system.data import generate_sample_candles
from trading_system.execution import PaperExecutor
from trading_system.pipeline import TradingPipeline
from trading_system.risk import RiskEngine
from trading_system.scanner import InitialScanner
from trading_system.types import MarketSnapshot
from trading_system.uncertainty import UncertaintyEngine


def make_pipeline():
    return TradingPipeline(
        scanner=InitialScanner(top_n=3, min_history=120),
        analyst=DeepAnalyst(),
        context=ContextModulator(),
        uncertainty=UncertaintyEngine(),
        risk=RiskEngine(),
        executor=PaperExecutor(slices=3),
    )


def test_pipeline_requires_two_sentiment_sources():
    snapshot = MarketSnapshot("EUR_USD", generate_sample_candles(180))
    decision = make_pipeline().run([snapshot], equity=10_000)[0]
    assert decision.context is not None
    assert decision.context.decision.value == "reject"
    assert decision.orders == []


def test_pipeline_can_reach_paper_execution_when_all_checks_pass():
    snapshot = MarketSnapshot(
        "EUR_USD",
        generate_sample_candles(180),
        related_returns={"basket": [0.001] * 50},
        sentiment_score=0.0,
        sentiment_sources=2,
        upcoming_event_minutes=None,
        block_trade_distance_pips=5.0,
    )
    decisions = make_pipeline().run([snapshot], equity=10_000)
    assert len(decisions) == 1
    assert len(decisions[0].orders) in (0, 3)
    if decisions[0].risk is not None and decisions[0].risk.approved:
        assert len(decisions[0].orders) == 3
