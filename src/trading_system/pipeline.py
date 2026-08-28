from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .analyst import DeepAnalyst
from .context import ContextModulator
from .execution import PaperExecutor
from .risk import RiskEngine
from .scanner import InitialScanner
from .types import Decision, MarketSnapshot, PaperOrder, RiskPlan, ScannerOpportunity, Side
from .uncertainty import UncertaintyEngine


@dataclass
class PipelineDecision:
    opportunity: ScannerOpportunity
    analyst: Any | None
    context: Any | None
    uncertainty: Any | None
    risk: RiskPlan | None
    orders: list[PaperOrder]


@dataclass
class TradingPipeline:
    scanner: InitialScanner
    analyst: DeepAnalyst
    context: ContextModulator
    uncertainty: UncertaintyEngine
    risk: RiskEngine
    executor: PaperExecutor

    def run(
        self,
        snapshots: list[MarketSnapshot],
        equity: float,
        daily_pnl: float = 0.0,
        open_positions: int = 0,
    ) -> list[PipelineDecision]:
        opportunities = self.scanner.scan(snapshots)
        by_symbol = {snapshot.symbol: snapshot for snapshot in snapshots}
        decisions: list[PipelineDecision] = []
        for opportunity in opportunities:
            snapshot = by_symbol[opportunity.symbol]
            analyst_signal = self.analyst.analyze(snapshot, opportunity)
            context_assessment = self.context.assess(snapshot, analyst_signal)
            if context_assessment.decision != Decision.APPROVE:
                decisions.append(PipelineDecision(opportunity, analyst_signal, context_assessment, None, None, []))
                continue
            uncertainty_assessment = self.uncertainty.assess(snapshot, analyst_signal)
            if uncertainty_assessment.decision != Decision.APPROVE:
                decisions.append(PipelineDecision(opportunity, analyst_signal, context_assessment, uncertainty_assessment, None, []))
                continue
            risk_plan = self.risk.build_plan(
                signal=analyst_signal,
                equity=equity,
                daily_pnl=daily_pnl,
                open_positions=open_positions,
                size_multiplier=context_assessment.size_multiplier,
            )
            orders = self.executor.submit_limit_plan(opportunity.symbol, opportunity.side, risk_plan)
            if orders:
                open_positions += 1
            decisions.append(PipelineDecision(opportunity, analyst_signal, context_assessment, uncertainty_assessment, risk_plan, orders))
        return decisions
