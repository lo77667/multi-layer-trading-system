from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .types import PaperOrder, RiskPlan, Side


class LiveTradingDisabled(RuntimeError):
    """Raised whenever code attempts to place a live order in this repository."""


@dataclass
class PaperExecutor:
    slices: int = 3
    slice_interval_seconds: int = 60
    orders: list[PaperOrder] = field(default_factory=list)

    def submit_limit_plan(self, symbol: str, side: Side, plan: RiskPlan) -> list[PaperOrder]:
        if not plan.approved:
            return []
        count = max(1, min(self.slices, 4))
        units_per_slice = plan.units / count
        now = datetime.now(timezone.utc)
        created: list[PaperOrder] = []
        for index in range(count):
            order = PaperOrder(
                order_id=f"paper-{uuid.uuid4().hex[:12]}",
                symbol=symbol,
                side=side,
                units=units_per_slice,
                limit_price=plan.entry,
                status="simulated_pending",
                created_at=now,
                metadata={
                    "slice": index + 1,
                    "slice_count": count,
                    "scheduled_offset_seconds": index * self.slice_interval_seconds,
                    "stop_loss": plan.stop_loss,
                    "take_profit": plan.take_profit,
                    "trailing_activation": plan.trailing_activation,
                },
            )
            created.append(order)
        self.orders.extend(created)
        return created

    def fill_all_at_limit(self) -> None:
        for order in self.orders:
            if order.status == "simulated_pending":
                order.status = "simulated_filled"


class LiveExecutionGuard:
    def submit(self, *_args, **_kwargs) -> None:
        raise LiveTradingDisabled(
            "Live execution is intentionally disabled. Complete independent validation and build a reviewed broker adapter."
        )
