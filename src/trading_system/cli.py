from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .analyst import DeepAnalyst
from .backtest import ConservativeBacktester
from .context import ContextModulator
from .data import generate_sample_candles, snapshot_from_csv, write_candles_csv
from .execution import PaperExecutor
from .pipeline import TradingPipeline
from .readiness import evaluate_readiness
from .risk import RiskEngine
from .scanner import InitialScanner
from .types import MarketSnapshot, TradeResult
from .uncertainty import UncertaintyEngine


def load_config(path: str = "config/default.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_pipeline(config: dict) -> TradingPipeline:
    scanner_cfg = config["scanner"]
    context_cfg = config["context"]
    risk_cfg = config["risk"]
    execution_cfg = config["execution"]
    return TradingPipeline(
        scanner=InitialScanner(scanner_cfg["top_n"], scanner_cfg["min_history"]),
        analyst=DeepAnalyst(risk_cfg["atr_period"]),
        context=ContextModulator(context_cfg["economic_event_blackout_minutes"], context_cfg["sentiment_extreme_threshold"]),
        uncertainty=UncertaintyEngine(context_cfg["volume_average_window"]),
        risk=RiskEngine(
            daily_loss_limit_pct=risk_cfg["daily_loss_limit_pct"],
            risk_per_trade_pct=risk_cfg["risk_per_trade_pct"],
            minimum_reward_risk=risk_cfg["minimum_reward_risk"],
            stop_atr_multiple=risk_cfg["stop_atr_multiple"],
            trailing_activation_profit_pct=risk_cfg["trailing_activation_profit_pct"],
            max_open_positions=risk_cfg["max_open_positions"],
            max_notional_pct_of_equity=risk_cfg["max_notional_pct_of_equity"],
        ),
        executor=PaperExecutor(execution_cfg["slices"], execution_cfg["slice_interval_seconds"]),
    )


def snapshots_from_dir(data_dir: str) -> list[MarketSnapshot]:
    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return [snapshot_from_csv(path.stem.upper(), path) for path in paths]


def cmd_generate_sample(args: argparse.Namespace) -> None:
    write_candles_csv(generate_sample_candles(args.rows), args.output)
    print(f"Wrote {args.rows} candles to {args.output}")


def cmd_backtest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    report = ConservativeBacktester(build_pipeline(config).risk).run(args.symbol, snapshot_from_csv(args.symbol, args.csv).candles, args.equity)
    print(json.dumps({
        "initial_equity": report.initial_equity,
        "final_equity": report.final_equity,
        "net_pnl": report.net_pnl,
        "trades": len(report.trades),
        "win_rate": report.win_rate,
    }, indent=2))


def cmd_scan(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    snapshots = snapshots_from_dir(args.data_dir)
    scanner = InitialScanner(args.top if args.top is not None else config["scanner"]["top_n"], config["scanner"]["min_history"])
    print(json.dumps([item.__dict__ | {"side": item.side.value} for item in scanner.scan(snapshots)], indent=2))


def cmd_paper_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    snapshots = snapshots_from_dir(args.data_dir)
    for snapshot in snapshots:
        snapshot.sentiment_sources = 2
        snapshot.block_trade_distance_pips = 5.0
        snapshot.related_returns = {"basket": [0.0001] * 50}
    decisions = build_pipeline(config).run(snapshots, args.equity)
    print(json.dumps([
        {
            "symbol": decision.opportunity.symbol,
            "scanner_score": decision.opportunity.score,
            "context": decision.context.decision.value if decision.context else None,
            "uncertainty": decision.uncertainty.decision.value if decision.uncertainty else None,
            "risk_approved": decision.risk.approved if decision.risk else False,
            "orders": len(decision.orders),
        }
        for decision in decisions
    ], indent=2))


def cmd_preflight(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    readiness = config["readiness"]
    report = evaluate_readiness(
        trades=[],
        paper_started_at=datetime.now(timezone.utc) - timedelta(days=0),
        required_paper_days=readiness["required_paper_days"],
        required_trades=readiness["required_trades"],
        minimum_win_rate=readiness["minimum_win_rate"],
        require_consecutive_wins=readiness["require_consecutive_wins"],
    )
    print(json.dumps(report.__dict__, indent=2))
    print("LIVE EXECUTION: DISABLED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first multi-layer trading system")
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("generate-sample")
    sample.add_argument("--output", required=True)
    sample.add_argument("--rows", type=int, default=500)
    sample.set_defaults(func=cmd_generate_sample)
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--csv", required=True)
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--equity", type=float, default=10_000)
    backtest.add_argument("--config", default="config/default.json")
    backtest.set_defaults(func=cmd_backtest)
    scan = sub.add_parser("scan")
    scan.add_argument("--data-dir", required=True)
    scan.add_argument("--top", type=int)
    scan.add_argument("--config", default="config/default.json")
    scan.set_defaults(func=cmd_scan)
    paper = sub.add_parser("paper-run")
    paper.add_argument("--data-dir", required=True)
    paper.add_argument("--equity", type=float, default=10_000)
    paper.add_argument("--config", default="config/default.json")
    paper.set_defaults(func=cmd_paper_run)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", default="config/default.json")
    preflight.set_defaults(func=cmd_preflight)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
