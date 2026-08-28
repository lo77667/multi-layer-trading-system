from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .advanced_backtest import AdvancedBacktester, StressWindow, TradingCostModel
from .analyst import DeepAnalyst
from .backtest import ConservativeBacktester
from .context import ContextModulator
from .data import generate_sample_candles, snapshot_from_csv, write_candles_csv
from .execution import PaperExecutor
from .pipeline import TradingPipeline
from .public_api_sources import TwelveDataClient
from .modeling import XGBoostScannerTrainer
from .reporting import write_typst_report
from .readiness import evaluate_readiness
from .risk import RiskEngine
from .scanner import InitialScanner
from .types import MarketSnapshot
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


def cmd_train_scanner(args: argparse.Namespace) -> None:
    candles = snapshot_from_csv(args.symbol, args.csv).candles
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainer = XGBoostScannerTrainer(horizon=args.horizon, label_threshold=args.label_threshold)
    result = trainer.fit(candles, test_fraction=args.test_fraction)
    model_path = output / "xgboost_scanner.json"
    result.model.save_model(str(model_path))
    importance_path = trainer.plot_top_features(output / "top10_features.png", top_n=10)
    (output / "training_metadata.json").write_text(json.dumps({
        "symbol": args.symbol,
        "best_params": result.best_params,
        "validation_size": result.validation_size,
        "feature_importance": result.feature_importance.to_dict(),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(model_path), "feature_importance_plot": str(importance_path), "best_params": result.best_params}, indent=2))


def cmd_download_twelve(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = TwelveDataClient()
    for symbol in args.symbols:
        if args.start_date and args.end_date:
            candles = client.download_range(symbol, args.start_date, args.end_date, args.interval, args.chunk_days, args.sleep_seconds)
        else:
            candles = client.time_series(symbol, args.interval, args.outputsize)
        filename = symbol.replace("/", "_").upper() + ".csv"
        write_candles_csv(candles, output / filename)
        print(f"Downloaded {len(candles)} candles for {symbol} to {output / filename}")


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


def _stress_windows(values: list[str]) -> tuple[StressWindow, ...]:
    windows: list[StressWindow] = []
    for value in values:
        name, start, end, multiplier = value.split(",", maxsplit=3)
        windows.append(StressWindow(name, datetime.fromisoformat(start), datetime.fromisoformat(end), float(multiplier)))
    return tuple(windows)


def _advanced_report(args: argparse.Namespace):
    config = load_config(args.config)
    risk = build_pipeline(config).risk
    costs = TradingCostModel(
        commission_per_unit=args.commission_per_unit,
        base_slippage_pips=args.base_slippage_pips,
        news_slippage_multiplier=args.news_slippage_multiplier,
        stressed_windows=_stress_windows(args.stress_window),
    )
    candles = snapshot_from_csv(args.symbol, args.csv).candles
    return AdvancedBacktester(risk=risk, costs=costs).run_sma(args.symbol, candles, args.equity)


def cmd_advanced_backtest(args: argparse.Namespace) -> None:
    report = _advanced_report(args)
    print(json.dumps(report.metrics(), indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    report = _advanced_report(args)
    pdf = write_typst_report(report, args.output_dir, args.symbol, args.data_source)
    print(json.dumps({"pdf": str(pdf), **report.metrics()}, indent=2))


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
        paper_started_at=datetime.now(timezone.utc),
        required_paper_days=readiness["required_paper_days"],
        required_trades=readiness["required_trades"],
        minimum_win_rate=readiness["minimum_win_rate"],
        require_consecutive_wins=readiness["require_consecutive_wins"],
    )
    print(json.dumps(report.__dict__, indent=2))
    print("LIVE EXECUTION: DISABLED")


def _add_cost_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--equity", type=float, default=10_000)
    parser.add_argument("--commission-per-unit", type=float, default=0.00001)
    parser.add_argument("--base-slippage-pips", type=float, default=0.2)
    parser.add_argument("--news-slippage-multiplier", type=float, default=3.0)
    parser.add_argument("--stress-window", action="append", default=[], help="name,start_iso,end_iso,slippage_multiplier")
    parser.add_argument("--config", default="config/default.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first multi-layer trading system")
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("generate-sample")
    sample.add_argument("--output", required=True)
    sample.add_argument("--rows", type=int, default=500)
    sample.set_defaults(func=cmd_generate_sample)
    train = sub.add_parser("train-scanner")
    train.add_argument("--csv", required=True)
    train.add_argument("--symbol", required=True)
    train.add_argument("--output-dir", default="models/scanner")
    train.add_argument("--horizon", type=int, default=8)
    train.add_argument("--label-threshold", type=float, default=0.0005)
    train.add_argument("--test-fraction", type=float, default=0.2)
    train.set_defaults(func=cmd_train_scanner)
    download = sub.add_parser("download-twelve")
    download.add_argument("--symbols", nargs="+", default=["EUR/USD", "GBP/JPY"])
    download.add_argument("--interval", default="5min")
    download.add_argument("--outputsize", type=int, default=5000)
    download.add_argument("--start-date")
    download.add_argument("--end-date")
    download.add_argument("--chunk-days", type=int, default=14)
    download.add_argument("--sleep-seconds", type=float, default=1.0)
    download.add_argument("--output-dir", default="data/market")
    download.set_defaults(func=cmd_download_twelve)
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--csv", required=True)
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--equity", type=float, default=10_000)
    backtest.add_argument("--config", default="config/default.json")
    backtest.set_defaults(func=cmd_backtest)
    advanced = sub.add_parser("advanced-backtest")
    advanced.add_argument("--csv", required=True)
    advanced.add_argument("--symbol", required=True)
    _add_cost_args(advanced)
    advanced.set_defaults(func=cmd_advanced_backtest)
    report = sub.add_parser("report")
    report.add_argument("--csv", required=True)
    report.add_argument("--symbol", required=True)
    report.add_argument("--output-dir", default="reports/backtest_run")
    report.add_argument("--data-source", default="Twelve Data CSV")
    _add_cost_args(report)
    report.set_defaults(func=cmd_report)
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
