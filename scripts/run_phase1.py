#!/usr/bin/env python3
"""
scripts/run_phase1.py

Downloads two years of 5min Twelve Data for EUR/USD and GBP/JPY using the
TWELVE_DATA_API_KEY environment variable, runs advanced backtest and generates
reports (PDF, report_data.json, equity_curve.png) into reports/backtest_run/<SYMBOL>/.

This script is intended to be run from the repository root (the directory that
contains pyproject.toml). It DOES NOT commit or push any files. Use the Git
commands in the README to add/commit/push only the reports directory.

Exact parameters used (hard-coded):
- symbols: ["EUR/USD", "GBP/JPY"]
- interval: "5min"
- date range: last 2 years from runtime (end = now UTC, start = end - 730 days)
- chunk_days: 14
- sleep_seconds: 1.0
- equity: 10000

"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = REPO_ROOT / "reports" / "backtest_run"
DATA_DIR = REPO_ROOT / "data" / "market"

SYMBOLS = ["EUR/USD", "GBP/JPY"]
INTERVAL = "5min"
CHUNK_DAYS = 14
SLEEP_SECONDS = 1.0
EQUITY = 10000


def iso(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat() + "Z"


def check_env() -> str:
    key = os.environ.get("TWELVE_DATA_API_KEY")
    if not key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY not found in environment. Set it before running the script."
        )
    return key


def run_cmd(cmd: list[str]) -> dict[str, object]:
    """Run a subprocess command and capture stdout/stderr and return code."""
    print("RUN:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.CalledProcessError as exc:
        return {"returncode": exc.returncode, "stdout": exc.stdout, "stderr": exc.stderr}


def symbol_to_filename(symbol: str) -> str:
    return symbol.replace("/", "_").upper() + ".csv"


def main() -> int:
    start_time = datetime.now(timezone.utc)
    key = check_env()

    end = start_time
    start = end - timedelta(days=730)
    start_iso = start.replace(microsecond=0).isoformat()
    end_iso = end.replace(microsecond=0).isoformat()

    os.chdir(REPO_ROOT)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    run_log: dict[str, object] = {
        "start_time": start_time.isoformat(),
        "commands": [],
        "per_symbol": {},
    }

    for symbol in SYMBOLS:
        filename = symbol_to_filename(symbol)
        csv_path = DATA_DIR / filename

        # 1) download
        cmd_download = [
            sys.executable,
            "-m",
            "trading_system.cli",
            "download-twelve",
            "--symbols",
            symbol,
            "--interval",
            INTERVAL,
            "--start-date",
            start_iso,
            "--end-date",
            end_iso,
            "--chunk-days",
            str(CHUNK_DAYS),
            "--sleep-seconds",
            str(SLEEP_SECONDS),
            "--output-dir",
            str(DATA_DIR),
        ]
        out_download = run_cmd(cmd_download)
        run_log["commands"].append({"cmd": cmd_download, "result": {k: out_download.get(k) for k in ("returncode",)}})

        # 2) advanced backtest (prints metrics JSON)
        cmd_ab = [
            sys.executable,
            "-m",
            "trading_system.cli",
            "advanced-backtest",
            "--csv",
            str(csv_path),
            "--symbol",
            symbol.replace("/", "_").upper(),
            "--equity",
            str(EQUITY),
            "--config",
            "config/default.json",
        ]
        out_ab = run_cmd(cmd_ab)
        run_log["per_symbol"][symbol] = {
            "download": {"returncode": out_download.get("returncode")},
            "advanced_backtest": {"returncode": out_ab.get("returncode"), "stdout": out_ab.get("stdout")},
        }

        # 3) report (generates PDF + report_data.json + equity_curve.png)
        report_outdir = OUTPUT_BASE / symbol_to_filename(symbol).replace('.CSV','')
        cmd_report = [
            sys.executable,
            "-m",
            "trading_system.cli",
            "report",
            "--csv",
            str(csv_path),
            "--symbol",
            symbol.replace("/", "_").upper(),
            "--output-dir",
            str(report_outdir),
            "--data-source",
            "Twelve Data CSV",
            "--equity",
            str(EQUITY),
        ]
        out_report = run_cmd(cmd_report)
        run_log["per_symbol"][symbol]["report"] = {"returncode": out_report.get("returncode"), "stdout": out_report.get("stdout")}

    end_time = datetime.now(timezone.utc)
    run_log["end_time"] = end_time.isoformat()

    # list generated files under reports/backtest_run
    generated = []
    for path in sorted(OUTPUT_BASE.rglob("*")):
        if path.is_file():
            rel = path.relative_to(REPO_ROOT)
            generated.append(str(rel))
    run_log["generated_files"] = generated

    # Write metadata/phase1_run.json in reports/backtest_run
    meta_path = OUTPUT_BASE / "phase1_run.json"
    meta_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(f"WROTE METADATA: {meta_path}")

    print("Done. Review reports in:")
    for f in generated:
        print(" -", f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
