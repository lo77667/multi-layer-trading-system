from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .advanced_backtest import AdvancedBacktestReport


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _pct(value: float) -> str:
    return f"{value:.2%}"


def write_typst_report(report: AdvancedBacktestReport, output_dir: str | Path, symbol: str, data_source: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chart_path = output / "equity_curve.png"
    report.plot_equity_curve(str(chart_path))
    metrics_path = output / "report_data.json"
    metrics_path.write_text(json.dumps(report.metrics(), indent=2), encoding="utf-8")
    winners = sum(trade.won for trade in report.trades)
    losers = len(report.trades) - winners
    stress_count = sum(trade.stress_event is not None for trade in report.trades)
    rows = [
        ("Symbol", symbol),
        ("Data source", data_source),
        ("Initial equity", _money(report.initial_equity)),
        ("Final equity", _money(report.final_equity)),
        ("Buy & hold final equity", _money(report.benchmark_final_equity)),
        ("Trades", str(len(report.trades))),
        ("Winning trades", str(winners)),
        ("Losing trades", str(losers)),
        ("Win rate", _pct(report.win_rate)),
        ("Average duration", f"{report.average_duration_minutes:.1f} minutes"),
        ("Max drawdown", _pct(report.max_drawdown)),
        ("Sharpe ratio", f"{report.sharpe_ratio:.3f}"),
        ("Sortino ratio", f"{report.sortino_ratio:.3f}"),
        ("Stress-window trades", str(stress_count)),
        ("Risk-engine halted days", str(len(report.halted_dates))),
    ]
    table_rows = ",\n".join(f"    [{key}], [{value}]" for key, value in rows)
    main = f'''#import "report-theme.typ": report-accent, report-theme

#show: report-theme.with(
  title: "Advanced Historical Backtest Report",
  author: "Manus AI",
  rhythm: "report",
  running-header: true,
)

#page(margin: (top: 30%, x: 2.2cm), numbering: none, header: none)[
  #set par(first-line-indent: 0em)
  #align(center)[
    #text(size: 25pt, weight: "bold", fill: report-accent)[Advanced Historical Backtest]
    #v(0.6em)
    #text(size: 13pt)[{symbol} — paper-trading research]
    #v(1.8em)
    #text(size: 11pt)[Generated locally from {data_source}]
  ]
]

#page(numbering: none, header: none)[
  #outline(title: [Contents], indent: 1.5em)
]

#counter(page).update(1)

= Executive Summary

This report is a research artifact. It does not authorize live trading and does not imply future performance. The backtest uses chronological information only, conservative stop-first resolution when stop and target overlap, explicit commission, and time-dependent slippage.

= Performance Metrics

#table(
  columns: (1fr, 1fr),
  stroke: 0.4pt + luma(180),
  fill: (col, row) => if row == 0 {{ report-accent }} else if calc.odd(row) {{ luma(245) }} else {{ white }},
  inset: 6pt,
  [*Metric*], [*Value*],
{table_rows}
)

= Equity Curve

#figure(
  image("equity_curve.png", width: 100%),
  caption: [Strategy equity compared with a buy-and-hold benchmark.]
)

= Trade Distribution

The run produced {winners} winning trades and {losers} losing trades. The average holding duration was {report.average_duration_minutes:.1f} minutes. {stress_count} trades were associated with configured stress windows or news-hour slippage.

= Risk Controls

The Risk Engine was evaluated on every entry. It enforced the configured daily loss limit, position risk, maximum open positions, stop distance, and minimum reward-to-risk ratio. Halted dates are reported above; a zero count is not evidence that the limit is unnecessary.

= Data and Reproducibility

The selected source is {data_source}. The report generator stores the machine-readable metrics in `report_data.json` beside this document and stores the equity chart as `equity_curve.png`. API keys, downloaded market files, and live broker credentials must remain outside Git.
'''
    theme_source = Path(__file__).resolve().parents[2] / "reports" / "backtest_report" / "report-theme.typ"
    if not theme_source.exists():
        raise FileNotFoundError("Typst report-theme.typ is missing; run the report template setup first")
    shutil.copy2(theme_source, output / "report-theme.typ")
    main_path = output / "main.typ"
    main_path.write_text(main, encoding="utf-8")
    pdf_path = output / "backtest_report.pdf"
    subprocess.run(["typst", "compile", str(main_path), str(pdf_path)], check=True, capture_output=True, text=True)
    return pdf_path
