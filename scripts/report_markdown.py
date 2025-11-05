"""Generate markdown reports summarizing nightly results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate markdown report")
    ap.add_argument("--metrics", default="reports/metrics.json")
    ap.add_argument("--output-dir", default="reports")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file at {metrics_path}")
    metrics = json.loads(metrics_path.read_text())

    report_date = args.date or metrics.get("date") or dt.date.today().isoformat()
    report_path = Path(args.output_dir) / f"nightly_{report_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# Nightly Report {report_date}", "", "## Metrics"]
    for key in sorted(metrics.keys()):
        if key in {"generated_at", "date"}:
            continue
        value = metrics[key]
        if isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        lines.append(f"- **{key}**: {formatted}")
    if "generated_at" in metrics:
        lines.append("")
        lines.append(f"_Generated at {metrics['generated_at']}_")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
