"""Command-line interface.

factlens check --source article.txt --summary summary.txt
factlens batch input.jsonl --out results.jsonl   # lines: {"source": ..., "summary": ...}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aggregate import AGGREGATORS
from .lens import FactLens
from .scorers import DEFAULT_MODEL, NLIScorer

_COLORS = {"supported": "\033[32m", "unverified": "\033[33m", "contradicted": "\033[31m"}
_RESET = "\033[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="factlens", description="Sentence-level factual consistency checks.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face NLI model")
    parser.add_argument("--aggregator", default="gca", choices=sorted(AGGREGATORS))
    parser.add_argument("--device", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check one source/summary pair")
    check.add_argument("--source", required=True, type=Path)
    check.add_argument("--summary", required=True, type=Path)
    check.add_argument("--json", action="store_true", help="print the full report as JSON")

    batch = sub.add_parser("batch", help="check a JSONL file of {source, summary} records")
    batch.add_argument("input", type=Path)
    batch.add_argument("--out", type=Path, default=None, help="output JSONL (default: stdout)")

    args = parser.parse_args(argv)
    lens = FactLens(NLIScorer(args.model, device=args.device), aggregator=args.aggregator)

    if args.command == "check":
        report = lens.check(args.source.read_text(), args.summary.read_text())
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            for s in report.sentences:
                color = _COLORS[s.label] if sys.stdout.isatty() else ""
                print(f"{color}[{s.label:>12} {s.support:.2f}]{_RESET if color else ''} {s.text}")
            print(f"\nscore ({report.aggregator}): {report.score:.3f}   flagged: {len(report.flagged)}")
        return 0

    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    reports = lens.check_many((r["source"], r["summary"]) for r in records)
    out = args.out.open("w") if args.out else sys.stdout
    for record, report in zip(records, reports):
        out.write(json.dumps({**record, "factlens": report.to_dict()}) + "\n")
    if args.out:
        out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
