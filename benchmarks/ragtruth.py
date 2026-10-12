"""Hallucination detection on the RAGTruth summarization test split (900 human-annotated summaries).

Compares holistic scoring (whole summary as one hypothesis) with factlens'
sentence-level scoring + GCA aggregation, at two levels:

* response level: is there any hallucinated span in the summary?
* sentence level: does this sentence overlap a hallucinated span?

Metric: ROC-AUC and PR-AUC of (1 - score) as a hallucination detector.

    python benchmarks/ragtruth.py --model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score

from factlens import FactLens, LexicalScorer, NLIScorer
from factlens.aggregate import AGGREGATORS


def load_ragtruth(limit: int | None):
    data = load_dataset("wandb/RAGTruth-processed", split="test").filter(lambda r: r["task_type"] == "Summary")
    if limit:
        data = data.select(range(limit))
    return [(r["context"], r["output"], json.loads(r["hallucination_labels"])) for r in data]


def metrics(labels: list[int], risk: list[float]) -> dict:
    return {"roc_auc": round(roc_auc_score(labels, risk), 4), "pr_auc": round(average_precision_score(labels, risk), 4)}


def evaluate(scorer, name: str, rows) -> dict:
    lens = FactLens(scorer, aggregator="mean")
    started = time.perf_counter()
    reports = []
    for i in range(0, len(rows), 100):
        reports += lens.check_many((src, out) for src, out, _ in rows[i : i + 100])
        print(f"  [{name}] scored {len(reports)}/{len(rows)} summaries, {time.perf_counter() - started:.0f}s", flush=True)
    elapsed = time.perf_counter() - started

    response_labels = [int(bool(spans)) for _, _, spans in rows]
    results: dict = {"scorer": name, "n": len(rows), "seconds_per_summary": round(elapsed / len(rows), 3)}

    # Sentence-level aggregations, recomputed from the same sentence scores.
    for agg_name, agg in AGGREGATORS.items():
        risk = [1 - agg([s.support for s in r.sentences]) if r.sentences else 1.0 for r in reports]
        results[f"response/{agg_name}"] = metrics(response_labels, risk)

    sentence_labels, sentence_risk = [], []
    for report, (_, _, spans) in zip(reports, rows):
        for s in report.sentences:
            sentence_labels.append(int(any(s.start < sp["end"] and sp["start"] < s.end for sp in spans)))
            sentence_risk.append(1 - s.support)
    results["sentence"] = {**metrics(sentence_labels, sentence_risk), "n": len(sentence_labels)}

    # Holistic baselines.
    holistic = [1 - lens.holistic(src, out) for src, out, _ in rows]
    print(f"  [{name}] holistic baseline done", flush=True)
    results["response/holistic_best_window"] = metrics(response_labels, holistic)
    if isinstance(scorer, NLIScorer):
        naive = scorer.score_pairs([src for src, _, _ in rows], [out for _, out, _ in rows])
        results["response/holistic_truncated"] = metrics(response_labels, [1 - s.entailment for s in naive])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="NLI model(s); default: base + xsmall")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "ragtruth.json")
    args = parser.parse_args()

    rows = load_ragtruth(args.limit)
    print(f"{len(rows)} summaries, {sum(bool(r[2]) for r in rows)} with hallucinations")
    models = args.model or ["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "cross-encoder/nli-deberta-v3-xsmall"]
    all_results = [evaluate(LexicalScorer(), "lexical-overlap", rows)]
    print(json.dumps(all_results[-1], indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for model in models:
        all_results.append(evaluate(NLIScorer(model), model, rows))
        print(json.dumps(all_results[-1], indent=2), flush=True)
        args.out.write_text(json.dumps(all_results, indent=2))  # save after every model
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
