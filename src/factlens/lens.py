"""Sentence-level factual consistency checking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from .aggregate import get_aggregator
from .scorers import NLIScorer, PairScore, Scorer
from .segment import Sentence, chunk_sentences, split_sentences

Label = Literal["supported", "unverified", "contradicted"]


@dataclass
class SentenceResult:
    text: str
    start: int
    end: int
    support: float
    contradiction: float
    label: Label
    evidence: str


@dataclass
class Report:
    score: float
    aggregator: str
    sentences: list[SentenceResult] = field(default_factory=list)

    @property
    def flagged(self) -> list[SentenceResult]:
        return [s for s in self.sentences if s.label != "supported"]

    def to_dict(self) -> dict:
        return {"score": self.score, "aggregator": self.aggregator, "sentences": [asdict(s) for s in self.sentences]}


@dataclass
class Preference:
    chosen: Literal["a", "b", "tie"]
    score_a: float
    score_b: float

    @property
    def margin(self) -> float:
        return abs(self.score_a - self.score_b)


class FactLens:
    """Checks every sentence of a summary (or any generated text) against a source.

    The source is split into overlapping windows of ``chunk_size`` sentences so
    that long documents fit the NLI model's context. Each summary sentence is
    scored against every window and keeps its best-supporting one, in the
    spirit of SummaC-ZS. Sentence scores are then combined into one score with
    a Granular Credit Assignment aggregator.
    """

    def __init__(
        self,
        scorer: Scorer | None = None,
        aggregator: str = "gca",
        chunk_size: int = 3,
        stride: int = 2,
        support_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
    ) -> None:
        self.scorer = scorer if scorer is not None else NLIScorer()
        self.aggregator = aggregator
        self._aggregate = get_aggregator(aggregator)
        self.chunk_size = chunk_size
        self.stride = stride
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold

    def check(self, source: str, summary: str) -> Report:
        return self.check_many([(source, summary)])[0]

    def check_many(self, pairs: Iterable[tuple[str, str]]) -> list[Report]:
        """Check many (source, summary) pairs with a single batched scorer pass."""
        jobs = []
        premises: list[str] = []
        hypotheses: list[str] = []
        for source, summary in pairs:
            chunks = self._chunks(source)
            claims = split_sentences(summary) if chunks else []
            offset = len(premises)
            for claim in claims:
                premises.extend(chunks)
                hypotheses.extend([claim.text] * len(chunks))
            jobs.append((chunks, claims, offset))

        scores = self.scorer.score_pairs(premises, hypotheses) if premises else []
        return [self._report(chunks, claims, scores[offset:]) for chunks, claims, offset in jobs]

    def holistic(self, source: str, summary: str) -> float:
        """Baseline: score the whole summary as one hypothesis (best window)."""
        chunks = self._chunks(source)
        if not chunks or not summary.strip():
            return 0.0
        scores = self.scorer.score_pairs(chunks, [summary] * len(chunks))
        return max(s.entailment for s in scores)

    def compare(self, source: str, summary_a: str, summary_b: str, margin: float = 0.0) -> Preference:
        """Pick the more factually consistent of two summaries."""
        report_a, report_b = self.check_many([(source, summary_a), (source, summary_b)])
        diff = report_a.score - report_b.score
        chosen = "tie" if abs(diff) <= margin else ("a" if diff > 0 else "b")
        return Preference(chosen, report_a.score, report_b.score)

    def _chunks(self, source: str) -> list[str]:
        return chunk_sentences(split_sentences(source), self.chunk_size, self.stride)

    def _report(self, chunks: Sequence[str], claims: Sequence[Sentence], scores: Sequence[PairScore]) -> Report:
        results = []
        n = len(chunks)
        for i, claim in enumerate(claims):
            window = scores[i * n : (i + 1) * n]
            best = max(range(n), key=lambda j: window[j].entailment)
            support = window[best].entailment
            contradiction = max(s.contradiction for s in window)
            results.append(
                SentenceResult(
                    text=claim.text,
                    start=claim.start,
                    end=claim.end,
                    support=support,
                    contradiction=contradiction,
                    label=self._label(support, contradiction),
                    evidence=chunks[best],
                )
            )
        score = self._aggregate([r.support for r in results]) if results else 0.0
        return Report(score=score, aggregator=self.aggregator, sentences=results)

    def _label(self, support: float, contradiction: float) -> Label:
        if support >= self.support_threshold:
            return "supported"
        if contradiction >= self.contradiction_threshold:
            return "contradicted"
        return "unverified"
