"""Premise/hypothesis scorers.

A scorer takes aligned lists of premises and hypotheses and returns, for each
pair, the probability that the premise entails the hypothesis and the
probability that it contradicts it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
FAST_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


@dataclass(frozen=True)
class PairScore:
    entailment: float
    contradiction: float


class Scorer(Protocol):
    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[PairScore]: ...


class NLIScorer:
    """Scores pairs with any Hugging Face NLI sequence classifier.

    Label indices are read from the model config, so models with different
    label orders (e.g. ``entailment, neutral, contradiction`` vs
    ``contradiction, entailment, neutral``) work without configuration.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or _default_device(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(self.device)
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        try:
            self._entail = next(i for name, i in labels.items() if name.startswith("entail"))
            self._contra = next(i for name, i in labels.items() if name.startswith("contra"))
        except StopIteration as exc:
            raise ValueError(f"{model_name} does not look like an NLI model: {labels}") from exc

    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[PairScore]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        # Batch pairs of similar length together to minimise padding, then restore order.
        order = sorted(range(len(premises)), key=lambda i: len(premises[i]) + len(hypotheses[i]))
        out: list[PairScore | None] = [None] * len(premises)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = self.tokenizer(
                [premises[i] for i in idx],
                [hypotheses[i] for i in idx],
                return_tensors="pt",
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
            ).to(self.device)
            with self._torch.no_grad():
                probs = self.model(**batch).logits.float().softmax(-1).cpu()
            for i, p in zip(idx, probs):
                out[i] = PairScore(float(p[self._entail]), float(p[self._contra]))
        return out  # type: ignore[return-value]


class LexicalScorer:
    """Dependency-free baseline: fraction of hypothesis content words found in the premise.

    Useful for tests and as a sanity baseline in benchmarks. It cannot detect
    contradictions, so ``contradiction`` is always 0.
    """

    _STOP: ClassVar[frozenset[str]] = frozenset(
        [
            "a",
            "an",
            "the",
            "of",
            "to",
            "in",
            "on",
            "at",
            "for",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "by",
            "with",
            "as",
            "that",
            "this",
            "it",
            "its",
            "from",
            "has",
            "have",
            "had",
            "he",
            "she",
            "they",
            "his",
            "her",
            "their",
            "which",
            "who",
        ]
    )

    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[PairScore]:
        return [PairScore(self._overlap(p, h), 0.0) for p, h in zip(premises, hypotheses, strict=True)]

    def _overlap(self, premise: str, hypothesis: str) -> float:
        words = [w for w in _tokens(hypothesis) if w not in self._STOP]
        if not words:
            return 1.0
        vocab = set(_tokens(premise))
        return sum(w in vocab for w in words) / len(words)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _default_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
