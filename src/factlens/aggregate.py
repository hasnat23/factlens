"""Granular Credit Assignment (GCA): sentence scores -> one summary score.

``gca`` combines the average sentence score with the weakest one:

    score = mean(s) * (min(s) / mean(s)) ** alpha

With ``alpha=0`` it is the plain mean. Larger ``alpha`` pulls the score towards
the weakest sentence, which is what you want when the goal is *detecting* a
single unsupported sentence in an otherwise good summary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial

Aggregator = Callable[[Sequence[float]], float]


def mean(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores)


def minimum(scores: Sequence[float]) -> float:
    return min(scores)


def gca(scores: Sequence[float], alpha: float = 0.5) -> float:
    avg = mean(scores)
    if alpha == 0 or avg <= 0:
        return avg
    return avg * (min(scores) / avg) ** alpha


def harmonic(scores: Sequence[float], eps: float = 1e-6) -> float:
    return len(scores) / sum(1.0 / max(s, eps) for s in scores)


def soft_min(scores: Sequence[float], temperature: float = 0.1) -> float:
    """Smooth minimum: -T * log(mean(exp(-s / T))). Approaches ``min`` as T -> 0."""
    lowest = min(scores)
    total = sum(math.exp(-(s - lowest) / temperature) for s in scores) / len(scores)
    return lowest - temperature * math.log(total)


AGGREGATORS: dict[str, Aggregator] = {
    "mean": mean,
    "min": minimum,
    "gca": gca,
    "gca_strict": partial(gca, alpha=1.0),
    "harmonic": harmonic,
    "soft_min": soft_min,
}


def get_aggregator(name: str) -> Aggregator:
    try:
        return AGGREGATORS[name]
    except KeyError:
        raise ValueError(f"unknown aggregator {name!r}; choose from {sorted(AGGREGATORS)}") from None
