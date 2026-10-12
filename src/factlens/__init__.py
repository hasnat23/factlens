"""factlens: sentence-level factual consistency checking for LLM outputs."""

from .aggregate import AGGREGATORS, gca
from .lens import FactLens, Preference, Report, SentenceResult
from .scorers import DEFAULT_MODEL, FAST_MODEL, LexicalScorer, NLIScorer, PairScore
from .segment import Sentence, split_sentences

__version__ = "0.1.0"

__all__ = [
    "AGGREGATORS",
    "DEFAULT_MODEL",
    "FAST_MODEL",
    "FactLens",
    "LexicalScorer",
    "NLIScorer",
    "PairScore",
    "Preference",
    "Report",
    "Sentence",
    "SentenceResult",
    "gca",
    "split_sentences",
]
