# 🔎 factlens

[![CI](https://github.com/hasnat23/factlens/actions/workflows/ci.yml/badge.svg)](https://github.com/hasnat23/factlens/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Hugging%20Face-yellow.svg)](https://huggingface.co/spaces/chhasnat23/factlens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Which sentence of your LLM's summary is hallucinated?** factlens checks every sentence of a generated summary or RAG answer
against its source with an NLI model, labels it **supported / unverified / contradicted**, points to the best supporting evidence,
and aggregates everything into one factual-consistency score.

🤗 **[Try the live demo](https://huggingface.co/spaces/chhasnat23/factlens)**: it runs entirely in your browser (transformers.js, `nli-deberta-v3-xsmall`), so no data leaves your machine.

![factlens demo](assets/demo.png)

```python
from factlens import FactLens

lens = FactLens()                       # DeBERTa-v3 NLI, runs on CPU / CUDA / Apple MPS
report = lens.check(source=article, summary=llm_summary)

report.score                            # low score -> at least one sentence is not supported
for s in report.flagged:
    print(s.label, f"{s.support:.2f}", s.text)
# contradicted 0.00 Thomas purchased a ticket to Barcelona but was arrested at the airport.
```

## ✨ Features

- **Sentence-level verdicts** with character offsets (for highlighting) and the best-supporting source window as evidence
- **Long sources** are split into overlapping sentence windows so they fit the NLI context, and each claim keeps its best window (SummaC-style)
- **Contradiction vs. missing support** are distinguished: "the source says otherwise" vs. "the source doesn't say this"
- **Granular Credit Assignment** aggregators turn sentence scores into one score: `gca` (mean pulled toward the weakest sentence), `mean`, `min`, `harmonic`, `soft_min`
- **Pairwise preferences**: `lens.compare(source, a, b)` picks the more faithful of two summaries, e.g. to build preference data for reward models or DPO
- **Any NLI model** from the Hugging Face Hub; label order is read from the model config
- **Batched and length-sorted** scoring across many documents with `check_many`
- **CLI** for files and JSONL batches, plus a **Gradio app**

## 🧠 How it works

```mermaid
flowchart LR
    S[source document] --> W[overlapping<br/>sentence windows]
    G[generated text] --> C[sentences]
    W --> N{NLI model<br/>every sentence × every window}
    C --> N
    N --> B[best window per sentence<br/>P entail · P contradict]
    B --> L[label<br/>supported · unverified · contradicted]
    B --> A[GCA aggregation<br/>→ one score]
```

`gca` combines the average with the weakest sentence: `score = mean(s) · (min(s) / mean(s))^α`. With `α = 0` it is the plain mean;
larger `α` makes a single unsupported sentence pull the whole summary down, which is what you want for detection.

## 📊 Benchmark: RAGTruth (900 human-annotated summaries)

Detecting hallucinations in the 900 summaries of the [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) test split,
which humans annotated at span level. A summary is *hallucinated* if it contains any annotated span (22.7% of summaries); a sentence is
hallucinated if it overlaps one (5.2% of 4,992 sentences). Scores are ROC-AUC and PR-AUC of `1 − score` as a detector, using the default
model `DeBERTa-v3-base-mnli-fever-anli`.

**Response level: is anything in this summary hallucinated?**

| Method | ROC-AUC | PR-AUC |
|---|---:|---:|
| Holistic, source truncated to 512 tokens (naive NLI) | 0.632 | 0.321 |
| Holistic, best source window | 0.580 | 0.299 |
| Lexical overlap baseline (sentence-level, mean) | 0.675 | 0.375 |
| **factlens, sentence-level + `mean`** | **0.696 | 0.366** |
| factlens, sentence-level + `gca` | 0.691 | 0.372 |
| factlens, sentence-level + `min` | 0.682 | 0.370 |
| *random* | 0.500 | 0.227 |

**Sentence level: which sentence is hallucinated?**

| Method | ROC-AUC | PR-AUC |
|---|---:|---:|
| Lexical overlap baseline | 0.685 | 0.086 |
| **factlens (NLI support score)** | **0.793 | 0.182** |
| *random* | 0.500 | 0.052 |

**Takeaways**

- **Scoring sentence by sentence beats scoring the whole summary**: ROC-AUC 0.696 vs 0.580–0.632 with the same NLI model. A long
  summary with one bad sentence still "mostly entails" the source, so holistic scores hide the error.
- **Sentence-level localisation works**: PR-AUC 0.182 is 3.5× the random baseline and about 2× lexical overlap, so the flagged
  sentences really are the suspicious ones.
- **Honest caveats**: RAGTruth is hard. Many annotated hallucinations are subtle additions rather than contradictions, and a lexical-overlap
  baseline is surprisingly competitive at the response level (0.676). On this dataset `mean` aggregation edged out `gca`/`min` for
  ranking whole summaries; `gca` is still the default because it guarantees that one unsupported sentence lowers the score, which is the
  behaviour you want when you act on a threshold.
- Speed: 5.5 s per summary on an 8 GB M1 (MPS), dominated by long source documents (~93 sentence × window pairs per summary).

Reproduce: `pip install -e ".[bench]" && python benchmarks/ragtruth.py` · raw numbers in [`benchmarks/results/ragtruth.json`](benchmarks/results/ragtruth.json).

## 🚀 Installation & usage

```bash
pip install git+https://github.com/hasnat23/factlens
```

```python
from factlens import FactLens, NLIScorer, FAST_MODEL

lens = FactLens(NLIScorer(FAST_MODEL), aggregator="min")    # smaller, faster model
reports = lens.check_many([(src1, sum1), (src2, sum2)])     # batched across documents
pref = lens.compare(source, summary_a, summary_b)           # Preference(chosen="a", score_a=..., score_b=...)
```

**CLI**

```bash
factlens check --source article.txt --summary summary.txt
factlens batch data.jsonl --out scored.jsonl          # lines: {"source": ..., "summary": ...}
```

**Demo apps**

- [`space/index.html`](space/index.html) is the browser port deployed on Hugging Face (static, no server).
- [`app.py`](app.py) is a Gradio app using the full Python package and the larger default model:

```bash
pip install "factlens[app] @ git+https://github.com/hasnat23/factlens" && python app.py
```

## ⚠️ Limitations

- **Cross-window facts**: a claim that combines facts from distant parts of the source (e.g. a name from paragraph 4 and an
  action from paragraph 1) can be marked *unverified* even though it's true. Larger `chunk_size` helps, at the cost of more
  dilution and compute.
- NLI models judge **entailment**, not world knowledge: a summary that adds true background facts is still *unverified*.
- Sentence splitting is rule-based (abbreviations, initials, bullets); unusual formatting can merge or split sentences.

## 🧪 Development

```bash
pip install -e ".[dev]" && pytest                      # tests use a lexical scorer, no model download
pip install -e ".[bench]" && python benchmarks/ragtruth.py
```

## 📄 License

MIT © Muhammad Hasnat
