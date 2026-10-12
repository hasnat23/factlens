---
title: factlens
emoji: 🔎
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Sentence-level hallucination detection, in your browser
---

# factlens: sentence-level hallucination detection

Paste a source document and an LLM-generated summary (or RAG answer). Every sentence is checked against the source with an NLI model
(`Xenova/nli-deberta-v3-xsmall`, running **in your browser** via transformers.js) and labelled **supported**, **unverified** or
**contradicted**, then aggregated into one factual-consistency score.

This is a browser port of the [factlens](https://github.com/hasnat23/factlens) Python package, which also supports larger models,
batching, a CLI and the RAGTruth benchmark.
