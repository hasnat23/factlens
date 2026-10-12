---
title: factlens
emoji: 🔎
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Sentence-level hallucination detection for LLM summaries
---

# factlens: sentence-level hallucination detection

Paste a source document and an LLM-generated summary (or RAG answer). Every sentence is checked against the source with an NLI model and labelled **supported**, **unverified** or **contradicted**, then aggregated into one factual-consistency score.

Code, benchmark and Python package: [github.com/hasnat23/factlens](https://github.com/hasnat23/factlens)
