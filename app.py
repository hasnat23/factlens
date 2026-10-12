"""Gradio demo: paste a source and an LLM summary, see which sentences are supported."""

import gradio as gr

from factlens import DEFAULT_MODEL, FactLens, NLIScorer

AGGREGATORS = ["gca", "mean", "min", "harmonic"]
scorer = NLIScorer(DEFAULT_MODEL, batch_size=8)
lenses = {name: FactLens(scorer, aggregator=name) for name in AGGREGATORS}

EXAMPLE_SOURCE = (
    "The FBI charged a Philadelphia woman on Thursday with trying to travel overseas to fight for ISIS. "
    "She's one of three women arrested this week on terror charges. Two New York women were also taken into "
    "custody. An FBI complaint cites numerous social media messages dating back to August 2013 that were sent "
    "by Keonna Thomas, 30. In one exchange, a man told Thomas he could help her get to Syria. Thomas purchased "
    "an electronic visa to Turkey on March 23, the complaint says."
)
EXAMPLE_SUMMARY = (
    "A Philadelphia woman was charged by the FBI with trying to travel overseas to fight for ISIS. "
    "Thomas purchased a ticket to Barcelona but was arrested at the airport. "
    "The FBI cited social media messages dating back to 2013 as evidence."
)


def analyse(source: str, summary: str, aggregator: str):
    report = lenses[aggregator].check(source, summary)
    spans, cursor = [], 0
    for s in report.sentences:
        spans.append((summary[cursor : s.start], None))
        spans.append((summary[s.start : s.end], s.label))
        cursor = s.end
    spans.append((summary[cursor:], None))
    rows = [[s.text, s.label, round(s.support, 3), round(s.contradiction, 3), s.evidence] for s in report.sentences]
    verdict = f"### Factual consistency: {report.score:.2f}  ·  {len(report.flagged)} of {len(report.sentences)} sentences flagged"
    return verdict, spans, rows


with gr.Blocks(title="factlens: sentence-level hallucination detection") as demo:
    gr.Markdown(
        "# factlens\nSentence-level hallucination detection for LLM summaries and RAG answers. "
        "Each sentence is checked against the source with an NLI model "
        f"(`{DEFAULT_MODEL}`), then aggregated into one score. "
        "[GitHub](https://github.com/hasnat23/factlens)"
    )
    with gr.Row():
        source = gr.Textbox(label="Source document", lines=12, value=EXAMPLE_SOURCE)
        summary = gr.Textbox(label="Generated summary / answer", lines=12, value=EXAMPLE_SUMMARY)
    aggregator = gr.Radio(AGGREGATORS, value="gca", label="Aggregator")
    button = gr.Button("Check", variant="primary")
    verdict = gr.Markdown()
    highlighted = gr.HighlightedText(
        label="Sentences",
        color_map={"supported": "green", "unverified": "orange", "contradicted": "red"},
    )
    table = gr.Dataframe(headers=["sentence", "label", "support", "max contradiction", "best evidence"], wrap=True)
    button.click(analyse, [source, summary, aggregator], [verdict, highlighted, table])
    demo.load(analyse, [source, summary, aggregator], [verdict, highlighted, table])

if __name__ == "__main__":
    demo.launch()
