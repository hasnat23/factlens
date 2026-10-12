import math

import pytest

from factlens import FactLens, LexicalScorer, PairScore, split_sentences
from factlens.aggregate import gca, get_aggregator, harmonic, soft_min
from factlens.segment import chunk_sentences


class TestSegmentation:
    def test_basic_split_keeps_offsets(self):
        text = "The FBI charged a woman. She was arrested on Thursday! Was she alone?"
        sents = split_sentences(text)
        assert [s.text for s in sents] == [
            "The FBI charged a woman.",
            "She was arrested on Thursday!",
            "Was she alone?",
        ]
        for s in sents:
            assert text[s.start : s.end] == s.text

    def test_abbreviations_and_initials(self):
        text = "Dr. Smith met J. K. Rowling in the U.S. on Monday. They talked."
        assert len(split_sentences(text)) == 2

    def test_lowercase_continuation_is_not_a_boundary(self):
        assert len(split_sentences("Prices rose 3.5 percent. in total, it was a lot.")) == 1

    def test_paragraphs_and_bullets(self):
        text = "Intro line\n\n- first point\n- second point"
        assert [s.text for s in split_sentences(text)] == ["Intro line", "- first point", "- second point"]

    def test_empty(self):
        assert split_sentences("   ") == []

    def test_chunking_covers_all_sentences(self):
        sents = split_sentences(" ".join(f"Sentence {i} here." for i in range(7)))
        chunks = chunk_sentences(sents, size=3, stride=2)
        assert chunks[0].startswith("Sentence 0") and chunks[-1].endswith("Sentence 6 here.")


class TestAggregation:
    def test_gca_alpha_zero_is_mean(self):
        assert gca([0.2, 0.8], alpha=0) == pytest.approx(0.5)

    def test_gca_alpha_one_is_min(self):
        assert gca([0.2, 0.8], alpha=1.0) == pytest.approx(0.2)

    def test_gca_penalises_one_bad_sentence(self):
        assert gca([0.9, 0.9, 0.9, 0.1]) < sum([0.9, 0.9, 0.9, 0.1]) / 4

    def test_soft_min_between_min_and_mean(self):
        s = [0.1, 0.9]
        assert 0.1 <= soft_min(s) <= 0.5
        assert math.isclose(soft_min(s, temperature=1e-4), 0.1, abs_tol=1e-3)

    def test_harmonic(self):
        assert harmonic([0.5, 0.5]) == pytest.approx(0.5)

    def test_unknown_aggregator(self):
        with pytest.raises(ValueError):
            get_aggregator("median-ish")


SOURCE = (
    "The city council approved a new park on Tuesday. The park will cost 2 million dollars. "
    "Construction starts in May. The mayor said the park would open next year."
)


class TestFactLens:
    def setup_method(self):
        self.lens = FactLens(LexicalScorer(), aggregator="min")

    def test_flags_unsupported_sentence(self):
        report = self.lens.check(SOURCE, "The council approved a park. Aliens funded the stadium.")
        assert [s.label for s in report.sentences] == ["supported", "unverified"]
        assert report.flagged[0].text == "Aliens funded the stadium."
        assert report.score < 0.5

    def test_evidence_points_to_best_window(self):
        report = self.lens.check(SOURCE, "Construction starts in May.")
        assert "Construction starts in May." in report.sentences[0].evidence

    def test_compare_prefers_faithful_summary(self):
        pref = self.lens.compare(SOURCE, "The park will cost 2 million dollars.", "The park will cost nothing at all.")
        assert pref.chosen == "a" and pref.margin > 0

    def test_check_many_matches_check(self):
        pairs = [(SOURCE, "The mayor spoke."), (SOURCE, "The park opens next year.")]
        assert [r.score for r in self.lens.check_many(pairs)] == [self.lens.check(*p).score for p in pairs]

    def test_empty_inputs(self):
        assert self.lens.check("", "Anything.").sentences == []
        assert self.lens.check(SOURCE, "").score == 0.0

    def test_contradiction_label(self):
        class Contradicts:
            def score_pairs(self, premises, hypotheses):
                return [PairScore(0.05, 0.9) for _ in premises]

        report = FactLens(Contradicts()).check(SOURCE, "The park was rejected.")
        assert report.sentences[0].label == "contradicted"
