"""
Claim splitting module for extracting atomic facts from text.

This module provides various strategies for splitting text into atomic claims,
from simple word-based splitting to sophisticated LLM-based extraction.
"""

from abc import ABC, abstractmethod
from typing import List, Sequence, Literal, Dict, Any

import spacy
from spacy.matcher import Matcher

from granuscore.text_cleaner import TextCleaner
from granuscore.utils import ensure_spacy_model


SplitScope = Literal["document", "sentence"]


class UnitSplitter(ABC):
    """
    Abstract base class for unit splitting strategies.

    Defines the interface for extracting referential units from text,
    supporting both single-text and batch processing.
    """

    def __call__(self, batch: List[str], split_scope: SplitScope = "document") -> list[list[dict[str, Any]]]:
        """Allow callable interface for batch processing."""
        return self.get_referential_units_batch(batch, split_scope)

    @abstractmethod
    def get_referential_units(self, text: str, split_scope: SplitScope = "document") -> list[dict[str, Any]]:
        """
        Extract referential units from a single text.

        Args:
            text: Input text to process
            split_scope: Defines the structural scope over inside which referential units should be split.
            "document" splits the whole text.
            "sentence" applies splitting within each sentence.

        Returns:
            List[Dict[str, Any]]
                One entry per group (document or sentence) with:
                - text: str
                - referential_units: List[str]
            """
        pass

    @abstractmethod
    def get_referential_units_batch(self, texts: Sequence[str], split_scope: SplitScope = "document") -> list[list[dict[str, Any]]]:
        """
        Extract referential units from multiple texts.

        Args:
            texts: List of input texts to process
            split_scope: Defines the structural scope over inside which referential units should be split.
            "document" splits the whole text.
            "sentence" applies splitting within each sentence.

        Returns:
            List[List[Dict[str, Any]]]:
                A nested list structure where:
                - the outer list indexes input texts,
                - the second level indexes document- or sentence-level groups
                  (depending on ``split_scope``),
                - each group is represented as a dictionary with the following keys:
                    - ``"text"`` (str):
                        The text span corresponding to the group.
                    - ``"referential_units"`` (List[str]):
                        The extracted referential units for the group.
        """
        pass


class SpacyNounPhraseSplitter(UnitSplitter):
    """
    Spacy-based noun phrase extractor.

    Extracts noun phrases and significant tokens using spaCy's NLP pipeline.
    Includes cleaning options for academic text (LaTeX, figures, etc.).
    """

    def __init__(
            self,
            model: str = "en_core_web_sm",
            skip_brackets: bool = False,
            skip_latex: bool = False,
            skip_arxiv_meta: bool = False,
            skip_figures: bool = False,
            skip_pdf_noise: bool = False,
    ):
        """
        Initialize with spaCy model and cleaning options.

        Args:
            model: spaCy model name to load
            skip_brackets: Remove bracketed content
            skip_latex: Remove LaTeX blocks
            skip_arxiv_meta: Remove arXiv metadata
            skip_figures: Remove figure captions
            skip_pdf_noise: Remove PDF noise
        """
        ensure_spacy_model("en_core_web_sm")
        self.nlp = spacy.load(model)
        self._add_et_al_merging()
        self.clean_kwargs = {
            'skip_brackets': skip_brackets,
            'skip_latex': skip_latex,
            'skip_arxiv_meta': skip_arxiv_meta,
            'skip_figures': skip_figures,
            'skip_pdf_noise': skip_pdf_noise,
        }

    def _add_et_al_merging(self):
        """Add custom component to merge 'et al' into single tokens."""
        matcher = Matcher(self.nlp.vocab)
        matcher.add(
            "ET_AL",
            [[
                {"IS_TITLE": True},
                {"LOWER": "et"},
                {"LOWER": "al"},
            ]]
        )

        @spacy.Language.component("merge_et_al")
        def merge_et_al(doc):
            matches = matcher(doc)
            with doc.retokenize() as retokenizer:
                for _, start, end in matches:
                    retokenizer.merge(doc[start:end])
            return doc

        self.nlp.add_pipe("merge_et_al", before="parser")

    @staticmethod
    def _is_valid_noun_chunk(chunk) -> bool:
        """Check if a noun chunk is valid for extraction."""
        # Must be nominal
        if chunk.root.pos_ not in {"NOUN", "PROPN", "PRON"}:
            return False

        # Reject punctuation-only
        if all(tok.is_punct for tok in chunk):
            return False

        # Reject single hyphen or fragments
        if chunk.text.strip() in {"-", "–", "—"}:
            return False

        return True


    def _extract_from_doclike(self, text) -> list[str]:
        """
        Extract noun phrases and significant tokens from a spaCy Doc or Span.
        Returns a flat list of units for that segment.
        """
        doclike = self.nlp(text)

        spans = {
            (chunk.start, chunk.end)
            for chunk in doclike.noun_chunks
            if self._is_valid_noun_chunk(chunk)
        }

        results: List[str] = []
        i = 0
        while i < len(doclike):
            for start, end in spans:
                if i == start:
                    results.append(doclike[start:end].text)
                    i = end
                    break
            else:
                tok = doclike[i]
                if (not tok.is_stop or tok.like_num) and any(c.isalnum() for c in tok.text):
                    results.append(tok.text)
                i += 1

        return results

    def get_referential_units(self, text: str, split_scope: SplitScope = "document") -> list[dict[str, Any]]:
        """
        Extract noun phrases and significant tokens, optionally grouped by sentence.

        Returns
        -------
        List[Dict[str, Any]]
            One entry per group (document or sentence) with:
            - referential_units: List[str]
            - cleaned_text: str
            - original_text: str
        """
        text = TextCleaner.clean(text, **self.clean_kwargs)

        if split_scope == "document":
            return [{
                "referential_units": self._extract_from_doclike(text),
                "text": text,
            }]

        if split_scope == "sentence":
            doc = self.nlp(text)
            groups: List[Dict[str, Any]] = []
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                groups.append({
                    "referential_units": self._extract_from_doclike(sent_text),
                    "text": sent_text,
                })
            if not groups:
                groups.append({
                    "referential_units": [],
                    "text": text,
                })
            return groups

        raise ValueError('split_scope must be either "document" or "sentence"')

    def get_referential_units_batch(self, texts: Sequence[str], split_scope: SplitScope = 'document') -> list[list[dict[str, Any]]]:
        """Process multiple texts."""
        return [self.get_referential_units(text, split_scope) for text in texts]


class SpacyObjectSplitter(SpacyNounPhraseSplitter):
    DEP_KEEP = {
        "nsubj", "nsubjpass",  # subjects
        "dobj", "obj", "iobj",  # objects
        "obl",  # oblique nominal (where/when/how)
    }

    def _extract_from_doclike(self, text: str) -> list[str]:
        """
        Extract units in document order:
          1) noun chunks (preferred)
          2) otherwise, subject/object-like tokens (dep-based)
        """
        doc = self.nlp(text)

        # spans for noun chunks
        spans = {
            (chunk.start, chunk.end)
            for chunk in doc.noun_chunks
            if self._is_valid_noun_chunk(chunk)
        }
        start_to_end = {start: end for start, end in spans}

        results = []
        seen = set()

        i = 0
        while i < len(doc):
            end = start_to_end.get(i)
            if end is not None:
                text_unit = doc[i:end].text.strip()
                if text_unit and text_unit not in seen:
                    results.append(text_unit)
                    seen.add(text_unit)
                i = end
                continue

            tok = doc[i]
            if tok.dep_ in self.DEP_KEEP and any(c.isalnum() for c in tok.text):
                if tok.text not in seen:
                    results.append(tok.text)
                    seen.add(tok.text)

            i += 1

        return results

if __name__ == '__main__':
    # Example: Using the noun phrase splitter
    splitter = SpacyNounPhraseSplitter(
        skip_brackets=True,
        skip_latex=True,
        skip_figures=True,
        skip_arxiv_meta=True,
        skip_pdf_noise=True
    )

    sample_texts = [
        " O'Reilly",
        "The population is 12,000 (approx.)."
        "three",
        "Two."
    ]

    results = splitter.get_referential_units_batch(sample_texts)

    for i, claims in enumerate(results, 1):
        print(f"\nText {i} claims:")
        for claim in claims:
            print(f"  - {claim}")