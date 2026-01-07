import re


class TextCleaner:
    """Utilities for cleaning and normalizing text."""

    # Regex patterns
    LATEX_PATTERN = re.compile(
        r"<\s*l\s*a\s*t\s*e\s*x\s*i\s*t(?:\s+[^>]*)?>.*?<\s*/\s*l\s*a\s*t\s*e\s*x\s*i\s*t\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    ARXIV_META_PATTERN = re.compile(r"arXiv:\d+\.\d+v\d+", re.IGNORECASE)
    FIGURE_CAPTION_PATTERN = re.compile(r"(Figure|Fig\.)\s*\d+[:].*", re.IGNORECASE)
    BRACKETS_PATTERN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
    SINGLE_LETTER_NOISE_PATTERN = re.compile(r"(?:\b[A-Za-z0-9]\b\s+){2,}\b[A-Za-z0-9]\b")

    @classmethod
    def clean(
            cls,
            text: str,
            *,
            skip_brackets: bool = False,
            skip_latex: bool = False,
            skip_arxiv_meta: bool = False,
            skip_figures: bool = False,
            skip_pdf_noise: bool = False,
    ) -> str:
        """
        Clean text by removing various unwanted patterns.

        Args:
            text: Input text to clean
            skip_brackets: Remove content in brackets/parentheses
            skip_latex: Remove LaTeX blocks
            skip_arxiv_meta: Remove arXiv metadata
            skip_figures: Remove figure captions
            skip_pdf_noise: Remove PDF extraction noise
        Returns:
            Cleaned text
        """
        if skip_latex:
            text = cls.LATEX_PATTERN.sub(" ", text)
        if skip_arxiv_meta:
            text = cls.ARXIV_META_PATTERN.sub(" ", text)
        if skip_figures:
            text = cls.FIGURE_CAPTION_PATTERN.sub(" ", text)
        if skip_brackets:
            text = cls.BRACKETS_PATTERN.sub(" ", text)
        if skip_pdf_noise:
            text = cls.SINGLE_LETTER_NOISE_PATTERN.sub(" ", text)

        # Normalize spacing
        text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def normalize_spacing(text: str) -> str:
        """Ensure proper spacing after sentence-ending punctuation."""
        return re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)

