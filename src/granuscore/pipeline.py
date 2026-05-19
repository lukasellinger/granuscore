from dataclasses import dataclass
from typing import Iterable, Literal, Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from granuscore.claim_splitter import UnitSplitter, SpacyNounPhraseSplitter
from granuscore.granularity_predictor import (
    HitGranularityPredictor,
    SentenceTransformerGranularityPredictor,
    LLMGranularityPredictor,
    WordNetGranularityPredictor,
    LengthGranularityPredictor,
    PredictorType
)
from granuscore.utils import iter_batches

Pooling = Literal["sum", "mean", "lower_quantile_mean", "min", "max", "weighted_mean"]
PoolingScope = Literal["document", "sentence"]
PREDICTOR_REGISTRY = {
    "hit": HitGranularityPredictor,
    "sentence_transformer": SentenceTransformerGranularityPredictor,
    "llm": LLMGranularityPredictor,
    "wordnet": WordNetGranularityPredictor,
    "length": LengthGranularityPredictor,
}


@dataclass(frozen=True)
class PreSplitText:
    """
    Container for pre-split referential units.

    This can be used to bypass Granuscore's automatic claim splitting
    and directly score a provided decomposition.

    Attributes
    ----------
    units:
        A list of pre-extracted referential units.
    text:
        Original text.
    """
    referential_units: list[str]
    text: str


class GranuScore:
    """
    End-to-end granularity scoring pipeline.

    GranuScore splits text into referential units, assigns a
    continuous granularity score to each unit, and aggregates these
    scores into a single text-level score.
    """

    def __init__(
            self,
            predictor_type: PredictorType = "hit",
            claim_splitter: UnitSplitter | None = None,
            **predictor_kwargs,
    ):
        """
        Initialize the Granuscore pipeline.

        All parameters are optional and only need to be provided if you want
        to customize parts of the pipeline. By default, Granuscore uses the settings from the paper.

        Default configuration
        ---------------------
        - predictor_type: ``"hit"``
        - model_name: ``"Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun"``
        - search_method: ``"random_anchors"``
        - random_anchors_k: ``999``
        - use_cache: ``True``

        Parameters
        ----------
        predictor_type:
            Optional predictor_type.
        model_name:
            Optional model_name.
        claim_splitter:
            Optional component used to split text into referential units.
            Defaults to a spaCy-based noun phrase splitter.
        **predictor_kwargs:
            Additional optional parameters passed to predictor.
        """
        self.claim_splitter = claim_splitter or SpacyNounPhraseSplitter()

        if predictor_type not in PREDICTOR_REGISTRY:
            raise ValueError(f"Unknown predictor_type '{predictor_type}'.")
        else:
            predictor_cls = PREDICTOR_REGISTRY[predictor_type]
        self.granu_predictor = predictor_cls(**predictor_kwargs)
        self.percentile_output = self.granu_predictor.percentile_output

    def __call__(
            self,
            answers: str | list[str] | np.ndarray,
            split: bool = True,
            pooling: Pooling = "mean",
            pooling_scope: PoolingScope = "sentence",
            scope_pooling_method: Pooling = "lower_quantile_mean",
            return_details: bool = False,
            return_buckets: bool = False,
            return_percentiles: bool = True,
            percentile_before_pooling: bool = True,
            batched: bool = True,
            batch_size: int = 128,
            tail_q: float = 0.5,
            scope_tail_q: float = 0.8,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            encoding_batch_size: int | None = 128,
    ) -> str | float | list[float] | list[str] | np.ndarray | torch.Tensor | dict | list[dict]:
        """
        Shorthand for :meth:`predict`.

        Allows calling the instance directly as a function.

        Parameters
        ----------
        answers:
            One or more input texts.
        split:
            Whether to split texts into referential units before scoring.
        pooling:
            Aggregation strategy for referential scores ("mean", "min", "lower_quantile_mean", "sum").
            "lower_quantile_mean" selects the lowest ``tail_q`` fraction of referential scores
            (the lower tail of the distribution) and returns the mean score within this
            subset.
        pooling_scope:
            Defines the structural scope over which referential units are grouped
            prior to aggregation.
            "document" pools over all referential units in the text.
            "sentence" applies the specified ``pooling`` strategy independently
            within each sentence, followed by aggregation over sentence-level
            scores.
        scope_pooling_method:
            Aggregation strategy applied over structural scopes when
            ``pooling_scope`` is not ``"document"``. For example, when
            ``pooling_scope="sentence"``, this parameter controls how
            sentence-level scores are aggregated into a single text-level score.
            Defaults to ``"lower_quantile_mean"``.
        return_details:
            Whether to return detailed per-claim scores.
        return_buckets:
            Whether to output the score as text interpretation. Default False.
        return_percentiles:
            Whether to output the score as in percentiles. Default True.
        percentile_before_pooling:
            Whether you want to work with percentiles before pooling operations. Default True.
        batched:
            If True, process in chunks of batch_size.
        batch_size:
            Chunk size used when batched=True.
        tail_q:
            Lower-tail fraction used by ``pooling="lower_quantile_mean"``. Must be in (0, 1]. Defaults to 0.1.
        scope_tail_q:
            Lower-tail fraction used by ``scope_pooling_method="lower_quantile_mean"``. Must be in (0, 1]. Defaults to 0.8.
        show_progress_bar:
            If True, display a progress bar over input batches.
        convert_to_numpy:
            If True, convert returned scores to NumPy types. For non-detailed output,
            this yields a ``numpy.ndarray`` (or ``numpy.float32`` for single input).
            For detailed output, score values are converted to NumPy scalars.
        convert_to_tensor:
            If True, convert returned scores to PyTorch tensors. For non-detailed
            output, this yields a 1D ``torch.Tensor`` (or 0D tensor for single input).
            For detailed output, score values are converted to scalar tensors.
        encoding_batch_size:
            Maximum number of texts to encode simultaneously in the underlying model.
            If None, no batching limit is applied. Use lower values if experiencing OOM errors.

        Returns
        -------
        str | float | list[float] | list[str] | np.ndarray | torch.tensor | dict | list[dict]
            Granularity score(s), optionally with detailed information.
        """
        return self.predict(
            answers=answers,
            split=split,
            pooling=pooling,
            pooling_scope=pooling_scope,
            scope_pooling_method=scope_pooling_method,
            return_details=return_details,
            return_buckets=return_buckets,
            return_percentiles=return_percentiles,
            percentile_before_pooling=percentile_before_pooling,
            batched=batched,
            batch_size=batch_size,
            tail_q=tail_q,
            scope_tail_q=scope_tail_q,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            encoding_batch_size=encoding_batch_size
        )

    @staticmethod
    def _validate_pooling(pooling: Pooling) -> None:
        if pooling not in {"sum", "mean", "lower_quantile_mean", "min", "max", "weighted_mean"}:
            raise ValueError('pooling must be either "sum", "mean", "weighted_mean", "lower_quantile_mean", "min", or "max"')

    @staticmethod
    def _validate_batching(batched: bool, batch_size: int) -> None:
        if not isinstance(batched, bool):
            raise TypeError("batched must be a bool")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    @staticmethod
    def _validate_encoding_batch_size(encoding_batch_size: int) -> None:
        if isinstance(encoding_batch_size, int) and encoding_batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def _pool(self, scores: Iterable[float], pooling: Pooling, tail_q: float, weights: np.ndarray | None = None) -> float:
        scores_list = list(scores)
        if pooling == "sum":
            return float(sum(scores_list))
        if pooling == "min":
            return float(min(scores_list))
        if pooling == "max":
            return float(max(scores_list))
        if pooling == "lower_quantile_mean":
            if not (0.0 < tail_q <= 1.0):
                raise ValueError("tail_q must be in (0, 1].")
            scores_sorted = sorted(scores_list)
            k = max(1, int(np.ceil(len(scores_sorted) * tail_q)))
            lower_tail = scores_sorted[:k]
            return float(np.mean(lower_tail))
        if pooling == "weighted_mean":
            assert weights is not None
            return float(np.average(scores_list, weights=weights)) if scores_list else float(self.granu_predictor.NO_FACT_SCORE)

        return float(np.mean(scores_list)) if scores_list else float(self.granu_predictor.NO_FACT_SCORE)

    @staticmethod
    def _conversion_functions(convert_to_tensor: bool, convert_to_numpy: bool) -> tuple[Callable, Any]:
        if convert_to_tensor:
            def to_scalar(x: float):
                # 0-d tensor scalar
                return torch.tensor(float(x), dtype=torch.float32)

            def finalize_vector(xs: list[float]):
                return torch.tensor([float(v) for v in xs], dtype=torch.float32)

        elif convert_to_numpy:
            def to_scalar(x: float):
                return np.float32(x)

            def finalize_vector(xs: list[float]):
                return np.asarray(xs, dtype=np.float32)

        else:
            def to_scalar(x: float):
                return float(x)

            def finalize_vector(xs: list[float]):
                return xs
        return to_scalar, finalize_vector

    def _process_batch_of_facts(
            self,
            texts: list[str],
            facts_per_text: list[list[dict[str, Any]]],
            pooling: Pooling,
            scope_pooling_method: Pooling,
            tail_q: float,
            scope_tail_q: float,
            return_details: bool,
            to_scalar: Callable,
            percentile_before_pooling: bool = True,
            encoding_batch_size: int | None = None,
    ) -> tuple[list[float], list[dict]]:
        """
        Process a batch of texts with their referential units in a truly batched manner.

        Parameters
        ----------
        texts:
            Original texts.
        facts_per_text:
            List of lists of fact lists, for each scope inside the texts.
        pooling:
            Aggregation strategy.
        scope_pooling_method:
            Aggregation strategy applied over structural scopes when
            ``pooling_scope`` is not ``"document"``. For example, when
            ``pooling_scope="sentence"``, this parameter controls how
            sentence-level scores are aggregated into a single text-level score.
        tail_q:
            Lower-tail fraction used by ``pooling="lower_quantile_mean"``. Must be in (0, 1]. Defaults to 0.1.
        scope_tail_q:
            Lower-tail fraction used by ``scope_pooling_method="lower_quantile_mean"``. Must be in (0, 1]. Defaults to 0.1.
        return_details:
            Whether to build detailed output dicts.
        to_scalar:
            Conversion function for scalar scores.
        encoding_batch_size:
            Maximum number of texts to encode simultaneously in the underlying model.
            If None, no batching limit is applied. Use lower values if experiencing OOM errors.
        percentile_before_pooling:
            Whether you want to work with percentiles before pooling operations. Default True.

        Returns
        -------
        tuple[list[float], list[dict]]
            Pooled scores and optional detailed dicts.
        """
        # facts_per_text: List[List[List[str]]]  == [text][scope][fact]
        all_facts: list[str] = []
        fact_map: list[tuple[int, int, int]] = []  # (text_idx, scope_idx, fact_pos)

        for text_idx, scopes in enumerate(facts_per_text):
            for scope_idx, scope in enumerate(scopes):
                for fact_pos, fact in enumerate(scope.get('referential_units', [])):
                    all_facts.append(fact)
                    fact_map.append((text_idx, scope_idx, fact_pos))

        # Score all facts in one call (internally batched via encoding_batch_size)
        if all_facts:
            all_scores = list(self.granu_predictor(all_facts, encoding_batch_size=encoding_batch_size))
            if percentile_before_pooling:
                all_scores = self.percentile_output.score_to_percentile(all_scores)
        else:
            all_scores = []

        scores_per_text: list[list[list[float]]] = [
            [[] for _ in scopes] for scopes in facts_per_text
        ]

        for (text_idx, scope_idx, fact_pos), score in zip(fact_map, all_scores):
            scores_per_text[text_idx][scope_idx].append(float(score))

        # Handle scopes with no facts: keep a sentinel in that scope
        for text_idx, scopes in enumerate(scores_per_text):
            for scope_idx, scores in enumerate(scopes):
                if not scores:
                    no_fact_score = float(self.granu_predictor.NO_FACT_SCORE)
                    if percentile_before_pooling:
                        no_fact_score = self.percentile_output.score_to_percentile(no_fact_score)
                    scores_per_text[text_idx][scope_idx] = [no_fact_score]

        # Pool and create outputs
        pooled_scores = []
        detailed = []

        for text, facts_scopes, score_scopes in zip(texts, facts_per_text, scores_per_text):
            weights = np.array([len(scope) for scope in score_scopes])
            scope_pooled_scores = [
                self._pool(scope_scores, pooling, tail_q) for scope_scores in score_scopes
            ]
            pooled = to_scalar(self._pool(scope_pooled_scores, scope_pooling_method, scope_tail_q, weights=weights))
            pooled_scores.append(pooled)

            if return_details:
                scopes_detail = []
                for scope_idx, (scope, scope_scores, scope_pooled) in enumerate(
                        zip(facts_scopes, score_scopes, scope_pooled_scores)
                ):
                    pooled_percentile = scope_pooled if percentile_before_pooling else self.percentile_output.score_to_percentile(scope_pooled)
                    scopes_detail.append(
                        {
                            "scope_index": scope_idx,
                            "scope_text": scope['text'],
                            "pooled_score": to_scalar(scope_pooled),
                            "pooled_percentile": pooled_percentile,
                            "bucket_output": self.percentile_output.bucket(pooled_percentile),
                            "unit_scores": {
                                fact: to_scalar(score)
                                for fact, score in zip(scope['referential_units'], scope_scores)
                            },
                        }
                    )

                detailed.append(
                    {
                        "text": text,
                        "pooled_score": to_scalar(pooled),
                        "pooled_percentile": pooled if percentile_before_pooling else self.percentile_output.score_to_percentile(pooled),
                        "bucket_output": self.percentile_output.bucket(pooled),
                        "scopes": scopes_detail,
                    }
                )

        return pooled_scores, detailed

    def score_to_percentile(self, scores: float | list[float] | np.ndarray) -> float | list[float] | np.ndarray:
        if self.percentile_output:
            return self.percentile_output.score_to_percentile(scores)
        else:
            raise AttributeError("Percentile Output not supported with current Predictor Type and setting.")

    def predict(
            self,
            answers: str | list[str] | np.ndarray,
            split: bool = True,
            pooling: Pooling = "mean",
            pooling_scope: PoolingScope = "sentence",
            scope_pooling_method: Pooling = "lower_quantile_mean",
            return_details: bool = False,
            return_buckets: bool = False,
            return_percentiles: bool = True,
            percentile_before_pooling: bool = True,
            batched: bool = True,
            batch_size: int = 128,
            tail_q: float = 0.5,
            scope_tail_q: float = 0.8,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            encoding_batch_size: int | None = 128,
    ) -> str | float | list[float] | list[str] | np.ndarray | torch.Tensor | dict | list[dict]:
        """
        Predict granularity scores for one or more texts.

        Parameters
        ----------
        answers:
            One or more input texts.
        split:
            Whether to split texts into referential units before scoring.
        pooling:
            Aggregation strategy for referential scores ("mean", "min", "lower_quantile_mean", "sum").
            "lower_quantile_mean" selects the lowest ``tail_q`` fraction of referential scores
            (the lower tail of the distribution) and returns the mean score within this
            subset.
        pooling_scope:
            Defines the structural scope over which referential units are grouped
            prior to aggregation.
            "document" pools over all referential units in the text.
            "sentence" applies the specified ``pooling`` strategy independently
            within each sentence, followed by aggregation over sentence-level
            scores.
        scope_pooling_method:
            Aggregation strategy applied over structural scopes when
            ``pooling_scope`` is not ``"document"``. For example, when
            ``pooling_scope="sentence"``, this parameter controls how
            sentence-level scores are aggregated into a single text-level score.
            Defaults to ``"lower_quantile_mean"``.
        return_details:
            Whether to return detailed per-claim scores.
        return_buckets:
            Whether to output the score as text interpretation. Default False.
        return_percentiles:
            Whether to output the score as in percentiles. Default True.
        percentile_before_pooling:
            Whether you want to work with percentiles before pooling operations. Default True.
        batched:
            If True, process in chunks of batch_size.
        batch_size:
            Chunk size used when batched=True.
        tail_q:
            Lower-tail fraction used by ``pooling="lower_quantile_mean"``. Must be in (0, 1]. Defaults to 0.1.
        scope_tail_q:
            Lower-tail fraction used by ``scope_pooling_method="lower_quantile_mean"``. Must be in (0, 1]. Defaults to 0.8.
        show_progress_bar:
            If True, display a progress bar over input batches.
        convert_to_numpy:
            If True, convert returned scores to NumPy types. For non-detailed output,
            this yields a ``numpy.ndarray`` (or ``numpy.float32`` for single input).
            For detailed output, score values are converted to NumPy scalars.
        convert_to_tensor:
            If True, convert returned scores to PyTorch tensors. For non-detailed
            output, this yields a 1D ``torch.Tensor`` (or 0D tensor for single input).
            For detailed output, score values are converted to scalar tensors.
        encoding_batch_size:
            Maximum number of texts to encode simultaneously in the underlying model.
            If None, no batching limit is applied. Use lower values if experiencing OOM errors.

        Returns
        -------
        str | float | list[float] | list[str] | np.ndarray | torch.tensor | dict | list[dict]
            Granularity score(s), optionally with detailed information.
        """
        self._validate_pooling(pooling)
        self._validate_batching(batched, batch_size)
        self._validate_encoding_batch_size(encoding_batch_size)
        if convert_to_tensor:
            convert_to_numpy = False
        if return_buckets:
            return_percentiles = False
        to_scalar, finalize_vector = self._conversion_functions(convert_to_tensor, convert_to_numpy)
        percentile_before_pooling = percentile_before_pooling and return_percentiles

        if self.percentile_output is None:
            if (
                    return_buckets
                    or return_percentiles
                    or percentile_before_pooling
            ):
                raise ValueError(
                    "The selected predictor does not support percentile-based outputs. "
                    "Disable return_buckets, return_percentiles, "
                    "and percentile_before_pooling."
                )

        is_single_input = isinstance(answers, str)
        if is_single_input:
            answers = [answers]

        if not batched:
            batch_size = len(answers)
        num_batches = (len(answers) + batch_size - 1) // batch_size
        batch_iter = iter_batches(answers, batch_size)
        if show_progress_bar:
            batch_iter = tqdm(
                batch_iter,
                total=num_batches,
                desc="Granuscore",
            )

        total_scores: list[float] = []
        detailed: list[dict] = []

        for batch in batch_iter:
            if not split:
                scores = [float(s) for s in self.granu_predictor(batch, encoding_batch_size=encoding_batch_size)]
                if return_details:
                    for answer, score in zip(batch, scores):
                        detailed.append({
                            "answer": answer,
                            "pooled_score": to_scalar(score),
                            "atomic_scores": {},
                        })
                else:
                    total_scores.extend(scores)
            else:
                facts_per_answer = self.claim_splitter(batch, split_scope=pooling_scope)
                batch_scores, batch_detailed = self._process_batch_of_facts(
                    texts=list(batch),
                    facts_per_text=facts_per_answer,
                    pooling=pooling,
                    percentile_before_pooling=percentile_before_pooling,
                    scope_pooling_method=scope_pooling_method,
                    tail_q=tail_q,
                    scope_tail_q=scope_tail_q,
                    return_details=return_details,
                    to_scalar=to_scalar,
                    encoding_batch_size=encoding_batch_size
                )
                total_scores.extend(batch_scores)
                if return_details:
                    detailed.extend(batch_detailed)

        if return_details:
            output = detailed
        elif return_buckets:
            output = self.percentile_output.bucket(total_scores)
        elif return_percentiles:
            if split:
                output = finalize_vector(total_scores if percentile_before_pooling else self.percentile_output.score_to_percentile(total_scores))
            else:
                output = finalize_vector(self.percentile_output.score_to_percentile(total_scores))
        else:
            output = finalize_vector(total_scores)

        return output[0] if is_single_input else output
