from dataclasses import dataclass
from typing import Iterable, Literal, Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from granuscore import HitGranularityPredictor
from granuscore.bucket_output import PercentileOutput
from granuscore.claim_splitter import UnitSplitter, SpacyNounPhraseSplitter
from granuscore.artifcats import ArtifactSpec, ArtifactManager
from granuscore.granularity_predictor import SearchMethod
from granuscore.utils import iter_batches

DEFAULT_FAISS_INDEX = ArtifactSpec(
    name="50k-hit-index.faiss",
    subdir='faiss',
    sha256="02bcb37669bf68f7859ea16e6aa7c85e81eacf2632a944eb94113d5c751d28b6",
)

DEFAULT_ANCHORS = ArtifactSpec(
    name="50k-hit-random_anchors_999.npy",
    subdir='faiss',
    sha256="92432aabbec27507a71d56595bdd0e8c6ca5aeedafd13ed70af68b01de7e9436",
)

DEFAULT_LGB_MODEL = ArtifactSpec(
    name="50k-hit-random_anchors-999_model.txt",
    subdir='lgb_models',
    sha256="5aba55b9e13c09d86561b50773d77e5c21c0a7f7c7092473537e28c247d451b9",
)

DEFAULT_NOUN_SCORES = ArtifactSpec(
    name="scores-wordnet_nouns.npy",
    subdir='references',
    sha256="487a2c46afd74af3875058fb2115a5a1dc68d48a7ebde2e7aabeb3b4e2551a53",
)

Pooling = Literal["sum", "mean", "lower_quantile_mean", "min", "max", "weighted_mean"]
PoolingScope = Literal["document", "sentence"]


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
    NO_FACT_SCORE = 5 # higher than rest -> gets set to 100 percentile

    def __init__(
            self,
            hierarchy_model: str = "Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun",
            faiss_index_path: str | None = None,
            lgb_model_path: str | None = None,
            search_method: SearchMethod = 'random_anchors',
            reference_scores_path: str | None = None,
            anchor_path: str | None = None,
            claim_splitter: UnitSplitter | None = None,
            default_faiss_index: ArtifactSpec = DEFAULT_FAISS_INDEX,
            default_lgb_model: ArtifactSpec = DEFAULT_LGB_MODEL,
            default_reference_scores: ArtifactSpec = DEFAULT_NOUN_SCORES,
            default_anchors: ArtifactSpec = DEFAULT_ANCHORS,
    ):
        """
        Initialize the Granuscore pipeline.

        All parameters are optional and only need to be provided if you want
        to customize parts of the pipeline. By default, Granuscore uses the settings from the paper.

        Parameters
        ----------
        hierarchy_model:
            Optional name or path of the hierarchical sentence embedding model.
            Defaults to a general-purpose WordNet-based model.
        faiss_index_path:
            Optional path to a custom FAISS index. If not provided, a default
            index is automatically downloaded and cached.
        lgb_model_path:
            Optional path to a custom LightGBM model. If not provided, a default
            model is automatically downloaded and cached.
        search_method:
            Method used to gather input data into lgb model. Must match the lgb_model.
            For default lgb model: 'radial_anchors'.
        claim_splitter:
            Optional component used to split text into referential units.
            Defaults to a spaCy-based noun phrase splitter.
        default_faiss_index:
            Artifact specification for the default FAISS index. This only needs
            to be overridden for advanced use cases.
        default_lgb_model:
            Artifact specification for the default LightGBM model. This only needs
            to be overridden for advanced use cases.
        """
        manager = ArtifactManager()
        faiss_index_path = faiss_index_path or str(manager.ensure(default_faiss_index))
        lgb_model_path = lgb_model_path or str(manager.ensure(default_lgb_model))
        reference_scores_path = reference_scores_path or str(manager.ensure(default_reference_scores))
        anchor_path = anchor_path or str(manager.ensure(default_anchors))

        self.granu_predictor = HitGranularityPredictor(
            hierarchy_model, faiss_index_path, lgb_model_path, search_method
        )
        self.claim_splitter = claim_splitter or SpacyNounPhraseSplitter()
        self.percentile_output = PercentileOutput(reference_scores_path)

    def __call__(
            self,
            answers: str | list[str] | np.ndarray,
            split: bool = True,
            pooling: Pooling = "mean",
            pooling_scope: PoolingScope = "sentence",
            scope_pooling_method: Pooling = "lower_quantile_mean",
            detailed_infos: bool = False,
            bucket_output: bool = False,
            percentile_output: bool = True,
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
        detailed_infos:
            Whether to return detailed per-claim scores.
        bucket_output:
            Whether to output the score as text interpretation. Default False.
        percentile_output:
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
            detailed_infos=detailed_infos,
            bucket_output=bucket_output,
            percentile_output=percentile_output,
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
        if pooling not in {"sum", "mean", "lower_quantile_mean", "min", "max"}:
            raise ValueError('pooling must be either "sum", "mean", "lower_quantile_mean" or "min", "max"')

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
            return float(np.average(scores_list, weights=weights)) if scores_list else float(self.NO_FACT_SCORE)

        return float(np.mean(scores_list)) if scores_list else float(self.NO_FACT_SCORE)

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
            detailed_infos: bool,
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
        detailed_infos:
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
                    no_fact_score = float(self.NO_FACT_SCORE)
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

            if detailed_infos:
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
        return self.percentile_output.score_to_percentile(scores)

    def predict(
            self,
            answers: str | list[str] | np.ndarray,
            split: bool = True,
            pooling: Pooling = "mean",
            pooling_scope: PoolingScope = "sentence",
            scope_pooling_method: Pooling = "lower_quantile_mean",
            detailed_infos: bool = False,
            bucket_output: bool = False,
            percentile_output: bool = True,
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
        detailed_infos:
            Whether to return detailed per-claim scores.
        bucket_output:
            Whether to output the score as text interpretation. Default False.
        percentile_output:
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
        if bucket_output:
            percentile_output = False
        to_scalar, finalize_vector = self._conversion_functions(convert_to_tensor, convert_to_numpy)
        percentile_before_pooling = percentile_before_pooling and percentile_output

        if isinstance(answers, str):
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
                if detailed_infos:
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
                    detailed_infos=detailed_infos,
                    to_scalar=to_scalar,
                    encoding_batch_size=encoding_batch_size
                )
                total_scores.extend(batch_scores)
                if detailed_infos:
                    detailed.extend(batch_detailed)

        if detailed_infos:
            output = detailed
        elif bucket_output:
            output = self.percentile_output.bucket(total_scores)
        elif percentile_output:
            if split:
                output = finalize_vector(total_scores if percentile_before_pooling else self.percentile_output.score_to_percentile(total_scores))
            else:
                output = finalize_vector(self.percentile_output.score_to_percentile(total_scores))
        else:
            output = finalize_vector(total_scores)
        return output if len(output) > 1 else output[0]
