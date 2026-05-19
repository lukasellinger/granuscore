import os
import platform
import re
import time
from abc import ABC, abstractmethod
from typing import Literal, Any, get_args

import lightgbm as lgb
import faiss
import numpy as np
import torch
from hierarchy_transformers import HierarchyTransformer
from numpy.random.mtrand import Sequence
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from nltk.corpus import wordnet as wn

from granuscore.artifcats import ArtifactManager
from granuscore.bucket_output import PercentileOutput
from granuscore.cache import GranuscoreCache
from granuscore.default_assets import DEFAULT_LGB_MODEL, DEFAULT_ANCHORS, DEFAULT_FAISS_INDEX, DEFAULT_NOUN_SCORES, \
    DEFAULT_ORIGINAL_VECTORS
from granuscore.utils import get_best_device, iter_batches, map_with_progress

_FAISS_THREADING_CONFIGURED = False

def configure_faiss_threading():
    global _FAISS_THREADING_CONFIGURED
    if _FAISS_THREADING_CONFIGURED:
        return
    _FAISS_THREADING_CONFIGURED = True

    system = platform.system()
    if system == "Darwin":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        faiss.omp_set_num_threads(1)
        print("[GranuScorer] macOS detected → using FAISS single-thread mode.")
    else:
        max_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(max_threads)
        print(f"[GranuScorer] Linux detected → enabling FAISS parallelism ({max_threads} threads).")


SearchMethod = Literal["nearest_neighbor", "random", "random_anchors", "radial_anchors"]
PredictorType = Literal["hit", "sentence_transformer", "llm"]

class GranularityPredictor(ABC):
    """
    Pure interface defining the granularity predictor contract.
    No implementation details.
    """

    NO_FACT_SCORE = 5.0
    percentile_output: PercentileOutput = None

    @abstractmethod
    def score(
        self,
        answers: Sequence[str],
        encoding_batch_size: int | None = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def score_wo_cache(
        self,
        answers: Sequence[str],
        encoding_batch_size: int | None = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def extract_lgb_features(
        self,
        answers: Sequence[str],
        encoding_batch_size: int | None = None,
    ) -> np.ndarray:
        pass

    def __call__(self, answers, encoding_batch_size=None):
        return self.score(answers, encoding_batch_size)


class BaseIndexGranularityPredictor(GranularityPredictor):
    """
    Abstract base class for all granularity predictors.

    Defines the public interface and shared cache behavior.
    Concrete subclasses must implement embedding and feature extraction.
    """
    DEFAULT_MODEL_NAME: str | None = None

    def __init__(
        self,
        model_name: str | None = None,
        faiss_index_path: str | None = None,
        lgb_model_path: str | None = None,
        anchor_path: str | None = None,
        search_method: SearchMethod = "random_anchors",
        random_anchors_k: int = 999,
        use_cache: bool = True,
    ):
        """
        Initialize an index-based granularity predictor.

        The default configuration reproduces the setup used in the paper and
        works out of the box without requiring any additional resources.

        Default configuration
        ---------------------
        - model_name:
          ``"Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun"``
        - search_method:
          ``"random_anchors"``
        - random_anchors_k:
          ``999``
        - use_cache:
          ``True``

        Parameters
        ----------
        model_name:
            Embedding model used to encode referential units.
            If None, the predictor-specific default model is used.

        faiss_index_path:
            Optional path to a FAISS index containing searchable reference vectors.
            Required for search methods relying on nearest-neighbor retrieval
            (e.g. ``"nearest_neighbor"`` or ``"random"``).

        lgb_model_path:
            Optional path to the LightGBM granularity model used for final score
            prediction.

        anchor_path:
            Optional path to precomputed anchor vectors used by anchor-based
            search methods (e.g. ``"random_anchors"`` or ``"radial_anchors"``).

        search_method:
            Strategy used to retrieve or construct comparison vectors for feature
            extraction.

        random_anchors_k:
            Number of anchor vectors used for ``"random_anchors"``.

        use_cache:
            Whether to cache computed granularity scores.

        Notes
        -----
        The provided resources must be compatible with each other.

        In particular, ``lgb_model_path``, ``search_method``,
        ``faiss_index_path``, ``anchor_path``, and related resources are expected
        to originate from the same training/configuration setup.

        For example, using the default LightGBM model trained for
        ``search_method="random_anchors"`` together with
        ``search_method="nearest_neighbor"`` may produce invalid results or
        undefined behavior.

        Similarly, anchor vectors, FAISS indices, reference percentile scores,
        and LightGBM models must correspond to the same embedding space and
        feature configuration.

        Compatibility between custom resources is not validated automatically
        and must be ensured by the user.
        """
        if search_method not in get_args(SearchMethod):
            raise ValueError(
                f"Invalid search_method '{search_method}'. "
                f"Expected one of: {get_args(SearchMethod):}"
            )

        model_name = model_name or self.DEFAULT_MODEL_NAME
        if model_name is None:
            raise ValueError("model_name must be provided.")

        self.search_method = search_method
        self.random_anchors_k = random_anchors_k

        manager = ArtifactManager()

        # Defaults to settings in paper (random_anchors with k=999)
        lgb_model_path = lgb_model_path or str(manager.ensure(DEFAULT_LGB_MODEL))
        if search_method in {"random_anchors", "radial_anchors"}:
            anchor_path = anchor_path or str(manager.ensure(DEFAULT_ANCHORS))
        elif search_method in {"nearest_neighbor", "random"}:
            faiss_index_path = faiss_index_path or str(manager.ensure(DEFAULT_FAISS_INDEX))

        self.index = None
        self.N = 0
        if faiss_index_path is not None:
            self.index = self._load_index(faiss_index_path)
            self.N = self.index.ntotal

        self.anchors = None
        if anchor_path is not None:
            self.anchors = np.load(anchor_path)

        self.rng = np.random.default_rng(42) if search_method == "random" else None

        self.model = self._load_model(model_name)
        self.granu_predictor = self._load_granu_predictor(lgb_model_path)

        if search_method == "random_anchors" and random_anchors_k == 999:
            reference_scores_path = str(manager.ensure(DEFAULT_NOUN_SCORES))
            self.percentile_output = PercentileOutput(reference_scores_path)

        self.cache = None
        if use_cache:
            self.cache = GranuscoreCache(
                scorer=self,
                hierarchy_model=model_name,
                faiss_index=str(faiss_index_path),
                lgb_model=str(lgb_model_path),
            )

    @abstractmethod
    def _load_model(self, model_name: str):
        pass

    @abstractmethod
    def _embed_answers(self, answers: Sequence[str]):
        pass

    @abstractmethod
    def _build_lgb_features(
            self,
            ans_vec: np.ndarray,
            sims: np.ndarray,
            neighbors: np.ndarray,
    ) -> np.ndarray:
        pass

    def _load_index(self, faiss_index: str):
        configure_faiss_threading()
        return faiss.read_index(faiss_index)

    def _load_granu_predictor(self, granu_predictor_name: str | None):
        if granu_predictor_name:
            return lgb.Booster(model_file=granu_predictor_name)
        return None

    def __call__(
        self,
        answers: Sequence[str],
        encoding_batch_size: int | None = None,
    ):
        return self.score(answers, encoding_batch_size)

    def score(
        self,
        answers: Sequence[str],
        encoding_batch_size: int | None = None,
    ):
        if self.cache:
            return self.cache.batch(answers, encoding_batch_size)
        else:
            return self.score_wo_cache(answers, encoding_batch_size)

    def score_wo_cache(
        self,
        answers: Sequence[str],
        encoding_batch_size: int | None = None,
    ):
        if not self.granu_predictor:
            raise ValueError("LGB model is None.")

        features = self.extract_lgb_features(
            answers=answers,
            encoding_batch_size=encoding_batch_size,
        )

        return self.granu_predictor.predict(features)

    def extract_lgb_features(
            self,
            answers: Sequence[str],
            encoding_batch_size: int | None = None,
            show_progress_bar: bool = False,
            k: int = 1000
    ) -> np.ndarray:
        """
        Compute the feature vectors fed into the LightGBM granularity model.
        """
        encoding_batch_size = encoding_batch_size or len(answers)
        k = min(k, self.N)

        feats_all = []

        batch_iter = iter_batches(answers, encoding_batch_size)
        if show_progress_bar:
            num_batches = (len(answers) + encoding_batch_size - 1) // encoding_batch_size
            batch_iter = tqdm(
                batch_iter,
                total=num_batches,
                desc="LGB features",
            )

        for ans_batch in batch_iter:
            ans_vec = self._embed_answers(ans_batch)
            sims, neighbors = self._build_sim_neighbors(ans_vec, k)
            feats_all.append(
                self._build_lgb_features(
                    ans_vec=ans_vec,
                    sims=sims,
                    neighbors=neighbors,
                )
            )

        return np.concatenate(feats_all, axis=0)

    @abstractmethod
    def _build_sim_neighbors(self, ans_vec: np.ndarray, k) -> tuple[Any, Any]:
        pass

    def _search(self, query_vec: np.ndarray, k: int, return_neighbors: bool = False):
        """
        Search top-k most similar concepts to query_vec.
        Returns (scores, indices) with shapes (k,), (k,)
        """
        norms = np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12
        query_vec = query_vec / norms

        scores, indices = self.index.search(query_vec, k)
        if return_neighbors:
            flat = indices.flatten()  # shape: (batch * k,)
            vecs = [self.index.reconstruct(int(i)) for i in flat]
            neighbor_vectors = np.array(vecs).reshape(indices.shape[0], indices.shape[1], self.index.d)
            return scores, indices, neighbor_vectors
        return scores, indices

    def _search_random(self, query_vec: np.ndarray, k: int, return_neighbors: bool = False):
        """
        Random neighbor ablation:
        - sample k random index vectors
        - compute real similarity to query_vec
        """
        norms = np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12
        query_vec = query_vec / norms

        batch_size, d = query_vec.shape
        k = min(k, self.N)

        # 1) sample random indices
        indices = self.rng.integers(0, self.N, size=(batch_size, k))

        # 2) reconstruct vectors
        flat = indices.flatten()
        vecs = [self.index.reconstruct(int(i)) for i in flat]
        neighbor_vectors = np.array(vecs).reshape(batch_size, k, d)

        # 3) compute similarity
        sims = np.einsum("bd,bkd->bk", query_vec, neighbor_vectors)

        order = np.argsort(-sims, axis=1)
        sims = np.take_along_axis(sims, order, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)

        if return_neighbors:
            order_exp = order[..., None]
            neighbor_vectors = np.take_along_axis(neighbor_vectors, order_exp, axis=1)
            return sims.astype(np.float32), indices, neighbor_vectors.astype(np.float32)

        return sims.astype(np.float32), indices

    def _search_anchors(self, query_vec, return_neighbors=False):
        # normalize query
        norms = np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12
        query_vec = query_vec / norms

        b = query_vec.shape[0]
        neighbor_vectors = np.broadcast_to(
            self.anchors[None, :, :],  # (1, k, d)
            (b, self.anchors.shape[0], self.anchors.shape[1])
        )

        # cosine similarity
        neighbor_norms = np.linalg.norm(neighbor_vectors, axis=2, keepdims=True) + 1e-12
        neighbor_normed = neighbor_vectors / neighbor_norms

        sims = np.einsum("bd,bkd->bk", query_vec, neighbor_normed)

        # canonical ordering (important!)
        order = np.argsort(-sims, axis=1)

        sims = np.take_along_axis(sims, order, axis=1)

        if return_neighbors:
            order_exp = order[..., None]
            neighbor_vectors = np.take_along_axis(neighbor_vectors, order_exp, axis=1)
            return sims.astype(np.float32), neighbor_vectors.astype(np.float32)

        return sims.astype(np.float32)


class HitGranularityPredictor(BaseIndexGranularityPredictor):
    DEFAULT_MODEL_NAME = "Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun"

    def __init__(
        self,
        model_name: str | None = None,
        search_method: SearchMethod = "random_anchors",
        random_anchors_k: int = 999,
        use_cache: bool = True,
        faiss_index_path: str | None = None,
        lgb_model_path: str | None = None,
        anchor_path: str | None = None,
        original_vectors_path: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            faiss_index_path=faiss_index_path,
            lgb_model_path=lgb_model_path,
            anchor_path=anchor_path,
            search_method=search_method,
            random_anchors_k=random_anchors_k,
            use_cache=use_cache,
        )

        self.original_vectors = None

        if search_method in {"nearest_neighbor", "random"}:
            manager = ArtifactManager()

            if original_vectors_path is None:
                original_vectors_path = str(
                    manager.ensure(DEFAULT_ORIGINAL_VECTORS)
                )

            self.original_vectors = np.load(
                original_vectors_path
            )

    def _load_model(self, model_name: str):
        return HierarchyTransformer.from_pretrained(model_name).to(
            get_best_device()
        )

    def _build_sim_neighbors(self, ans_vec: np.ndarray, k) -> tuple[Any, Any]:
        if self.search_method in ['radial_anchors', 'random_anchors']:
            sims, neighbors = self._search_anchors(ans_vec, return_neighbors=True)
        else:
            if self.search_method == 'random':
                sims, indices, neighbors = self._search_random(ans_vec, k, return_neighbors=True)
            elif self.search_method == 'nearest_neighbor':
                sims, indices, neighbors = self._search(ans_vec, k, return_neighbors=True)
            else:
                raise ValueError("Unknown search method '{}'".format(self.search_method))
            neighbors = self.original_vectors[indices]
        return sims, neighbors

    def _embed_answers(self, answers: Sequence[str]):
        if hasattr(answers, "tolist"):  # pandas Series
            answers = answers.tolist()
        else:
            answers = list(answers)

        return self.model.encode(answers).astype("float32")

    def _build_lgb_features(
            self,
            ans_vec: np.ndarray,
            sims: np.ndarray,
            neighbors: np.ndarray,
    ) -> np.ndarray:
        """
        Build the feature vector that is fed into the LightGBM model.

        Returns
        -------
        np.ndarray
            Shape: [batch_size, 1 + k + k]
            (dist0, similarities, distances)
        """
        device = get_best_device()

        ans_vec_t = torch.from_numpy(ans_vec).to(device)
        neighbors_t = torch.from_numpy(neighbors).to(device)

        # distance to origin
        ans_vec_dist0 = self.model.manifold.dist0(ans_vec_t)  # [b]

        # distances to neighbors
        dists = self.model.manifold.dist(
            neighbors_t,
            ans_vec_t.unsqueeze(1)  # [b, 1, d]
        )  # [b, k]

        features = np.concatenate(
            [
                ans_vec_dist0.cpu().unsqueeze(-1),  # [b, 1]
                sims,  # [b, k]
                dists.cpu(),  # [b, k]
            ],
            axis=1,
        )

        return features


class HitWithoutDis0Predictor(HitGranularityPredictor):
    def _build_lgb_features(
            self,
            ans_vec: np.ndarray,
            sims: np.ndarray,
            neighbors: np.ndarray,
    ) -> np.ndarray:
        """
        Build the feature vector that is fed into the LightGBM model.

        Returns
        -------
        np.ndarray
            Shape: [batch_size, 1 + k + k]
            (dist0, similarities, distances)
        """
        device = get_best_device()

        ans_vec_t = torch.from_numpy(ans_vec).to(device)
        neighbors_t = torch.from_numpy(neighbors).to(device)

        # distances to neighbors
        dists = self.model.manifold.dist(
            neighbors_t,
            ans_vec_t.unsqueeze(1)  # [b, 1, d]
        )  # [b, k]

        features = np.concatenate(
            [
                sims,  # [b, k]
                dists.cpu(),  # [b, k]
            ],
            axis=1,
        )

        return features


class SentenceTransformerGranularityPredictor(BaseIndexGranularityPredictor):
    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def _embed_answers(self, answers: Sequence[str]):
        if hasattr(answers, "tolist"):  # pandas Series
            answers = answers.tolist()
        else:
            answers = list(answers)

        return self.model.encode(answers).astype("float32")

    def _build_sim_neighbors(self, ans_vec: np.ndarray, k) -> tuple[Any, Any]:
        if self.search_method in ['radial_anchors', 'random_anchors']:
            sims, neighbors = self._search_anchors(ans_vec, return_neighbors=True)
        else:
            if self.search_method == 'random':
                sims, indices, neighbors = self._search_random(ans_vec, k, return_neighbors=True)
            elif self.search_method == 'nearest_neighbor':
                sims, indices, neighbors = self._search(ans_vec, k, return_neighbors=True)
            else:
                raise ValueError("Unknown search method '{}'".format(self.search_method))
        return sims, neighbors

    def _build_lgb_features(self, ans_vec: np.ndarray, sims: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        """
        Build the feature vector that is fed into the LightGBM model.

        Returns
        -------
        np.ndarray
            Shape: [batch_size, k]
            (similarities)
        """
        return sims

    def _load_model(self, model_name: str):
        return SentenceTransformer(model_name).to(get_best_device())


class LengthGranularityPredictor(GranularityPredictor):
    """Simply returns the char length of each answer."""
    def __init__(self, use_cache: bool = True):
        self.cache = None
        if use_cache:
            self.cache = GranuscoreCache(
                scorer=self,
                hierarchy_model=f'length',
                faiss_index='None',
                lgb_model='None',
            )

    def score(self, answers: Sequence[str], encoding_batch_size: int | None = None) -> np.ndarray:
        if self.cache:
            return self.cache.batch(answers, encoding_batch_size)
        else:
            return self.score_wo_cache(answers, encoding_batch_size)

    def score_wo_cache(self, answers: Sequence[str], encoding_batch_size: int | None = None) -> np.ndarray:
        return self.extract_lgb_features(answers, encoding_batch_size)

    def extract_lgb_features(self, answers: Sequence[str], encoding_batch_size: int | None = None) -> np.ndarray:
        if hasattr(answers, "tolist"):
            answers = answers.tolist()
        else:
            answers = list(answers)

        lengths = np.array([len(a) for a in answers], dtype=np.float32)

        return lengths.reshape(-1, 1)


class WordNetGranularityPredictor(GranularityPredictor):
    """Returns the avg depth of each answer inside wordnet hierarchy."""
    def __init__(self, use_cache: bool = True):
        self.cache = None
        if use_cache:
            self.cache = GranuscoreCache(
                scorer=self,
                hierarchy_model=f'wordnet',
                faiss_index='None',
                lgb_model='None',
            )

    def score(self, answers: Sequence[str], encoding_batch_size: int | None = None) -> np.ndarray:
        if self.cache:
            return self.cache.batch(answers, encoding_batch_size)
        else:
            return self.score_wo_cache(answers, encoding_batch_size)

    def score_wo_cache(self, answers: Sequence[str], encoding_batch_size: int | None = None) -> np.ndarray:
        return self.extract_lgb_features(answers, encoding_batch_size)

    @staticmethod
    def avg_wordnet_depth(word):
        synsets = wn.synsets(word)

        if not synsets:
            return None

        sense_depths = []

        for s in synsets:
            paths = s.hypernym_paths()
            path_depths = [len(p) - 1 for p in paths]  # root depth = 0
            sense_depths.append(np.mean(path_depths))

        return np.mean(sense_depths) if sense_depths else np.nan

    def extract_lgb_features(self, answers: Sequence[str], encoding_batch_size: int | None = None) -> np.ndarray:
        if hasattr(answers, "tolist"):
            answers = answers.tolist()
        else:
            answers = list(answers)

        lengths = np.array([self.avg_wordnet_depth(a) for a in answers], dtype=np.float32)

        return lengths.reshape(-1, 1)


class LLMGranularityPredictor(GranularityPredictor):
    """Score with an LLM using a 5-point Likert scale."""
    PROMPT = """You are an expert annotator for granularity.

Your task is to assign a granularity score to an answer using a 4-point Likert scale.
Granularity refers to how fine- / coarse-grained an answer is.

Always assign exactly one score: 1, 2, 3, or 4.

---

Examples:
Answer: "The Animals"
Granularity: 1

Example 2
Answer: "English rock band"
Granularity: 2

Example 3
Answer: "English band"
Granularity: 3

Example 4
Answer: "English musicians"
Granularity: 4

---

Example 6
Answer: "Seattle"
Granularity: 1

Example 7
Answer: "King County"
Granularity: 2

Example 8
Answer: "Washington"
Granularity: 3

Example 9
Answer: "United States"
Granularity: 4

---

Example 11
Answer: "Banksy"
Granularity: 1

Example 12
Answer: "a graffiti artist'"
Granularity: 2

Example 13
Answer: "a painter"
Granularity: 3

Example 14
Answer: "a political activist"
Granularity: 4

---

Now assign a granularity score. Output only the score.

Answer: "{answer}"

Granularity:
        """

    def __init__(self, model_name: str = 'gpt-4.1-mini-2025-04-14', use_cache: bool = True):
        self.model_name = model_name
        self.provider = OpenAI()
        self.cache = None
        if use_cache:
            self.cache = GranuscoreCache(
                scorer=self,
                hierarchy_model=f'LLM-{model_name}',
                faiss_index='None',
                lgb_model='None',
            )

    def score(self, answers: Sequence[str], encoding_batch_size: int | None = None) -> np.ndarray:
        if self.cache:
            return self.cache.batch(answers, encoding_batch_size)
        else:
            return self.score_wo_cache(answers, encoding_batch_size)

    def score_wo_cache(self, answers: Sequence[str], encoding_batch_size: int | None = None) -> np.ndarray:
        return self.extract_lgb_features(answers, encoding_batch_size)

    def _score_single(self, answer: str) -> int:
        for attempt in range(3):
            try:
                params = {
                    "model": self.model_name,
                    "input": [{'role': 'user', 'content': self.PROMPT.format(answer=answer)}],
                    "temperature": 0,
                }

                resp = self.provider.responses.create(**params)
                text = resp.output_text.strip()

                match = re.search(r"\b([1-4])\b", text)
                if not match:
                    raise ValueError(f"No Likert score found: {text}")

                return int(match.group(1))

            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.5)

    def extract_lgb_features(
            self,
            answers: Sequence[str],
            encoding_batch_size: int | None = None,
    ) -> np.ndarray:

        if hasattr(answers, "tolist"):
            answers = answers.tolist()
        else:
            answers = list(answers)

        scores = map_with_progress(
            f=self._score_single,
            xs=answers,
        )

        return np.array(scores, dtype=np.float32)
