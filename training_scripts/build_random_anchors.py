import numpy as np

from evaluation.config import PROJECT_DIR
from granuscore.granularity_predictor import SentenceTransformerGranularityPredictor


def hit_random_anchors(K: int = 1000, seed: int = 42):
    original_vectors = np.load(f"{PROJECT_DIR}/src/granuscore/assets/faiss/50k-hit-original-vectors.npy")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(original_vectors, size=K, replace=False)
    np.save(f"50k-hit-random_anchors_{K}.npy", sampled.astype(np.float32))


def all_min_random_anchors(K: int = 1000, seed: int = 42):
    INDEX_PATH = f"{PROJECT_DIR}/src/granuscore/assets/faiss/50k-all-min-index.faiss"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    scorer = SentenceTransformerGranularityPredictor(
        model_name=MODEL_NAME,
        faiss_index_path=INDEX_PATH,
        lgb_model_path=None,
        search_method='random')
    ans_vec = scorer.model.encode(['test']).astype("float32")
    sims, indices, neighbors = scorer._search_random(ans_vec, K, return_neighbors=True)

    np.save("50k-all-min-random_anchors.npy", neighbors[0].astype(np.float32))

if __name__ == '__main__':
    seed = 42
    K = 999  # should be dividable by 3
    hit_random_anchors(K, seed)
    all_min_random_anchors(K, seed)

    ablated_ks = [
        33,
        66,
        99,
        333,
        666,
        1332,
        1665,
    ]
    for k in ablated_ks:
        hit_random_anchors(k, seed)

