import pickle
from typing import List, Dict, Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm

from granuscore.granularity_predictor import SentenceTransformerGranularityPredictor, BaseIndexGranularityPredictor
from training_scripts.config import PROJECT_DIR
from training_scripts.utils import normalize_to_likert

# =========================
# Configuration
# =========================
MAX_GRANOLA = 4
BATCH_SIZE = 64
K = 999
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# =========================
# Utilities
# =========================
def batch_iter(data: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def build_lgb_entries(entries: List[Dict], max_granola: int) -> List[Dict]:
    lgb_data = []

    for e in entries:
        answers = []
        granolas = []
        for i in range(1, max_granola + 1):
            answer = e.get(f"granola_answer_{i}")
            if answer:
                answers.append(answer)
                granolas.append(i)
        if not answers:
            continue

        Y_norm = normalize_to_likert(granolas, L=max_granola)

        # append entries
        for answer, y in zip(answers, Y_norm):
            lgb_data.append(
                {
                    "id": e["id"],
                    "answer": answer,
                    "Y": float(y),
                }
            )

    return lgb_data


def enrich_with_hierarchy_features(
    data: List[Dict],
    scorer: BaseIndexGranularityPredictor,
    batch_size: int,
    k: int,
):
    for batch in tqdm(batch_iter(data, batch_size),
                      total=len(data) // batch_size + 1):

        # ========= ANSWERS =========
        answers = [b["answer"] for b in batch]
        answer_feats = compute_hierarchy_features(
            answers, scorer, k
        )

        for i, entry in enumerate(batch):
            entry["answer_index_scores"] = answer_feats["index_scores"][i]


def compute_hierarchy_features(
    inputs: List[str],
    scorer: BaseIndexGranularityPredictor,
    k: int,
):
    index_scores = scorer.extract_lgb_features(inputs, k=k)

    return {
        "index_scores": index_scores,
    }


def dump_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# =========================
# Main
# =========================
def main():
    scorer = SentenceTransformerGranularityPredictor(
        model_name=MODEL_NAME,
        entity_index=INDEX_PATH,
        granu_predictor_name=None,
        search_method=search_method)
    dataset = load_dataset(f"{PROJECT_DIR}/data/granola-eq")

    train_lgb = build_lgb_entries(dataset["train"].to_list(), MAX_GRANOLA)
    val_lgb = build_lgb_entries(dataset["validation"].to_list(), MAX_GRANOLA)
    test_lgb = build_lgb_entries(dataset["test"].to_list(), MAX_GRANOLA)

    enrich_with_hierarchy_features(
        val_lgb, scorer, BATCH_SIZE, K
    )
    dump_pickle(val_lgb, f"{PROJECT_DIR}/data/precalc_data/{INDEX_NAME}-{search_method}-eval_data_lgb.pkl")

    enrich_with_hierarchy_features(
        test_lgb, scorer, BATCH_SIZE, K
    )
    dump_pickle(test_lgb, f"{PROJECT_DIR}/data/precalc_data/{INDEX_NAME}-{search_method}-test_data_lgb.pkl")

    enrich_with_hierarchy_features(
        train_lgb, scorer, BATCH_SIZE, K
    )
    dump_pickle(train_lgb, f"{PROJECT_DIR}/data/precalc_data/{INDEX_NAME}-{search_method}-train_data_lgb.pkl")


if __name__ == "__main__":
    search_method = 'random_anchors'
    INDEX_NAME = f"50k-all-min"
    INDEX_PATH = f"{PROJECT_DIR}/src/granuscore/assets/faiss/{INDEX_NAME}-index.faiss"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    main()
