import pickle
from typing import List, Dict, Iterable

from datasets import load_dataset
from tqdm import tqdm

from granuscore.granularity_predictor import LengthGranularityPredictor, GranularityPredictor
from training_scripts.config import PROJECT_DIR
from training_scripts.utils import normalize_to_likert

# =========================
# Configuration
# =========================
MAX_GRANOLA = 4
BATCH_SIZE = 64

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
    scorer: GranularityPredictor,
    batch_size: int,
):
    for batch in tqdm(batch_iter(data, batch_size),
                      total=len(data) // batch_size + 1):

        # ========= ANSWERS =========
        answers = [b["answer"] for b in batch]
        answer_feats = compute_hierarchy_features(
            answers, scorer
        )

        for i, entry in enumerate(batch):
            entry["answer_index_scores"] = answer_feats["index_scores"][i]


def compute_hierarchy_features(
    inputs: List[str],
    scorer: GranularityPredictor,
):
    index_scores = scorer.extract_lgb_features(inputs)

    return {
        "index_scores": index_scores,
    }


def dump_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# =========================
# Main
# =========================
def main(scorer):
    dataset = load_dataset(f"{PROJECT_DIR}/data/granola-eq")

    train_lgb = build_lgb_entries(dataset["train"].to_list(), MAX_GRANOLA)
    val_lgb = build_lgb_entries(dataset["validation"].to_list(), MAX_GRANOLA)
    test_lgb = build_lgb_entries(dataset["test"].to_list(), MAX_GRANOLA)

    enrich_with_hierarchy_features(
        val_lgb, scorer, BATCH_SIZE
    )
    dump_pickle(val_lgb, f"{PROJECT_DIR}/data/precalc_data/{INDEX_NAME}-eval_data_lgb.pkl")

    enrich_with_hierarchy_features(
        test_lgb, scorer, BATCH_SIZE
    )
    dump_pickle(test_lgb, f"{PROJECT_DIR}/data/precalc_data/{INDEX_NAME}-test_data_lgb.pkl")

    enrich_with_hierarchy_features(
        train_lgb, scorer, BATCH_SIZE
    )
    dump_pickle(train_lgb, f"{PROJECT_DIR}/data/precalc_data/{INDEX_NAME}-train_data_lgb.pkl")


if __name__ == "__main__":
    INDEX_NAME = "len"
    scorer = LengthGranularityPredictor()
    main(scorer)
