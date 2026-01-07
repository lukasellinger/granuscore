from pathlib import Path

import numpy as np

from evaluation.config import PROJECT_DIR
from granuscore import GranuScore
from granuscore.loader import JSONLineReader

BASE_DIR = f"{PROJECT_DIR}/data/qa_datasets"
EVAL_MODELS = ["deepseek-v3-2", "olmo3-7b", "qwen3-06b", "qwen3-8b", "qwen3-8b-think", "qwen3-32b"]


def store_granularities(model_name, dataset, granularities, ans_mean_gr, rea_mean_gr):
    OVERVIEW_FILE = f"{PROJECT_DIR}/data/qa_datasets/overview_output_granularity.txt"
    GRANULARITIES_FILE = (
        f"{PROJECT_DIR}/data/qa_datasets/{dataset}/{model_name}/{model_name}-{dataset}-output-granularities.jsonl"
    )

    Path(GRANULARITIES_FILE).parent.mkdir(parents=True, exist_ok=True)
    JSONLineReader().write(GRANULARITIES_FILE, granularities)

    ans_str = "N/A" if ans_mean_gr is None else f"{ans_mean_gr:.4f}"
    rea_str = "N/A" if rea_mean_gr is None else f"{rea_mean_gr:.4f}"

    text = (
        f"{model_name} - {dataset}: "
        f"Answer {ans_str} - Reasoning {rea_str}\n"
    )
    with open(OVERVIEW_FILE, "a", encoding="utf-8") as f:
        f.write(text)


def run_single_dataset(model: GranuScore, samples):
    granularities = []

    ids = [b["id"] for b in samples]

    answer_texts = []
    answer_indices = []

    for i, b in enumerate(samples):
        a = b.get("predicted_answer")
        if isinstance(a, str) and a.strip():
            answer_texts.append(a)
            answer_indices.append(i)

    answer_grs = [None] * len(samples)
    if answer_texts:
        encoded = model(answer_texts, show_progress_bar=True, convert_to_numpy=False)
        for idx, gr in zip(answer_indices, encoded):
            answer_grs[idx] = gr

    reasoning_texts = []
    reasoning_indices = []

    for i, b in enumerate(samples):
        r = b.get("reasoning")
        if isinstance(r, str) and r.strip():
            reasoning_texts.append(r)
            reasoning_indices.append(i)

    reasoning_texts = [b.get("reasoning", "") for b in samples]
    has_reasoning = any([bool(r.strip()) for r in reasoning_texts])

    reasoning_grs = [None] * len(samples)
    if has_reasoning:
        reasoning_grs = model(reasoning_texts, show_progress_bar=True, convert_to_numpy=False)

    # ---------- Collect ----------
    for d_id, ans_gr, rea_gr in zip(ids, answer_grs, reasoning_grs):
        granularities.append({
            "id": d_id,
            "answer_granularity": ans_gr,      # None if missing
            "reasoning_granularity": rea_gr,   # None if missing
        })

    # ---------- Dataset-level means ----------
    answer_vals = [
        g["answer_granularity"]
        for g in granularities
        if g["answer_granularity"] is not None
    ]
    answer_mean_gr = float(np.mean(answer_vals)) if answer_vals else None

    reasoning_vals = [
        g["reasoning_granularity"]
        for g in granularities
        if g["reasoning_granularity"] is not None
    ]
    reasoning_mean_gr = (
        float(np.mean(reasoning_vals)) if reasoning_vals else None
    )

    return granularities, answer_mean_gr, reasoning_mean_gr


def main():
    scorer = GranuScore()

    datasets = ['facts_parametric', 'simple_qa', 'squad', 'truthful_qa']
    for dataset in datasets:
        for eval_model in EVAL_MODELS:
            samples = JSONLineReader().read(f"{BASE_DIR}/{dataset}/{eval_model}/outputs-{eval_model}-{dataset}.jsonl")

            print(f"\n=== Evaluating pipeline: {eval_model} - {dataset} ===")

            granularities, ans_mean_gr, reasoning_mean_gr = run_single_dataset(scorer, samples)
            store_granularities(eval_model, dataset, granularities, ans_mean_gr, reasoning_mean_gr)


if __name__ == '__main__':
    main()
