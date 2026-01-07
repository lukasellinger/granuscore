from pathlib import Path

import numpy as np

from evaluation.qa_datasets.eval_datasets import DATASETS

from evaluation.config import PROJECT_DIR
from granuscore import GranuScore
from granuscore.loader import JSONLineReader


def store_granularities(dataset, granularities, ans_mean_gr, que_mean_gr):
    OVERVIEW_FILE = f"{PROJECT_DIR}/data/qa_datasets/overview_granularity.txt"
    GRANULARITIES_FILE = (
        f"{PROJECT_DIR}/data/qa_datasets/{dataset}/granuscore-{dataset}.jsonl"
    )

    Path(GRANULARITIES_FILE).parent.mkdir(parents=True, exist_ok=True)
    JSONLineReader().write(GRANULARITIES_FILE, granularities)

    text = f"{dataset}: Answer {ans_mean_gr:.4f} - Question {que_mean_gr:.4f}\n"
    with open(OVERVIEW_FILE, "a", encoding="utf-8") as f:
        f.write(text)


def run_single_dataset(model: GranuScore, dataset):
    granularities = []
    ids = [b['id'] for b in dataset['data']]
    if dataset['mode'] == 'single':
        answer_batch_granularities = granularity_single_answer_mode(model, dataset['data'])
    else:
        answer_batch_granularities = granularity_multi_answer_mode(model, dataset['data'])
    question_batch_granularities = granularity_question(model, dataset['data'])
    granularities.extend([{'id': d_id, 'answer_granularity': ans_gr, 'question_granularity': que_gr} for d_id, ans_gr, que_gr in zip(ids, answer_batch_granularities, question_batch_granularities)])

    answer_mean_gr = float(
        np.mean([g["answer_granularity"] for g in granularities])
    )
    question_mean_gr = float(
        np.mean([g["question_granularity"] for g in granularities])
    )
    return granularities, answer_mean_gr, question_mean_gr


def granularity_question(model: GranuScore, batch):
    return model([b['question'] for b in batch], show_progress_bar=True, convert_to_numpy=False)


def granularity_single_answer_mode(model: GranuScore, batch):
    return model([b['answer'] for b in batch], show_progress_bar=True, convert_to_numpy=False)


def granularity_multi_answer_mode(model: GranuScore, batch):
    lens = [len(b['answer']) for b in batch]
    all_answers = [a for b in batch for a in b['answer']]
    all_granularities = model.predict(all_answers, show_progress_bar=True, convert_to_numpy=False)
    mean_granularities = []
    idx = 0
    for n in lens:
        mean_granularities.append(
            float(np.mean(all_granularities[idx : idx + n]))
        )
        idx += n
    return mean_granularities


def main():
    scorer = GranuScore()

    for dataset, entries in DATASETS.items():
        granularities, ans_mean_gr, que_mean_gr = run_single_dataset(scorer, entries)
        store_granularities(dataset, granularities, ans_mean_gr, que_mean_gr)

if __name__ == '__main__':
    main()
