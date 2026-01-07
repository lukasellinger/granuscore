import os

from evaluation.config import PROJECT_DIR
from evaluation.qa_datasets.simpleqa_eval import SimpleQAEval
from granuscore.loader import JSONLineReader

EVAL_MODELS = ["deepseek-v3-2", "olmo3-7b", "qwen3-06b", "qwen3-8b", "qwen3-8b-think", "qwen3-32b"]
DATASETS = ['facts_parametric', 'simple_qa', 'squad', 'truthful_qa']

def main():
    simple_eval = SimpleQAEval(None, None, None)
    base_dir = f"{PROJECT_DIR}/data/qa_datasets"

    for dataset in DATASETS:
        for model in EVAL_MODELS:
            input_file = f"{base_dir}/{dataset}/{model}/batch-output-judge-{model}-{dataset}.jsonl"

            stats = simple_eval.grade_batch(input_file)
            stored_stats = {
                "info": f"{model}-{dataset}",
                "score": round(float(stats["score"]), 4),
                "is_correct": round(float(stats["metrics"]["is_correct"]), 4),
                "is_incorrect": round(float(stats["metrics"]["is_incorrect"]), 4),
                "is_not_attempted": round(float(stats["metrics"]["is_not_attempted"]), 4),
                "is_given_attempted": round(float(stats["metrics"]["is_given_attempted"]), 4),
                "accuracy_given_attempted": round(float(stats["metrics"]["accuracy_given_attempted"]), 4),
            }
            stats_output_file = f"{base_dir}/graded-stats.jsonl"
            JSONLineReader().write(stats_output_file, [stored_stats])

if __name__ == "__main__":
    main()