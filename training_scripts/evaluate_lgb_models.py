from pathlib import Path

from datasets import load_dataset
import lightgbm as lgb

from granuscore.loader import JSONLineReader
from training_scripts.config import PROJECT_DIR
from training_scripts.utils import load_sim_data, build_tau_dataset, run_evaluation_dist0, run_evaluation, \
    bootstrap_paired_diff, bootstrap_pairwise_acc_diff, run_inverted_evaluation, run_evaluation_llm


def evaluate_single_model(hf_data, lgb_model: str, model_type: str, k):
    print(f"\n🚀 Evaluating model for lgb_model='{lgb_model}'")

    sim_data = {
        #"train": load_sim_data(lgb_model, "train"),
        #"val": load_sim_data(lgb_model, "eval"),
        "test": load_sim_data(lgb_model, "test"),
    }

    tau_sets = {
        #"train_answers": build_tau_dataset("train", hf_data, sim_data, model_type),
        #"validation_answers": build_tau_dataset("val", hf_data, sim_data, model_type, k=k),
        "test_answers": build_tau_dataset("test", hf_data, sim_data, model_type, k=k),
    }

    if model_type in {'sentence_transformer', 'hit', 'hit-embedding'}:
        predictor = lgb.Booster(model_file=f"{PROJECT_DIR}/src/granuscore/assets/lgb_models/{lgb_model}_model.txt")
    else:
        pass

    all_stats = {}
    all_min_stats = {}
    for k, v in tau_sets.items():
        if model_type in {'hit'}:
            stats_dist0 = run_evaluation_dist0(v)
            stats = run_evaluation(v, predictor)

            #sig_intra = bootstrap_paired_diff(stats['intra_pairwise_acc'], stats_dist0['intra_pairwise_acc'])
            #sig_overall = bootstrap_pairwise_acc_diff(stats_dist0['gold_grans'], stats['predicted_grans'], stats_dist0['predicted_grans'])
            all_stats[k] = {'prefix': f'{lgb_model}', 'stats': stats, 'stats_dist0': stats_dist0,} # "sig_intra": sig_intra, 'sig_overall': sig_overall}

            min_stats = {
                'not_nan_percentage': stats['not_nan_percentage'],
                'global_pairwise_acc': stats['global_pairwise_acc'],
                'global_kendall_tau': stats['global_kendall_tau'],
                'global_pearson': stats['global_pearson'],
                'avg_intra_pairwise_acc': stats['avg_intra_pairwise_acc'],
                'avg_intra_kendall_tau': stats['avg_intra_kendall_tau'],
                'mean_intra_full_correct': stats['mean_intra_full_correct'],
                'avg_granuscores': stats['avg_granuscores'],
                'intra_significance': stats['intra_significance'],
            }

            min_stats_dist0 = {
                'not_nan_percentage': stats_dist0['not_nan_percentage'],
                'global_pairwise_acc': stats_dist0['global_pairwise_acc'],
                'global_kendall_tau': stats_dist0['global_kendall_tau'],
                'global_pearson': stats_dist0['global_pearson'],
                'avg_intra_pairwise_acc': stats_dist0['avg_intra_pairwise_acc'],
                'avg_intra_kendall_tau': stats_dist0['avg_intra_kendall_tau'],
                'mean_intra_full_correct': stats_dist0['mean_intra_full_correct'],
                'avg_granuscores': stats_dist0['avg_granuscores'],
                'intra_significance': stats_dist0['intra_significance'],
            }

            all_min_stats[k] = {'prefix': f'{lgb_model}', 'stats': min_stats, 'stats_dist0': min_stats_dist0}
        else:
            if model_type in {'sentence_transformer', 'hit-embedding'}:
                model_stats = run_evaluation(v, predictor)
            elif model_type in {'len', 'wordnet'}:
                model_stats = run_inverted_evaluation(v)
            elif model_type in {'llm'}:
                model_stats = run_evaluation_llm(v)
            else:
                raise NotImplementedError
            min_model_stats = {
                'not_nan_percentage': model_stats['not_nan_percentage'],
                'global_pairwise_acc': model_stats['global_pairwise_acc'],
                'global_kendall_tau': model_stats['global_kendall_tau'],
                'global_pearson': model_stats['global_pearson'],
                'avg_intra_pairwise_acc': model_stats['avg_intra_pairwise_acc'],
                'avg_intra_kendall_tau': model_stats['avg_intra_kendall_tau'],
                'mean_intra_full_correct': model_stats['mean_intra_full_correct'],
                'avg_granuscores': model_stats['avg_granuscores'],
                'intra_significance': model_stats['intra_significance'],
            }

            all_stats[k] =  {'prefix': f'{lgb_model}', 'stats': model_stats}
            all_min_stats[k] = {'prefix': f'{lgb_model}', 'stats': min_model_stats}
    return all_stats, all_min_stats


def write_model_ranking(results, out_path):
    lines = []
    lines.append("\n🏆 MODEL RANKING (by test global_pairwise_acc)")
    lines.append("-" * 50)

    for rank, r in enumerate(results, 1):
        lines.append(
            f"{rank:>2}. {r['test_answers']['prefix']:<10} | "
            f"τ = {r['test_answers']['stats']['global_pairwise_acc']:.4f} | "
        )

    text = "\n".join(lines)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    # optional: still print to console
    print(text)


def compare_methods(method1, method2):
    hf_data = load_dataset(f"{PROJECT_DIR}/data/granola-eq")

    def get_results_single(method):
        lgb_model, model_type, k = method

        sim_data = {
            # "train": load_sim_data(lgb_model, "train"),
            # "val": load_sim_data(lgb_model, "eval"),
            "test": load_sim_data(lgb_model, "test"),
        }

        test_answers = build_tau_dataset("test", hf_data, sim_data, model_type, k=k)


        if model_type in {'sentence_transformer', 'hit', 'hit-embedding'}:
            predictor = lgb.Booster(model_file=f"{PROJECT_DIR}/src/granuscore/assets/lgb_models/{lgb_model}_model.txt")
        else:
            raise ValueError(f'Currently not defined for {model_type}')

        return run_evaluation(test_answers, predictor)

    stats_method1 = get_results_single(method1)
    stats_method2 = get_results_single(method2)
    sig_intra = bootstrap_paired_diff(stats_method1['intra_pairwise_acc'], stats_method2['intra_pairwise_acc'])
    sig_overall = bootstrap_pairwise_acc_diff(stats_method1['gold_grans'], stats_method1['predicted_grans'], stats_method2['predicted_grans'])
    JSONLineReader().write(f'{PROJECT_DIR}/training_scripts/evaluation_data/significance.jsonl', [sig_intra, sig_overall])
    print('Intra', sig_intra['p_value'])
    print('Overall', sig_overall['p_value'])

# sig_intra = bootstrap_paired_diff(stats['intra_pairwise_acc'], stats_dist0['intra_pairwise_acc'])
# sig_overall = bootstrap_pairwise_acc_diff(stats_dist0['gold_grans'], stats['predicted_grans'], stats_dist0['predicted_grans'])


def main():
    hf_data = load_dataset(f"{PROJECT_DIR}/data/granola-eq")

    MODELS = [
        #("50k-hit-random_anchors-33", "hit", 33),
        #("50k-hit-random_anchors-66", "hit", 66),
        #("50k-hit-random_anchors-99", "hit", 99),
        #("50k-hit-random_anchors-333", "hit", 333),
        #("50k-hit-random_anchors-666", "hit", 666),
        #("50k-hit-random_anchors-999", "hit", 999),
        #("50k-hit-random_anchors-1332", "hit", 1332),
        #("50k-hit-random_anchors-1665", "hit", 1665),
        #("50k-hit-radial_anchors", "hit", 999),
        #("50k-hit-random", "hit", 999),
        #("50k-hit-nearest_neighbor", "hit", 999),
        #("50k-all-min-random_anchors", "sentence_transformer", 999),
        #("50k-all-min-random", "sentence_transformer", 999),
        #("50k-all-min-nearest_neighbor", "sentence_transformer", 999),
        #("len", "len", 999),
        #("llm-gpt-4-1-mini", "llm", 999),
        #("llm-gpt-4-1", "llm", 999),
        #("wordnet", "wordnet", 999),
        #(f"50k-hit-embedding-radial_anchors", "hit-embedding", 999),
        #(f"50k-hit-embedding-random_anchors", "hit-embedding", 999),
        #(f"50k-hit-embedding-random", "hit-embedding", 999),
        #(f"50k-hit-embedding-nearest_neighbor", "hit-embedding", 999),
    ]

    results = []
    OUTPUT_FILE = f'{PROJECT_DIR}/training_scripts/evaluation_data/evaluate_lgb_models-new.jsonl'
    MIN_OUTPUT_FILE = f'{PROJECT_DIR}/training_scripts/evaluation_data/min_evaluate_lgb_models-new.jsonl'
    for lgb_model, model_type, k in MODELS:
        result, min_results = evaluate_single_model(hf_data, lgb_model, model_type, k)
        JSONLineReader().write(OUTPUT_FILE, [{'name': f'{lgb_model}', **result}])
        JSONLineReader().write(MIN_OUTPUT_FILE, [{'name': f'{lgb_model}', **min_results}])
        results.append(result)

    # ---- ranking ----
    results = sorted(results, key=lambda x: x["test_answers"]['stats']["global_pairwise_acc"], reverse=True)

    RANKING_OUTPUT_FILE = f'{PROJECT_DIR}/training_scripts/evaluation_data/evaluate_lgb_ranking-new.txt'
    write_model_ranking(results, RANKING_OUTPUT_FILE)

if __name__ == "__main__":
    #main()

    method1 = ("50k-hit-random_anchors-999", "hit", 999)
    method2 = ("50k-hit-embedding-random_anchors", "hit-embedding", 999)
    compare_methods(method1, method2)
