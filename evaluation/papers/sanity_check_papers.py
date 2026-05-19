from typing import List, Tuple

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from tqdm import tqdm

from evaluation.config import PROJECT_DIR
from granuscore import GranuScore
from granuscore.claim_splitter import SpacyNounPhraseSplitter
from granuscore.loader import JSONLineReader


def score_intro_vs_related(
    model: dict,
    papers: List[dict],
) -> Tuple[List[float], List[float]]:
    introductions = [paper['introduction_par'] for paper in papers]
    related_works = [paper['related_work_par'] for paper in papers]

    intro_scores = model['scorer'].predict(
        introductions,
        pooling=model['pooling'],
        pooling_scope=model['pooling_scope'],
        scope_pooling_method=model['scope_pooling_method'],
        tail_q=model.get('tail_q'),
        scope_tail_q=model.get('scope_tail_q'),
        encoding_batch_size=64,
        batch_size=64,
        show_progress_bar=True,
        return_percentiles=True,
        percentile_before_pooling=True
    )

    rel_scores = model['scorer'].predict(
        related_works,
        pooling=model['pooling'],
        pooling_scope=model['pooling_scope'],
        scope_pooling_method=model['scope_pooling_method'],
        tail_q=model.get('tail_q'),
        scope_tail_q=model.get('scope_tail_q'),
        encoding_batch_size=64,
        batch_size=64,
        show_progress_bar=True,
        return_percentiles=True,
        percentile_before_pooling=True
    )

    return intro_scores, rel_scores

def report_stats(intro: List[float], rel: List[float]) -> dict:
    intro = np.asarray(intro)
    rel = np.asarray(rel)

    stats = {
        "correct_ordering": np.mean(intro > rel),
        "intro_mean": np.mean(intro),
        "intro_std": np.std(intro),
        "rel_mean": np.mean(rel),
        "rel_std": np.std(rel),
        "ttest": ttest_rel(intro, rel),
        "wilcoxon": wilcoxon(intro, rel, alternative="greater"),
    }

    print(
        f"Intro Mean: {stats['intro_mean']:.4f} | Std: {stats['intro_std']:.4f}\n"
        f"Rel  Mean: {stats['rel_mean']:.4f} | Std: {stats['rel_std']:.4f}\n"
        f"T-test:    {stats['ttest']}\n"
        f"Wilcoxon:  {stats['wilcoxon']}\n"
    )

    return stats


def main():
    scorer = GranuScore(claim_splitter=SpacyNounPhraseSplitter(skip_brackets=True,
                                                               skip_figures=True,
                                                               skip_latex=True,
                                                               skip_arxiv_meta=True,
                                                               skip_pdf_noise=True))
    test_pipelines = dict()
    for pooling_scope in ['sentence', 'document']:
        scope_poolings = ['mean'] # if 'document' we do not need scope pooling
        if pooling_scope == 'sentence':
            scope_poolings = ['weighted_mean', 'sum', 'mean', 'lower_quantile_mean', 'min', 'max']
        for scope_pooling in scope_poolings:
            for pooling in ['sum', 'mean', 'lower_quantile_mean', 'min', 'max']:
                scope_tail_qs = [None]
                tail_qs = [None]
                if pooling == 'lower_quantile_mean':
                    tail_qs = [0.1, 0.3, 0.5]
                if scope_pooling == 'lower_quantile_mean':
                    scope_tail_qs = [0.9, 0.8, 0.7]
                for scope_tail_q in scope_tail_qs:
                    for tail_q in tail_qs:
                        name = f'scope_{pooling_scope}_spooling_{scope_pooling}_pooling_{pooling}'
                        config = {
                            'scorer': scorer,
                            'pooling': pooling,
                            'pooling_scope': pooling_scope,
                            'scope_pooling_method': scope_pooling,
                        }
                        if scope_tail_q is not None:
                            config['scope_tail_q'] = scope_tail_q
                            name += f'_stailq_{scope_tail_q}'
                        if tail_q is not None:
                            config['tail_q'] = tail_q
                            name += f'_tailq_{tail_q}'

                        test_pipelines[name] = config

    data_path = f"{PROJECT_DIR}/data/s2orc/filtered_papers.jsonl"
    papers = JSONLineReader().read(data_path)

    OUTPUT_FILE = f"{PROJECT_DIR}/data/s2orc/evaluations.jsonl"
    for name, model in tqdm(test_pipelines.items()):
        print(f"\n=== Evaluating: {name} ===")

        intro, rel = score_intro_vs_related(model, papers)
        stats = report_stats(intro, rel)

        JSONLineReader().write(OUTPUT_FILE, [{
            "name": name,
            "correct_ordering": float(stats["correct_ordering"]),
            "intro_mean": float(stats["intro_mean"]),
            "intro_std": float(stats["intro_std"]),
            "rel_mean": float(stats["rel_mean"]),
            "rel_std": float(stats["rel_std"]),
            "ttest_stat": float(stats["ttest"].statistic),
            "ttest_p": float(stats["ttest"].pvalue),
            "wilcoxon_stat": float(stats["wilcoxon"].statistic),
            "wilcoxon_p": float(stats["wilcoxon"].pvalue),
        }])


if __name__ == '__main__':
    main()