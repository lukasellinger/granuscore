import os
from typing import Dict, List

from granuscore import GranuScore
from granuscore.loader import JSONLineReader
from evaluation.config import PROJECT_DIR
from evaluation.sentence_specificity.utils import score_dataset, plot_multiple_gams, plot_gam_overlay, analyze_length_granularity_gam


def load_sentences_and_ratings(prefix: str):
    sents_path = os.path.join(PROJECT_DIR, f"data/domain-agnostic/{prefix}s.txt")
    ratings_path = os.path.join(PROJECT_DIR, f"data/domain-agnostic/{prefix}v.txt")

    with open(sents_path) as f:
        sentences = f.readlines()
    with open(ratings_path) as f:
        ratings = f.readlines()

    return [
        {"sentence": s.strip(), "rating": float(r)}
        for s, r in zip(sentences, ratings)
    ]


def main():
    scorer = GranuScore()

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

    datasets: Dict[str, List[dict]] = {
        name: load_sentences_and_ratings(name)
        for name in ["twitter", "yelp", "movie"]
    }

    for model_name, model in test_pipelines.items():
        print(f"\n=== Evaluating pipeline: {model_name} ===")

        outputfile_stats = f'{PROJECT_DIR}/data/domain-agnostic/evaluation/{{dataset}}-stats-domain-agnostic.jsonl'
        gam_curves_gran = dict()
        gam_curves_len = dict()
        for dataset_name, dataset in datasets.items():
            print(f"\nDataset: {dataset_name}")
            scores, ratings = score_dataset(
                model['scorer'],
                dataset,
                invert_rating=True,
                pooling=model['pooling'],
                pooling_scope=model['pooling_scope'],
                scope_pooling_method=model['scope_pooling_method'],
                tail_q=model.get('tail_q'),
                scope_tail_q=model.get('scope_tail_q'),
            )
            res = analyze_length_granularity_gam(dataset, ratings, scores)
            JSONLineReader().write(outputfile_stats.format(dataset=dataset_name), [{'model': model_name, **res}])
            gam_curves_gran[dataset_name] = res['gam_curves']['granularity']
            gam_curves_len[dataset_name] = res['gam_curves']['length']

        outputfile = f'{PROJECT_DIR}/data/domain-agnostic/evaluation/gams/{model_name}-gam-{{type}}-inverted.pdf'
        plot_multiple_gams(gam_curves_gran, outputfile.format(type='granularity'))
        plot_gam_overlay(gam_curves_gran, outputfile.format(type='granularity-overlay'))
        plot_multiple_gams(gam_curves_len, outputfile.format(type='length'))

if __name__ == "__main__":
    main()
