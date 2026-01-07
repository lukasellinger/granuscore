import os
import statistics

from evaluation.config import PROJECT_DIR
from granuscore import GranuScore
from granuscore.loader import JSONLineReader, JSONReader
from evaluation.sentence_specificity.utils import score_dataset, analyze_length_granularity_gam, plot_gam


def load_data():
    sentences = []
    for filename in os.listdir(f'{PROJECT_DIR}/data/news_articles'):
        if not filename.endswith('spec.json'):
            continue
        file_sentences = JSONReader().read(f'{PROJECT_DIR}/data/news_articles/{filename}')
        file_sentences = [{'file': filename, 'sentence': sentence} for sentence in file_sentences]
        sentences.extend(file_sentences)

    ds_list_all = []
    for sentence in sentences:
        snt_content = sentence['sentence']
        if len(text := snt_content['tkn']) > 0 and (ratings := snt_content['ratings']):
            ds_list_all.append({'file': sentence['file'], 'sent_id': snt_content['sentid'],
                                'rating': statistics.mean(ratings), 'sentence': text})
    return ds_list_all

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

    dataset = load_data()
    outputfile_stats = f'{PROJECT_DIR}/data/news_articles/evaluation/stats-news-articles.jsonl'
    for model_name, model in test_pipelines.items():
        print(f"\n=== Evaluating pipeline: {model_name} ===")

        scores, ratings = score_dataset(
            model['scorer'],
            dataset,
            invert_rating=False,
            pooling=model['pooling'],
            pooling_scope=model['pooling_scope'],
            scope_pooling_method=model['scope_pooling_method'],
            tail_q=model.get('tail_q'),
            scope_tail_q=model.get('scope_tail_q')
        )
        print('Dataset', len(ratings))

        res = analyze_length_granularity_gam(dataset, ratings, scores)
        JSONLineReader().write(outputfile_stats, [{'model': model_name, **res}])

        outputfile = f'{PROJECT_DIR}/data/news_articles/evaluation/gams/{model_name}-gam-{{type}}.pdf'
        plot_gam(res['gam_curves'], outputfile)

if __name__ == "__main__":
    main()