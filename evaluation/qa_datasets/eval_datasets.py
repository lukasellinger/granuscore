import pandas
from datasets import load_dataset

from evaluation.config import PROJECT_DIR
from evaluation.qa_datasets.utils import unique_normalized


def load_simple_qa():
    df = pandas.read_csv(f"{PROJECT_DIR}/data/qa_datasets/simple_qa/simple_qa_test_set.csv")
    return [
        {'id': i, 'question': q, 'answer': a}
        for i, (q, a) in enumerate(zip(df['problem'], df['answer']))
    ]

def load_facts_parametric_qa():
    df = pandas.read_csv(f"{PROJECT_DIR}/data/qa_datasets/facts_parametric/FACTS-Parametric-public.csv")
    return [
        {'id': i, 'question': q, 'answer': a}
        for i, (q, a) in enumerate(zip(df['query'], df['answer']))
    ]
def load_squad():
    ds = load_dataset('rajpurkar/squad', split='validation')
    return [{'id': d['id'], 'question': d['question'], 'answer': unique_normalized(d['answers']['text'])} for d in ds.to_list()]

def load_truthful_qa():
    ds = load_dataset('domenicrosati/TruthfulQA', split='train')
    return [{'id': i, "question": d['Question'], "answer": unique_normalized([a.strip() for a in d["Correct Answers"].split(";") if a.strip()])} for i, d in enumerate(ds.to_list())]

DATASETS = {
    'simple_qa': {'data': load_simple_qa(), 'mode': 'single'},
    'facts_parametric': {'data': load_facts_parametric_qa(), 'mode': 'single'},
    'squad': {'data': load_squad(), 'mode': 'multi'},
    'truthful_qa': {'data': load_truthful_qa(), 'mode': 'multi'},
}