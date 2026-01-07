from openai import OpenAI

from evaluation.config import PROJECT_DIR
from evaluation.qa_datasets.eval_datasets import DATASETS
from evaluation.qa_datasets.utils import map_with_progress
from granuscore.loader import JSONLineReader

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY"
)

for dataset, dataset_entry in DATASETS.items():
    OUTPUT_FILE = f'{PROJECT_DIR}/evaluation/data/qa_datasets/outputs/qwen-8b-think/outputs-qwen-8b-think-{dataset}.jsonl'
    def fn(row: dict):
        chat_history = [
            {'role': 'user', 'content': f'Answer following query in at most five complete sentences: {row['question']}'}]

        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=chat_history,
            temperature=0.0,
            max_tokens=2048,
        )
        response = response.choices[0].message
        output = {'id': row['id'],
                'question': row['question'],
                'target_answer': row['answer'],
                'predicted_answer': response.content,
                'reasoning': response.reasoning
        }
        JSONLineReader().write(OUTPUT_FILE, [output])
        return output

    data = dataset_entry['data']
    results = map_with_progress(fn, data)
