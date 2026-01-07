from dotenv import load_dotenv
from transformers import AutoTokenizer

from evaluation.config import PROJECT_DIR
from evaluation.qa_datasets.simpleqa_eval import SimpleQAEval
from granuscore.loader import JSONLineReader
from evaluation.qa_datasets.llm_client import LLMClient

# We only want responses that are answered by both LLMs in < 511 tokens for normal and < 2048 tokens to ensure the model
# wanted to end and was not interrupted

MAX_TOKENS = {
    "normal": 511,
    "think": 2047,
}

MODEL = 'Qwen/Qwen3-8B'
MODEL_SHORT = 'qwen3-8b-think'
DATASET = 'truthful_qa'
MODE = 'think' # 'normal'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
BASE_DIR = f"{PROJECT_DIR}/data/qa_datasets/{DATASET}/{MODEL_SHORT}"


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False)) if text else -1

load_dotenv()
llm = LLMClient(api_key='OPENAI_API_KEY_GRANU_SCORE')
samples = JSONLineReader().read(f"{BASE_DIR}/outputs-{MODEL_SHORT}-{DATASET}.jsonl")

filtered_samples = []
for sample in samples:
    n_predicted_answer = count_tokens(sample.get("predicted_answer", ""), TOKENIZER)
    n_reasoning = count_tokens(sample.get("reasoning", ""), TOKENIZER)
    n = n_predicted_answer + n_reasoning

    a = sample.get("predicted_answer", "")

    if n_predicted_answer != -1 and n <= MAX_TOKENS[MODE]:
        filtered_samples.append(sample)

if DATASET in {"simple_qa", "facts_parametric"}:
    simple_eval = SimpleQAEval(llm, samples, mode="single")
else:
    simple_eval = SimpleQAEval(llm, samples, mode="multi")

output_file = f"{BASE_DIR}/raw-judge-{MODEL_SHORT}-{DATASET}.jsonl"
simple_eval.create_grade_batch(output_file, upload_batch=True)
