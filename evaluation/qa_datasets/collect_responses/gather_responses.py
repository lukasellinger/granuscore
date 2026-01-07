import os
import random
import time

from dotenv import load_dotenv
from openai import OpenAI

from evaluation.config import PROJECT_DIR
from evaluation.qa_datasets.eval_datasets import DATASETS
from evaluation.qa_datasets.utils import map_with_progress
from granuscore.loader import JSONLineReader

load_dotenv()

model_name = "qwen/qwen3-8b"
model_name_short = "qwen3-8b-coarse-grained"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def sleep_backoff(attempt: int, base: float = 0.3, cap: float = 30.0, jitter: float = 0.25):
    """
    Exponential backoff with jitter.
    attempt: 1,2,3,...
    """
    delay = min(cap, base * (2 ** (attempt - 1)))
    # jitter in [1-jitter, 1+jitter]
    delay *= (1.0 + random.uniform(-jitter, jitter))
    time.sleep(max(0.0, delay))


for dataset, dataset_entry in DATASETS.items():
    OUTPUT_FILE = (
        f"{PROJECT_DIR}/qa_datasets/data/{dataset}/"
        f"{model_name_short}/outputs-{model_name_short}-{dataset}.jsonl"
    )

    ERROR_FILE = (
        f"{PROJECT_DIR}/qa_datasets/data/{dataset}/"
        f"{model_name_short}/errors-{model_name_short}-{dataset}.jsonl"
    )

    EXISTING_ENTRIES = JSONLineReader().read(OUTPUT_FILE) or []
    existing_ids = {row["id"] for row in EXISTING_ENTRIES if row["predicted_answer"] != ''}

    def log_error(row_id, msg):
        with open(ERROR_FILE, "a") as f:
            f.write(f"{row_id}\t{msg}\n")

    def call_model(row: dict) -> str:
        chat_history = [
            {
                "role": "user",
                "content": "Answer following query in at most five complete sentences: "
                           f"{row['question']}",
            }]

        resp = client.chat.completions.create(
            model=model_name,
            messages=chat_history,
            temperature=0.0,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "reasoning": {'enabled': False}},
        )

        text = resp.choices[0].message.content
        if text is None or text.strip() == "":
            raise ValueError("Empty response")

        return text

    def fn(row: dict, max_retries: int = 5):
        """
        Returns output dict on success, None on failure (after retries).
        Failed rows are appended to RETRY_QUEUE for later reruns.
        """
        last_err = None

        for attempt in range(1, max_retries + 1):
            try:
                text = call_model(row)

                output = {
                    "id": row["id"],
                    "question": row["question"],
                    "target_answer": row["answer"],
                    "predicted_answer": text,
                }
                JSONLineReader().write(OUTPUT_FILE, [output])
                return output
            except Exception as e:
                print('Retrying', attempt, row['id'])
                if attempt < max_retries:
                    sleep_backoff(attempt, base=1.0, cap=30.0, jitter=0.25)

        # exhausted retries -> re-queue
        JSONLineReader().write(ERROR_FILE, [{
            "id": row.get("id"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "error": str(last_err) if last_err else "Unknown error",
        }])
        return None

    data = dataset_entry["data"]
    filtered_data = [row for row in data if row["id"] not in existing_ids]
    results = map_with_progress(fn, filtered_data)
