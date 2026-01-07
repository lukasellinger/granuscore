import logging
import os

from granuscore.loader import JSONLineReader
from ollama import chat as ollama_chat
from openai import OpenAI
from openai.types import Batch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

class LLMClient:
    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "OPENAI_API_KEY_OWN",
            "supports_batch": True
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            #"api_key": Credentials.openrouter_api_key,
            "supports_batch": False
        },
        "fireworks": {
            "base_url": "https://api.fireworks.ai/inference/v1",
            "api_key": "FW_API_KEY",
            "supports_batch": False
        },
        "runpod": {
            "base_url": "https://api.runpod.ai/v2/antqvephydl3xa/openai/v1",
            #"api_key": Credentials.runpod_api_key,
            "supports_batch": False
        },
        "vllm": {
            "base_url": "http://localhost:8000/v1",
            # "api_key": Credentials.runpod_api_key,
            "supports_batch": False
        },
        "ollama": {
            "base_url": None,
            "api_key": None,
            "supports_batch": False
        },
        "hf_local": {
            "base_url": None,
            "api_key": "HF_API_KEY",
            "supports_batch": False
        }
    }

    @classmethod
    def register_provider(cls, name: str, base_url: str = None, api_key: str = None, supports_batch: bool = False):
        cls.PROVIDERS[name.lower()] = {
            "base_url": base_url,
            "api_key": api_key,
            "supports_batch": supports_batch
        }

    def __init__(self, provider: str = "openai", model: str = "gpt-4.1-nano-2025-04-14", temperature: float = 0.7, api_key: str = None):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

        config = self.PROVIDERS[self.provider]
        self.api_key = (
            os.getenv(api_key) or
            os.getenv(config["api_key"])
        )

        if self.provider == "hf_local":
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self._api_client = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype="auto",
                device_map="auto"
            )
        elif self.provider != "ollama":
            config = self.PROVIDERS[self.provider]
            self._api_client = OpenAI(api_key=self.api_key, base_url=config["base_url"])

    def get_model_name(self):
        return self.model

    def chat(self, messages: list[dict], temperature: float = None, use_temperature: bool = True) -> str:
        """Non-streaming chat: returns the full response as string."""
        if self.provider == "ollama":
            resp = ollama_chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature},
                stream=False
            )
            return resp["message"]["content"].strip()
        elif self.provider == "hf_local":
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._api_client.device)

            outputs = self._api_client.generate(**inputs, max_new_tokens=512)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
            return response.split('<end_of_turn>')[0].strip()
        else:
            params = {
                "model": self.model,
                "input": messages,
            }
            if use_temperature:
                params["temperature"] = temperature if temperature is not None else self.temperature

            resp = self._api_client.responses.create(**params)
            return resp.output_text.strip()

    def chat_stream(self, messages: list[dict], temperature: float = None):
        """Streaming chat: yields partial chunks as they arrive."""
        if self.provider == "ollama":
            for chunk in ollama_chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": temperature if temperature is not None else self.temperature},
                    stream=True
            ):
                yield chunk["message"]["content"]
        else:
            stream = self._api_client.responses.create(
                model=self.model,
                input=messages,
                stream=True
            )
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    def chat_structured(self, messages: list[dict], text_format, temperature: float = None, use_temperature: bool = True):
        if self.provider == "ollama":
            raise NotImplementedError("Structured parsing not implemented for Ollama.")
        else:
            params = {
                "model": self.model,
                "input": messages,
                "text_format": text_format
            }
            if use_temperature:
                params["temperature"] = temperature if temperature is not None else self.temperature

            response = self._api_client.responses.parse(**params)
            return response.output_parsed

    def upload_batch_file(self, file_name: str):
        if not self.PROVIDERS[self.provider]["supports_batch"]:
            raise NotImplementedError(f"Batch upload not supported for provider '{self.provider}'.")
        with open(file_name, "rb") as file:
            return self._api_client.files.create(file=file, purpose="batch")

    def create_batch_job(self, file_name: str, endpoint="/v1/responses") -> Batch:
        if not self.PROVIDERS[self.provider]["supports_batch"]:
            raise NotImplementedError(f"Batch jobs not supported for provider '{self.provider}'.")
        batch_file = self.upload_batch_file(file_name)
        return self._api_client.batches.create(
            input_file_id=batch_file.id,
            endpoint=endpoint,
            completion_window="24h"
        )

    @staticmethod
    def get_openai_batch_text(result):
        return result['response']['body']['output'][0]['content'][0]['text']


    def get_batch_result(self, output_file: str, result_file_id: str) -> str | None:
        """
        Retrieves the result of a completed batch job and saves it to a specified file.

        :param output_file: Path where the result file should be saved.
        :param result_file_id: file_id of the batch job.
        :return: Path to the output file where the result was saved, or None if no result is
        available.
        """
        if not result_file_id:
            return None

        result = self._api_client.files.content(result_file_id).content

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as file:
            file.write(result)

        return output_file

    def prompt_openai_batch(self, samples, file_out, prompt, model: str | None = None, temperature: float = 0, upload_batch: bool = True):
        tasks = []
        for sample in samples:
            task = {
                "custom_id": f"task-{sample['id']}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model or self.model,
                    "temperature": temperature,
                    "input": [
                        {
                            "role": "user",
                            "content": prompt.format(**sample)
                        }
                    ]
                }
            }
            tasks.append(task)

        JSONLineReader().write(file_out, tasks)
        if upload_batch:
            return self.create_batch_job(file_out)

if __name__ == "__main__":
    client_hf = LLMClient()
    response = client_hf.chat([{"role": "user", "content": "Hello, who are you?"}])
    print(response)
