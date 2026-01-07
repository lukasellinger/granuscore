import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from granuscore.loader import JSONLineReader


class GeminiApi:
    def __init__(self, api_key: str | None = None):
        load_dotenv()
        api_key = api_key or os.getenv("GENAI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def get_batch_text_output(result):
        return result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get(
            'text')

    def create_batch(self, samples, file_name: str | None, model: str = "gemini-3-pro-preview", store_input: bool = True, run_batch: bool = True):
        if store_input and not file_name:
            raise ValueError("file_name cannot be None if store_input is True")
        if not store_input:
            file_name = f'{time.time()}.jsonl'

        JSONLineReader().write(file_name, samples, mode="w")

        uploaded_file = self.client.files.upload(
            file=file_name,
            config=types.UploadFileConfig(display_name=file_name, mime_type='jsonl')
        )

        print(f"Uploaded file: {uploaded_file.name}")

        if run_batch:
            batch = self.client.batches.create(
                model=model,
                src=uploaded_file.name,
                config={
                    'display_name': uploaded_file.name,
                },
            )
            return batch.name

    def get_batch_overview(self):
        batch_list = self.client.batches.list()
        return [batch.name for batch in batch_list]

    def get_batch_update(self, batch_name: str):
        batch_job = self.client.batches.get(name=batch_name)
        return batch_job.state.name

    def get_continous_batch_update(self, batch_name: str):
        batch_job = self.client.batches.get(name=batch_name)
        completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

        print(f"Polling status for job: {batch_name}")
        while batch_job.state.name not in completed_states:
            print(f"Current state: {batch_job.state.name}")
            time.sleep(30)  # Wait for 30 seconds before polling again
            batch_job = self.client.batches.get(name=batch_name)

        print(f"Job finished with state: {batch_job.state.name}")
        if batch_job.state.name == 'JOB_STATE_FAILED':
            print(f"Error: {batch_job.error}")

    def save_batch_results(self, batch_name, output_file):
        batch_job = self.client.batches.get(name=batch_name)

        if batch_job.state.name == "JOB_STATE_SUCCEEDED":
            if batch_job.dest and batch_job.dest.file_name:
                result_file_name = batch_job.dest.file_name
                print(f"Results file: {result_file_name}")

                file_content = self.client.files.download(file=result_file_name)
                with open(output_file, "wb") as f:
                    f.write(file_content)

                print(f"Saved results to {output_file}")
            else:
                print("No result file found.")
        else:
            print(f"Job state: {batch_job.state.name}")
            if batch_job.error:
                print(batch_job.error)