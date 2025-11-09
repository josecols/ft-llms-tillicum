import os
from enum import Enum

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Prompts(Enum):
    @staticmethod
    def format_messages(article_text: str) -> list:
        return [
            {
                "role": "system",
                "content": "You are a specialist medical communicator responsible for translating biomedical articles into a clear, accurate 10-20 sentence summary for non-experts. The summary should be at a Flesch–Kincaid grade level of 10–14 and explain any technical terms.",
            },
            {"role": "user", "content": article_text},
            {"role": "assistant", "content": "Summary:"},
        ]


class LlamaSummarizer:
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        batch_size: int = 1,
        input_field: str = "article",
        max_new_tokens: int = 256,
        decoding: str = "greedy",
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    ):
        self._summaries = []
        self._dtype = torch.bfloat16
        self._model_id = base_model
        self._model = None
        self._tokenizer = None
        self._decoding = decoding
        self._chat_template_config = {
            "tokenize": False,
            "add_generation_prompt": False,
            "continue_final_message": True,
        }

        self.batch_size = batch_size
        self.output_path = f"{output_path}.csv"
        self.input_field = input_field
        self.checkpoint_rate = 4  # save state every n batches.
        self.max_new_tokens = max_new_tokens

        self.dataset = self._load_dataset(dataset_path)
        self._load_model()

        print(f"Running inference with config: {self.__dict__}")

    @classmethod
    def _load_dataset(cls, dataset_path: str) -> Dataset:
        return Dataset.from_pandas(pd.read_parquet(dataset_path))

    def _get_decoding_config(self) -> dict:
        decoding_config = {
            "dola": {
                "custom_generate": "transformers-community/dola",
                "do_sample": False,
                "dola_layers": "low",
                "trust_remote_code": True,
            },
            "greedy": {},
        }

        return decoding_config.get(self._decoding, {})

    def _load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, use_fast=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            dtype=self._dtype,
            device_map="auto",
        )

        self._model.generation_config.max_new_tokens = self.max_new_tokens

    def _read_checkpoint(self):
        if os.path.exists(self.output_path):
            df = pd.read_csv(self.output_path)
            self._summaries = df["summary"].tolist()
            print(f"Resuming from checkpoint (loaded {len(self._summaries)} records).")
        else:
            self._summaries = []
            print("No checkpoint found.")

        return len(self._summaries)

    def _write_checkpoint(self, batch_number: int):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        pd.DataFrame({"summary": self._summaries}).to_csv(self.output_path)
        print(f"Checkpoint saved at batch {batch_number}.")

    def _batch_data(self, start_index: int, end_index: int) -> list[dict]:
        batch = self.dataset[self.input_field][start_index:end_index]
        prompts = [
            self._tokenizer.apply_chat_template(
                Prompts.format_messages(sample),
                **self._chat_template_config,
            )
            for sample in batch
        ]

        return prompts

    def _parse_outputs(self, outputs):
        new_summaries = [s.strip() for s in outputs]
        self._summaries.extend(new_summaries)

    def _preprocess(self, start_index: int, end_index: int):
        data = self._batch_data(start_index, end_index)
        inputs = self._tokenizer(data, return_tensors="pt", padding=True).to(
            self._model.device
        )
        input_length = inputs["input_ids"].shape[1]

        return input_length, inputs

    def _postprocess(self, outputs, input_length: int, batch_number: int):
        self._parse_outputs(
            self._tokenizer.batch_decode(
                outputs[:, input_length:], skip_special_tokens=False
            )
        )

        if batch_number % self.checkpoint_rate == 0:
            self._write_checkpoint(batch_number)

    def generate(self):
        start_index = self._read_checkpoint()
        batch_number = start_index // self.batch_size
        total_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size

        while start_index < len(self.dataset):
            batch_number += 1
            print(f"Processing batch {batch_number}/{total_batches}...")
            end_index = min(start_index + self.batch_size, len(self.dataset))

            input_length, inputs = self._preprocess(start_index, end_index)
            outputs = self._model.generate(**inputs)
            self._postprocess(outputs, input_length, batch_number)

            start_index = end_index

        self._write_checkpoint(batch_number)

        return self._summaries


class LlamaSummarizerTuned(LlamaSummarizer):
    def __init__(
        self, adapter_path: str, input_field: str, batch_size: int, *args, **kwargs
    ):
        self._adapter_path = adapter_path
        super().__init__(
            input_field=input_field,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()

        self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
