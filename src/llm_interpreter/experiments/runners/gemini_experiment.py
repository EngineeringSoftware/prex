"""
Gemini experiment runner implementation.
"""

import os
import seutil as su
from typing import List, Dict, Any, Union
from pathlib import Path
from google import genai
from google.genai import types
from datetime import datetime
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_interpreter.experiments.base import BaseRunner
from llm_interpreter.macros import Macros


logger = su.log.get_logger(__name__, su.log.INFO)


class GeminiRunner(BaseRunner):
    def __init__(self, model_config_file: Union[str, Path], max_workers: int = 2, **kwargs):
        super().__init__(**kwargs)
        # load model's config
        self.model_config = su.io.load(model_config_file)
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.local = threading.local()
        self._ensure_client()

    # fed

    def _get_price_1_million_input_token(self) -> float:
        return 1.25

    # fed

    def _get_price_1_million_output_token(self) -> float:
        return 10.0

    # fed

    def do_experiment(self):
        self.setup_output_dir()
        results_file_path = (
            self.output_dir / f"results-{str(self.args.random_seed)}.jsonl"
        )
        dataset = self.load_dataset()
        results = self.multi_thread_run(dataset)
        su.io.dump(results_file_path, results)

    def _process_single_data_point(self, data_point: Dict, progress_bar=None) -> List:
        """
        Process a single data point.

        Args:
            data_point: The data point to process
            progress_bar: Optional tqdm progress bar to update

        Returns:
            The processed result
        """
        self._ensure_client()
        chat_prompt = self._prepare_prompt(data_point)
        result = self.query_llm(data_point, chat_prompt=chat_prompt)

        # Update progress bar if provided
        if progress_bar:
            with self.lock:
                progress_bar.update(1)

        return result

    def multi_thread_run(self, dataset: List) -> List:
        logger.info(
            f"Start running experiments with multi-threading: {str(self.args)} ..."
        )
        results = []
        failed_indices = []

        progress_bar = tqdm(total=len(dataset), desc="Inference Progress")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks to the executor
            future_to_idx = {
                executor.submit(
                    self._process_single_data_point, dataset[idx], progress_bar
                ): idx
                for idx in range(len(dataset))
            }

            completed_count = 0

            # Process results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    # Thread-safe append to results
                    with self.lock:
                        # Make sure results list has the right size
                        if len(results) < len(dataset):
                            results.extend([None] * (len(dataset) - len(results)))
                        results[idx] = result
                        completed_count += 1
                    # Save intermediate results periodically
                    if completed_count % 32 == 0:
                        self._save_checkpoint(results)

                except Exception as e:
                    logger.error(
                        f"Failed to process data point at index {idx}: {str(e)}"
                    )
                    failed_indices.append(idx)
                    with self.lock:
                        progress_bar.update(
                            1
                        )  # Still update progress bar for failed items
        progress_bar.close()
        if failed_indices:
            logger.warning(
                f"Failed to process {len(failed_indices)} items: {failed_indices}"
            )
        return results

    def _ensure_client(self):
        """
        Ensure that each thread has its own client instance.
        This should be implemented in a subclass based on the specific LLM API being used.

        )
        """
        if not hasattr(self.local, "client"):
            self.local.client = genai.Client(
                vertexai=True,
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location="us-central1",
                http_options=types.HttpOptions(api_version="v1"),
            )

    def _save_checkpoint(self, results):
        """
        Save intermediate results to a checkpoint file.
        """

        checkpoint_path = (
            Macros.tmp_dir
            / f"ckpt-{str(self.args)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        )

        # Use the lock to ensure thread-safe file operations
        with self.lock:
            su.io.dump(checkpoint_path, results)
            logger.info(
                f"Saved checkpoint with {len(results)} results to {checkpoint_path}"
            )

    def _prepare_content(self, chat: list[dict]) -> list[types.Content]:
        """
        Prepare the content for the Gemini model.
        """
        contents = []
        for message in chat:
            role = message["role"]
            if role == "user":
                role = "user"
            elif role == "assistant":
                role = "model"
            elif role == "system":
                continue
            else:
                raise ValueError(f"Unknown role: {role}")

            content = types.Content(
                role=role, parts=[types.Part.from_text(text=message["content"])]
            )
            contents.append(content)
        return contents

    def _query(
        self,
        chat: list[dict],
        stop: list[str] = [],
    ) -> list[str]:
        try:
            # Prepare the content for the Gemini model
            contents = self._prepare_content(chat)
            response = self.local.client.models.generate_content(
                model=self.args.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self._extract_system_prompt(chat),
                    max_output_tokens=self.model_config.get(
                        "max_completion_tokens", 2048
                    ),
                    stop_sequences=stop,
                ),
            )
            return [response.text]
        except Exception as e:
            logger.warning(
                f"Error while running query with model {self.args.model_name} : {e}"
            )
            return []

    def _extract_system_prompt(self, chat: list[dict]) -> str:
        """
        Extract the system prompt from the chat history.
        """
        for message in chat:
            if message["role"] == "system":
                return message["content"]
        return ""


# ssalc
