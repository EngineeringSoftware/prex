"""
Base experiment runner class.
"""

import os
import re
import tiktoken
import seutil as su
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from llm_interpreter.experiments.prompt_maker import (
    make_srp_prompt,
    make_pep_prompt,
    make_op_prompt,
    make_etp_prompt,
    make_translate_prompt,
    make_ig_prompt,
    make_iga_prompt,
    make_igaf_prompt,
    make_dataset_generation_prompt,
    make_ast_prompt,
    make_dgc_prompt,
    make_nl2rule_prompt,
    make_rule2nl_prompt
)
from llm_interpreter.experiments.args import ExperimentArgs
from llm_interpreter.macros import Macros
from llm_interpreter.experiments.prompts import system_prompt


logger = su.log.get_logger(__name__, su.log.INFO)


class BaseRunner(ABC):
    """Base class for all experiment runners."""

    def __init__(self, args: ExperimentArgs, use_hf: bool = False, hf_repo_id: str = None, hf_split: str = None, hf_config: str = None, hf_token: str = None, hf_train_split: bool = False):
        self.args = args
        if os.environ.get("PCP_TOTAL_SHARDS"):
            self.args.total_shards = int(os.environ["PCP_TOTAL_SHARDS"])
        if os.environ.get("PCP_SHARD"):
            self.args.shard = int(os.environ["PCP_SHARD"])
        self.use_hf = use_hf
        self.hf_repo_id = hf_repo_id
        self.hf_split = hf_split
        self.hf_config = hf_config
        self.hf_token = hf_token
        self.hf_train_split = hf_train_split
        self._system: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        logger.info(f"Querying model: {args.model_name}")

    @abstractmethod
    def _get_price_1_million_input_token(self) -> float:
        raise NotImplementedError(
            f"Input pricing per 1 million tokens is not implemented for {self.args.model_name}!"
        )

    # fed

    @abstractmethod
    def _get_price_1_million_output_token(self) -> float:
        raise NotImplementedError(
            f"Output pricing per 1 million tokens is not implemented for {self.args.model_name}!"
        )

    # fed

    def setup_output_dir(self):
        """Set up the output directory for the experiment."""
        self.output_dir = (
            Macros.results_dir
            / self.args.task
            / self.args.setup_name
            / (
                self.args.expr_name
                if self.args.dataset_name == "human_written"
                else f"{self.args.expr_name}-{self.args.dataset_name}"
            )
            / (
                self.args.model_name.replace("/", "-")
                + f"-{self.args.prompt_strategy.value}"
            )
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _merged_results_path(self) -> Path:
        return self.output_dir / f"results-{str(self.args.random_seed)}.jsonl"

    def _shard_results_path(self, shard: int | None = None) -> Path:
        shard_idx = self.args.shard if shard is None else shard
        total_shards = max(1, int(getattr(self.args, "total_shards", 1) or 1))
        return (
            self.output_dir
            / f"results-{str(self.args.random_seed)}.shard{shard_idx}-of-{total_shards}.jsonl"
        )

    def _merged_results_line_count(self, path: Path) -> int:
        if not path.is_file() or path.stat().st_size == 0:
            return 0
        return sum(1 for _ in path.open())

    def _try_merge_sharded_results(self) -> None:
        total_shards = max(1, int(getattr(self.args, "total_shards", 1) or 1))
        if total_shards <= 1:
            return

        merged_path = self._merged_results_path()
        shard_paths = [self._shard_results_path(shard) for shard in range(total_shards)]
        if not all(path.exists() for path in shard_paths):
            logger.info(
                "Shard merge pending: %s/%s shard files present for %s",
                sum(path.exists() for path in shard_paths),
                total_shards,
                merged_path,
            )
            return

        merged_records = []
        for path in shard_paths:
            merged_records.extend(su.io.load(path))
        expected_count = len(merged_records)
        merged_count = self._merged_results_line_count(merged_path)
        if merged_count == expected_count:
            return

        if merged_path.exists():
            logger.info(
                "Removing stale merged results (%s lines, expected %s): %s",
                merged_count,
                expected_count,
                merged_path,
            )
            merged_path.unlink()

        su.io.dump(merged_path, merged_records)
        logger.info(
            "Merged %s shard files into %s (%s records)",
            total_shards,
            merged_path,
            len(merged_records),
        )

    def do_experiment(self, print_output: bool = False):
        self.setup_output_dir()
        total_shards = max(1, int(getattr(self.args, "total_shards", 1) or 1))
        results_file_path = (
            self._shard_results_path()
            if total_shards > 1
            else self._merged_results_path()
        )
        dataset = self.load_dataset(shard, total_shards)
        results = self.run(dataset, print_output)
        su.io.dump(results_file_path, results)
        self._try_merge_sharded_results()

    def _num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string for a specific model."""
        try:
            encoding = tiktoken.encoding_for_model(self.args.model_name)
        except KeyError:
            # Model not found, using cl100k_base encoding as fallback (used by GPT-4, GPT-3.5-turbo)
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))

    def num_dataset_input_and_output_tokens(self, mutation_pattern: str = None) -> (int, int, int):
        dataset = self.load_dataset()
        input_tokens = []
        output_tokens = []
        num_input_tokens: int = 0
        max_output_tokens: int = int(
            self.model_config.get("max_completion_tokens", 16000)
        ) * len(dataset)
        min_output_tokens: int = 0
        for data in dataset:
            if mutation_pattern is not None and data["mutation-pattern"] != mutation_pattern:
                continue
            elif mutation_pattern is None and data["mutation-pattern"] is not None:
                continue
            chat_prompt = self._prepare_prompt(data)
            chat = self._assemble_chat(chat_prompt)
            input_token = self._num_tokens_from_string(str(chat))
            input_tokens.append(input_token)
            num_input_tokens += input_token
            match self.args.task:
                case "etp":
                    output_token = self._num_tokens_from_string(
                        str(data["ground-truth"])
                    )
                    output_tokens.append(output_token)
                    min_output_tokens += output_token
                case "op":
                    output_token = self._num_tokens_from_string(
                        str(data["ground-truth"])
                    )
                    output_tokens.append(output_token)
                    min_output_tokens += output_token
                case "srp":
                    output_token = self._num_tokens_from_string(
                        str(data["ground-truth"])
                    )
                    output_tokens.append(output_token)
                    min_output_tokens += output_token
                case "pcp":
                    min_output_tokens += self._num_tokens_from_string(
                        f"{str(data['ans'])} {str(data['semantic-error-rule'])}"
                    )
                case "nl2rule":
                    output_token = self._num_tokens_from_string(
                        str(data["options"][data["answer_index"]])
                    )
                    output_tokens.append(output_token)
                    min_output_tokens += output_token
                case "rule2nl":
                    output_token = self._num_tokens_from_string(
                        str(data["options"][data["answer_index"]])
                    )
                    output_tokens.append(output_token)
                    min_output_tokens += output_token
                # fi
        # rof
        return (num_input_tokens, min_output_tokens, max_output_tokens, input_tokens, output_tokens)

    # fed

    def get_cost_estimation_for_task(self) -> float:
        [num_input_tokens, min_output_tokens, max_output_tokens, _, _] = (
            self.num_dataset_input_and_output_tokens()
        )
        print(f"num_input_tokens: {num_input_tokens}, min_output_tokens: {min_output_tokens}, max_output_tokens: {max_output_tokens}")
        return (
            (
                (num_input_tokens * self._get_price_1_million_input_token())
                + (min_output_tokens * self._get_price_1_million_output_token())
            )
            / (10**6),
            (
                (num_input_tokens * self._get_price_1_million_input_token())
                + (max_output_tokens * self._get_price_1_million_output_token())
            )
            / (10**6),
        )

    # fed

    def load_dataset(self, shard: int = 0, total_shards: int = 1) -> List:
        logger.info(
            f"Loading dataset {self.args.dataset_name} for task {self.args.task}, under setup {self.args.setup_name} on {self.args.expr_name} ...."
        )
        if self.use_hf:
            if self.hf_train_split:
                dataset = load_dataset(self.hf_repo_id, name=self.hf_config, split=self.hf_split, token=self.hf_token)['train']
            else:
                dataset = load_dataset(self.hf_repo_id, name=self.hf_config, split=self.hf_split, token=self.hf_token)
        else:
            data_file_path = (
                Macros.data_dir
                / "dataset"
                / (
                    f"dataset-{self.args.task}-{self.args.setup_name}-{self.args.expr_name}.jsonl"
                    if self.args.dataset_name == "human_written"
                    else f"dataset-{self.args.task}-{self.args.setup_name}-{self.args.expr_name}-{self.args.dataset_name}.jsonl"
                )
            )
            dataset = su.io.load(data_file_path)
            total_shards = max(1, int(getattr(self.args, "total_shards", 1) or 1))
            shard = int(getattr(self.args, "shard", 0) or 0)
            if total_shards > 1:
                num_shard_samples = len(dataset) // total_shards
                start_idx = shard * num_shard_samples
                # last shard absorbs any remainder so we don't silently drop samples
                end_idx = (
                    len(dataset)
                    if shard == total_shards - 1
                    else (shard + 1) * num_shard_samples
                )
                dataset = dataset[start_idx:end_idx]
                logger.info(
                    f"Loaded {len(dataset)} samples for shard {shard} of {total_shards} "
                    f"for sample range [{start_idx} : {end_idx}]"
                )
        return dataset

    def run(self, dataset: List, print_output: bool = False) -> List:
        """
        Run the experiments on the provided dataset
        """
        logger.info(f"Start running experiments: {str(self.args)} ...")
        # temp = 0.0
        results = []
        count = 0
        for p in tqdm(dataset, total=len(dataset)):
            # if count == 5:
            #     break
            # count += 1
            # Save intermediate results periodically
            if len(results) % 128 == 0:
                su.io.dump(
                    Macros.tmp_dir / f"ckpt-{str(self.args)}-{datetime.now()}.jsonl",
                    results,
                )
            #
            chat_prompt = self._prepare_prompt(p)
            #result = {}
            result = self.query_llm(p, chat_prompt, print_output)
            results.append(result)
        return results

    def query_llm(
        self, dt: Dict, chat_prompt: List[dict], print_output: bool = False
    ) -> Dict:
        """
        Query LLM with the **single** data.
        """
        res = {}
        res.update(
            {
                key: value
                for key, value in dt.items()
                if key not in {"syntax", "semantics"}
            }
        )
        # query llm
        chat = self._assemble_chat(chat_prompt)
        model_responses = self._query(chat)
        # res["model-input"] = chat
        if len(model_responses) == 1:
            res["model-prediction"] = model_responses[0]
        elif len(model_responses) > 1:
            res["model-prediction"] = model_responses
        else:
            res["model-prediction"] = ""
            logger.warning(f"No response from model {self.args.model_name} for {dt}")
        if print_output:
            print(model_responses)
        # fi
        return res

    def _assemble_chat(self, chat_prompt: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create the chat format for LLMs."""
        if "deepseek" in self.args.model_name or "ollama" in self.args.model_name:
            # remove system prompt according to the deepseek model card and for ollama
            chat = chat_prompt
        else:
            # Check if the prompt already has a system message
            if chat_prompt and chat_prompt[0].get("role") == "system":
                # If it already has a system message, don't add another one
                chat = chat_prompt
            else:
                # Otherwise, add the default system prompt
                chat = self._system + chat_prompt

        return chat

    def _prepare_prompt(self, dt: Dict) -> List[dict]:
        """
        Return the prompt string given the one data.
        """
        chat = []
        if self.args.task == "pcp":
            chat = make_pep_prompt(self.args, dt)
        elif self.args.task == "op":
            chat = make_op_prompt(self.args, dt)
        elif self.args.task == "srp":
            chat = make_srp_prompt(self.args, dt)
        elif self.args.task == "etp":
            chat = make_etp_prompt(self.args, dt)
        elif self.args.task == "translate":
            chat = make_translate_prompt(self.args, dt)
        elif self.args.task == "ig":
            chat = make_ig_prompt(self.args, dt)
        elif self.args.task == "iga":
            chat = make_iga_prompt(self.args, dt)
        elif self.args.task == "nl2rule":
            chat = make_nl2rule_prompt(self.args, dt)
        elif self.args.task == "rule2nl":
            chat = make_rule2nl_prompt(self.args, dt)
        else:
            raise NotImplementedError(
                f"Prompt for {str(self.args)} is not implemented yet."
            )

        return chat

    # -----------------
    # helper functions
    # -----------------

    def _extract_code(self, res: str):
        code_match = re.search(
            r"```(\w*)\n([\s\S]*?)```",
            res,
            flags=re.MULTILINE,
        )
        if code_match is not None:
            return code_match.group(2)
        else:
            return ""

    def extract_content_between_tags(self, text: str, tag: str) -> str:
        """Regular expression to match content between <tag> and </tag>"""
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(
            pattern, text, re.DOTALL
        )  # re.DOTALL allows matching across newlines
        return matches[0]

    def add_linenum_to_program(self, program: str) -> str:
        """Add line numbers to the program."""
        lines = program.strip().split("\n")
        numbered_lines = [f"{i + 1}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)
