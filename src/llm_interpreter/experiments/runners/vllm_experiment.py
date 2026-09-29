"""
VLLM experiment runner implementation.
"""

import gc
import os
from typing import Any, Dict, List, Union
from pathlib import Path
from transformers import AutoTokenizer
from jsonargparse import CLI
import seutil as su
import torch
from torch.utils.data import Dataset
from datetime import datetime
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

from llm_interpreter.macros import Macros
from llm_interpreter.experiments.base import BaseRunner


logger = su.log.get_logger(__name__, su.log.INFO)

OOM_ERROR_MARKERS = (
    "out of memory",
    "cuda error",
    "hip error",
    "hip out of memory",
    "no available memory",
    "failed to allocate",
    "cannot allocate",
)

# Default number of sequences vLLM may run concurrently. This is decoupled from
# ``batch_size`` so concurrency is no longer capped at the old mini-batch size.
# vLLM only schedules as many sequences as the KV cache can actually hold, and
# the OOM fallback lowers this value if a smaller setting is needed.
DEFAULT_MAX_NUM_SEQS = 64

# Number of prompts submitted per vLLM ``generate`` call. This only controls how
# often intermediate safety checkpoints are written; vLLM continuously batches
# within each call up to ``max_num_seqs``. Most sharded runs have fewer prompts
# than this, i.e. a single ``generate`` call covers the whole shard.
DEFAULT_INFERENCE_CHECKPOINT_EVERY = 1024


def get_gpu_count():
    # torch.cuda.device_count() works on both CUDA (NVIDIA) and HIP (AMD/ROCm)
    # builds of PyTorch, and respects CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES.
    try:
        if not torch.cuda.is_available():
            logger.error("No GPUs detected: torch.cuda.is_available() is False")
            return 0
        gpu_count = torch.cuda.device_count()
        device_names = {torch.cuda.get_device_name(i) for i in range(gpu_count)}
        logger.info(f"{gpu_count} GPUs detected ({', '.join(sorted(device_names))})")
        return gpu_count
    except Exception as e:
        logger.error(f"Error while detecting GPUs: {e}")
        return 0


def get_valid_tensor_parallel_size(num_heads, max_gpus):
    for i in range(min(num_heads, max_gpus), 0, -1):
        if num_heads % i == 0:
            return i
    return 1  # Fallback to 1 if no division is possible


def _patch_mistral_multimodal_hf_config(config):
    """Patch Ministral/Mistral3 HF configs for vLLM 0.8.x on ROCm.

    Ministral 3 checkpoints leave text_config.architectures unset; vLLM 0.8.4 only
    backfills this for model_type \"mistral\", not \"ministral3\".
    """
    text_config = getattr(config, "text_config", None)
    if text_config is None or text_config.architectures is not None:
        return config
    if text_config.model_type in ("mistral", "ministral3"):
        text_config.architectures = ["MistralForCausalLM"]
    return config


# Sample dataset class
class TextDataset(Dataset):
    def __init__(self, prompts: List[Any]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class VLLMRunner(BaseRunner):
    def __init__(self, model_config_file: Union[str, Path], **kwargs):
        super().__init__(**kwargs)

        self.model_config_file = Path(model_config_file)
        self.model_config = su.io.load(self.model_config_file)
        self._runtime_overrides: Dict[str, Any] = {}
        self.model = None
        self.tensor_parallel_size = self._resolve_tensor_parallel_size()

        tokenizer_mode = self.model_config.get("tokenizer_mode", "auto")
        trust_remote_code = self.model_config.get("trust_remote_code", False)
        self.tokenizer_mode = tokenizer_mode
        effective_max_model_len = self._effective_max_model_len()
        if tokenizer_mode == "mistral":
            self.tokenizer = get_tokenizer(
                self.args.model_name,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name,
                model_max_length=effective_max_model_len,
                trust_remote_code=trust_remote_code,
            )

    def _resolve_tensor_parallel_size(self) -> int:
        if "tensor_parallel_size" in self.model_config:
            return int(self.model_config["tensor_parallel_size"])
        return get_valid_tensor_parallel_size(
            self.model_config["attention_head"], get_gpu_count()
        )

    def _effective_config(self) -> Dict[str, Any]:
        cfg = dict(self.model_config)
        cfg.update(self._runtime_overrides)
        if os.environ.get("PCP_BATCH_SIZE"):
            cfg["batch_size"] = int(os.environ["PCP_BATCH_SIZE"])
        if os.environ.get("PCP_MAX_NUM_SEQS"):
            cfg["max_num_seqs"] = int(os.environ["PCP_MAX_NUM_SEQS"])
        if os.environ.get("PCP_GPU_MEMORY_UTILIZATION"):
            cfg["gpu_memory_utilization"] = float(
                os.environ["PCP_GPU_MEMORY_UTILIZATION"]
            )
        return cfg

    def _effective_max_model_len(self, config: Dict[str, Any] | None = None) -> int:
        cfg = config or self._effective_config()
        max_model_len = int(cfg["max_model_len"])
        rope_scaling = cfg.get("rope_scaling")
        if rope_scaling is None:
            return max_model_len
        factor = float(rope_scaling.get("factor", 1.0))
        return int(max_model_len * factor)

    @staticmethod
    def _looks_like_oom(exc: BaseException) -> bool:
        messages = [str(exc).lower()]
        if exc.__cause__ is not None:
            messages.append(str(exc.__cause__).lower())
        return any(
            marker in message for message in messages for marker in OOM_ERROR_MARKERS
        )

    def _dedupe_profiles(self, profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for profile in profiles:
            key = tuple(sorted(profile.items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append(profile)
        return unique

    def _default_oom_fallback_profiles(self) -> List[Dict[str, Any]]:
        cfg = self._effective_config()
        if "oom_fallback_profiles" in self.model_config:
            return list(self.model_config["oom_fallback_profiles"])

        base_seqs = int(cfg.get("max_num_seqs", DEFAULT_MAX_NUM_SEQS))
        base_gpu = float(cfg.get("gpu_memory_utilization", 0.95))
        profiles: List[Dict[str, Any]] = []

        seqs = base_seqs
        while True:
            profiles.append({"max_num_seqs": seqs})
            if seqs == 1:
                break
            seqs = max(1, seqs // 2)

        for gpu_util in (0.90, 0.85):
            if gpu_util < base_gpu - 1e-9:
                profiles.append(
                    {
                        "max_num_seqs": 1,
                        "gpu_memory_utilization": gpu_util,
                        "enforce_eager": True,
                    }
                )
        return self._dedupe_profiles(profiles)

    def _unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _load_model(self, runtime_overrides: Dict[str, Any] | None = None) -> None:
        self._unload_model()
        self._runtime_overrides = dict(runtime_overrides or {})
        cfg = self._effective_config()
        effective_max_model_len = self._effective_max_model_len(cfg)

        tokenizer_mode = cfg.get("tokenizer_mode", "auto")
        config_format = cfg.get("config_format", "auto")
        load_format = cfg.get("load_format", "auto")
        trust_remote_code = cfg.get("trust_remote_code", False)

        logger.info(
            "Loading vLLM model with max_num_seqs=%s "
            "gpu_memory_utilization=%s enforce_eager=%s max_model_len=%s",
            cfg.get("max_num_seqs", DEFAULT_MAX_NUM_SEQS),
            cfg.get("gpu_memory_utilization", 0.95),
            cfg.get("enforce_eager", False),
            effective_max_model_len,
        )
        logger.info(f"Using tensor_parallel_size={self.tensor_parallel_size}")

        llm_kwargs = {
            "tokenizer_mode": tokenizer_mode,
            "config_format": config_format,
            "load_format": load_format,
        }
        if (
            tokenizer_mode == "mistral"
            and config_format == "auto"
            and load_format == "auto"
        ):
            llm_kwargs["hf_overrides"] = _patch_mistral_multimodal_hf_config
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0}

        extra_llm_kwargs: Dict[str, Any] = {}
        extra_llm_kwargs["max_num_seqs"] = int(
            cfg.get("max_num_seqs", DEFAULT_MAX_NUM_SEQS)
        )
        if "enforce_eager" in cfg:
            extra_llm_kwargs["enforce_eager"] = bool(cfg["enforce_eager"])
        if "swap_space" in cfg:
            extra_llm_kwargs["swap_space"] = int(cfg["swap_space"])

        llm_init_kwargs = {
            "model": self.args.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
            "max_model_len": effective_max_model_len,
            "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.95),
            "seed": self.args.random_seed,
            "enable_prefix_caching": True,
            **llm_kwargs,
            **extra_llm_kwargs,
        }
        if "rope_scaling" in cfg:
            rope_scaling = cfg["rope_scaling"]
            rope_type = rope_scaling.get("rope_type", "yarn")
            factor = rope_scaling.get("factor", 2.0)
            original_max_position_embeddings = rope_scaling.get(
                "original_max_position_embeddings", 32768
            )
            logger.info(
                "Using rope scaling: %s, factor: %s, original max position embeddings: %s, "
                "effective max_model_len: %s",
                rope_type,
                factor,
                original_max_position_embeddings,
                effective_max_model_len,
            )
            llm_init_kwargs["rope_scaling"] = {
                "rope_type": rope_type,
                "factor": factor,
                "original_max_position_embeddings": original_max_position_embeddings,
            }
        else:
            logger.info(
                "No rope scaling defined; using max_model_len=%s",
                effective_max_model_len,
            )
        self.model = LLM(**llm_init_kwargs)

    def do_experiment(self, print_output: bool = False):
        disable_fallback = os.environ.get("PCP_DISABLE_OOM_FALLBACK", "").lower() in {
            "1",
            "true",
            "yes",
        }
        profiles: List[Dict[str, Any]] = (
            [{}] if disable_fallback else self._default_oom_fallback_profiles()
        )
        last_error: BaseException | None = None

        for idx, profile in enumerate(profiles, start=1):
            try:
                if profile:
                    logger.info(
                        "OOM fallback attempt %s/%s with overrides: %s",
                        idx,
                        len(profiles),
                        profile,
                    )
                self._load_model(profile)
                return super().do_experiment(print_output)
            except Exception as exc:
                self._unload_model()
                if not self._looks_like_oom(exc) or idx == len(profiles):
                    raise
                last_error = exc
                logger.warning(
                    "GPU OOM with profile %s; retrying with smaller settings (%s/%s)",
                    profile or "default",
                    idx + 1,
                    len(profiles),
                )

        if last_error is not None:
            raise last_error

    # fed

    def _get_price_1_million_input_token(self) -> float:
        raise NotImplementedError(
            f"Input pricing per 1 million tokens is not implemented for {self.args.model_name}!"
        )

    # fed

    def _get_price_1_million_output_token(self) -> float:
        raise NotImplementedError(
            f"Output pricing per 1 million tokens is not implemented for {self.args.model_name}!"
        )

    # fed

    def run(self, dataset: List, print_output: bool = False):
        """
        Run the experiments on the provided dataset using vllm in batch mode.
        """
        prompt_list = [self._prepare_prompt(p) for p in dataset]
        processed = [p is not None for p in prompt_list]
        prompt_list = [p for p in prompt_list if p is not None]
        logger.info(f"Total prompts to process: {len(prompt_list)}")
        chats = [self._assemble_chat(prompt) for prompt in prompt_list]
        responses = self.vllm_inference(self._construct_vllm_dataset(chats))
        logger.info(f"Total responses received: {len(responses)}")
        su.io.dump(
            Macros.tmp_dir / "tmp-responses.json",
            {"responses": responses, "prompts": prompt_list},
        )
        results = self._aggregate_results(dataset, responses, prompt_list, processed)
        return results

    def vllm_inference(self, dataset: TextDataset):
        """
        Run the inference using vllm.

        All prompts are handed to vLLM at once (per checkpoint chunk) so the
        engine's continuous batching keeps the GPU saturated and avoids the
        head-of-line blocking of manual mini-batches. How many sequences run
        concurrently is governed by ``max_num_seqs`` (set when the engine is
        loaded), not by how the prompt list is sliced here. Chunking only
        bounds how often intermediate safety checkpoints are written.
        """
        cfg = self._effective_config()
        n = cfg.get("n", 1)
        checkpoint_every = max(
            1,
            int(
                cfg.get(
                    "inference_checkpoint_every", DEFAULT_INFERENCE_CHECKPOINT_EVERY
                )
            ),
        )
        prompts = [dataset[i] for i in range(len(dataset))]

        sampling_args = {
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "repetition_penalty": 1.05,
            "n": n,
            "seed": self.args.random_seed,
        }
        if sampling_args["n"] == 1 and sampling_args["temperature"] != 0:
            logger.error(
                "Temperature should be 0 when n is 1 for deterministic output."
            )
        if "max_new_tokens" in cfg:
            sampling_args["max_tokens"] = cfg["max_new_tokens"]
        sampling_params = SamplingParams(**sampling_args)

        responses: List[Any] = []
        total = len(prompts)
        for start in range(0, total, checkpoint_every):
            # Save intermediate results periodically (vLLM streams its own
            # per-call progress bar with throughput stats).
            su.io.dump(
                Macros.tmp_dir / f"ckpt-{str(self.args)}-{datetime.now()}.jsonl",
                responses,
            )
            chunk = prompts[start : start + checkpoint_every]
            chunk_outputs = self.model.generate(chunk, sampling_params)
            if n == 1:
                chunk_response = [output.outputs[0].text for output in chunk_outputs]
            else:
                chunk_response = [
                    [sample.text for sample in output.outputs]
                    for output in chunk_outputs
                ]
            responses.extend(chunk_response)
            logger.info("Generated %s/%s responses", len(responses), total)

        return responses

    def _construct_vllm_dataset(self, chats: List[str]):
        if self.tokenizer_mode == "mistral":
            prompt_all = [
                {"prompt_token_ids": self.tokenizer.apply_chat_template(msg)}
                for msg in chats
            ]
        else:
            prompt_all = [
                self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in chats
            ]
        dataset = TextDataset(prompt_all)
        return dataset

    def _aggregate_results(
        self, dataset: List, responses: List, chats: List, processed: List
    ):
        results = []
        if self.args.task != "igaf":
            assert len(dataset) == len(responses), (
                f"dataset and responses length mismatch: {len(dataset)} vs {len(responses)}"
            )
        # if chats contains None, skip corresponding dataset entry
        dataset = [dt for dt, proc in zip(dataset, processed) if proc]

        for dt, chat, response in zip(dataset, chats, responses):
            res = {}
            res.update(
                {
                    key: value
                    for key, value in dt.items()
                    if key not in {"syntax", "semantics"}
                }
            )
            # res["model-input"] = chat
            res["model-prediction"] = response
            results.append(res)
        # for

        return results


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(VLLMRunner, as_positional=False)
