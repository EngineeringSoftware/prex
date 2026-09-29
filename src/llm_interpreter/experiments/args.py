"""
Experiment arguments class.
"""

from dataclasses import dataclass, fields
from typing import Optional

from llm_interpreter.experiments.prompts import PROMPT_STRATEGY


@dataclass
class ExperimentArgs:
    """Arguments for experiment configuration."""

    task: str
    setup_name: str
    expr_name: str
    model_name: str
    dataset_name: Optional[str] = "human_written"
    prompt_strategy: Optional[str] = PROMPT_STRATEGY.DA
    random_seed: Optional[int] = 42
    shard: Optional[int] = 0
    total_shards: Optional[int] = 1

    def __str__(self):
        lines = []
        for field in fields(self):
            value = getattr(self, field.name)
            # exclude run-control knobs from the string representation so that
            # output paths and filenames stay stable across shard/seed changes
            if field.name in {"random_seed", "shard", "total_shards"}:
                continue
            lines.append(value)
        return "-".join(lines).replace("/", "-")
