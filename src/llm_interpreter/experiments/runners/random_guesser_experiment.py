"""
Random-guesser baseline for PCP (PredExe) experiments.

Samples predictions from the empirical label distribution of the PCP dataset
without conditioning on the program or semantics.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple

import seutil as su

from llm_interpreter.data.dataset_process import DatasetProcessor
from llm_interpreter.experiments.base import BaseRunner

logger = su.log.get_logger(__name__, su.log.INFO)

# Empirical label distribution on the uk PCP dataset (2946 samples).
PCP_DATASET_LABEL_DISTRIBUTION: Tuple[Tuple[str, float], ...] = (
    ("success", 1 / 6),
    ("continue_outside_loop", 1 / 6),
    ("modulo_zero", 1 / 6),
    ("break_outside_loop", 1 / 6),
    ("divide_by_zero", 1 / 6),
    ("var_use_before_declare", 1 / 6),
)


def _semantics_type_from_expr_name(expr_name: str) -> str:
    if "K" in expr_name:
        return "K"
    return "SOS"


def sample_pcp_prediction(
    rng: random.Random,
    semantics_type: str,
) -> str:
    """
    Sample a PCP prediction from the dataset label distribution.

    Returns model output in the same tagged format expected by extract_pcp_prediction.
    """
    labels, weights = zip(*PCP_DATASET_LABEL_DISTRIBUTION)
    label = rng.choices(labels, weights=weights, k=1)[0]
    if label == "success":
        return "<ans>##success##</ans>"
    rule = DatasetProcessor.SEMANTIC_INVALID_RULES_IMP[semantics_type][label]
    rule_number = re.search(r"\d+", rule).group(0)
    return f"<ans>##error##</ans>\n<rule>{rule_number}</rule>"


class RandomGuesserRunner(BaseRunner):
    """Baseline runner that ignores the prompt and samples from label priors."""

    MODEL_NAME = "random-guesser-dataset-distribution"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_config = {}
        self._rng = random.Random(self.args.random_seed)
        self._semantics_type = _semantics_type_from_expr_name(self.args.expr_name)

    def _get_price_1_million_input_token(self) -> float:
        return 0.0

    def _get_price_1_million_output_token(self) -> float:
        return 0.0

    def _query(self, chat: List[dict], stop: List[str] | None = None) -> List[str]:
        return [sample_pcp_prediction(self._rng, self._semantics_type)]

    def query_llm(
        self, dt: Dict, chat_prompt: List[dict], print_output: bool = False
    ) -> Dict:
        res = {
            key: value
            for key, value in dt.items()
            if key not in {"syntax", "semantics"}
        }
        prediction = sample_pcp_prediction(self._rng, self._semantics_type)
        res["model-prediction"] = prediction
        if print_output:
            print(prediction)
        return res
