"""
GPT experiment runner implementation.
"""

import os
from typing import Union
from pathlib import Path
import seutil as su
import dataclasses
import enum
from openai import OpenAI
# from openai import AzureOpenAI

from llm_interpreter.experiments.base import BaseRunner

logger = su.log.get_logger(__name__, su.log.INFO)


@dataclasses.dataclass(frozen=True)
class GPT_MODEL:
    name: str = ""
    reasoning: bool = False
    from_openai: bool = False
    api_base: str = None
    pricing_input_tokens: float = 0.0
    pricing_output_tokens: float = 0.0


class GPT_MODEL_ENUM(enum.Enum):
    GPT_4o = GPT_MODEL(
        name="gpt-4o",
        reasoning=False,
        from_openai=True,
        api_base=None,
        pricing_input_tokens=2.5,
        pricing_output_tokens=10.0,
    )
    GPT_4o_MINI = GPT_MODEL(
        name="gpt-4o-mini",
        reasoning=False,
        from_openai=True,
        api_base=None,
        pricing_input_tokens=0.15,
        pricing_output_tokens=0.6,
    )
    GPT_4o_MINI_SNAPSHOT = GPT_MODEL(
        name="gpt-4o-mini-2024-07-18",
        reasoning=False,
        from_openai=True,
        api_base=None,
        pricing_input_tokens=0.15,
        pricing_output_tokens=0.6,
    )
    GPT_5_MINI = GPT_MODEL(
        name="gpt-5-mini",
        reasoning=True,
        from_openai=True,
        api_base=None,
        pricing_input_tokens=0.25,
        pricing_output_tokens=2.0,
    )
    GPT_5_4_MINI = GPT_MODEL(
        name="gpt-5.4-mini",
        reasoning=True,
        from_openai=True,
        api_base=None,
        pricing_input_tokens=0.375,
        pricing_output_tokens=2.25,
    )
    GPT_5_4 = GPT_MODEL(
        name="gpt-5.4",
        reasoning=True,
        from_openai=True,
        api_base=None,
        pricing_input_tokens=0.375,
        pricing_output_tokens=2.25,
    )
    O3_MINI = GPT_MODEL(
        name="o3-mini",
        reasoning=True,
        from_openai=True,
        api_base=None,
        pricing_input_tokens=1.1,
        pricing_output_tokens=4.4,
    )
    LLAMA_3_70B = GPT_MODEL(
        name="meta-llama/Llama-3.3-70B-Instruct",
        reasoning=False,
        from_openai=False,
        api_base="http://localhost:8000/v1",
    )
    DeepSeek_LLAMA_3_70B = GPT_MODEL(
        name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        reasoning=True,
        from_openai=False,
        api_base="http://localhost:8000/v1",
    )


class GPTRunner(BaseRunner):
    def __init__(
        self,
        model_config_file: Union[str, Path],
        gpt_model: GPT_MODEL_ENUM,
        use_azure: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # load model's config
        self.model_config = su.io.load(model_config_file)
        self.gpt_model = gpt_model
        self.use_azure = use_azure
        self.setup_client()

    # fed

    def setup_client(self):
        if self.gpt_model.value.from_openai:
            if self.use_azure:
                self.client = OpenAI(
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    base_url="https://jessy-m9yqp0aw-eastus2.services.ai.azure.com/models",
                    # api_version="2025-02-01-preview",
                )
            else:
                self.client = OpenAI()
        else:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=self.gpt_model.value.api_base,
            )
        #

    def _get_price_1_million_input_token(self) -> float:
        return self.gpt_model.value.pricing_input_tokens

    # fed

    def _get_price_1_million_output_token(self) -> float:
        return self.gpt_model.value.pricing_output_tokens

    # fed

    def _query(
        self,
        chat: list[dict],
        stop: list[str] = [],
    ) -> list[str]:
        try:
            completion_kwargs = {
                "model": self.gpt_model.value.name,
                "messages": chat,
                "stop": stop,
            }
            if self.gpt_model == GPT_MODEL_ENUM.GPT_5_MINI or self.gpt_model == GPT_MODEL_ENUM.GPT_5_4_MINI or self.gpt_model == GPT_MODEL_ENUM.GPT_5_4:
                del completion_kwargs["stop"]
            # fi
            if self.gpt_model == GPT_MODEL_ENUM.GPT_5_4_MINI or self.gpt_model == GPT_MODEL_ENUM.GPT_5_4 or self.gpt_model == GPT_MODEL_ENUM.GPT_5_MINI:
                completion_kwargs["reasoning_effort"]="medium"
            # fi
            
            completion_kwargs["n"] = self.model_config.get("n", 1)
            if self.gpt_model.value.reasoning and self.gpt_model.value.from_openai:
                completion_kwargs["max_completion_tokens"] = self.model_config.get(
                    "max_completion_tokens", 16000
                )
            else:
                completion_kwargs["max_completion_tokens"] = self.model_config.get(
                    "max_completion_tokens", 2048
                )
                completion_kwargs["temperature"] = self.model_config.get(
                    "temperature", 0
                )
            response = self.client.chat.completions.create(**completion_kwargs)
            return [c.message.content for c in response.choices]
        except Exception as e:
            # raise ModelRunnerException(
            logger.warning(
                f"Error while running query with model {self.args.model_name} : {e}"
            )
            return []
        # yrt

    # fed


# ssalc
