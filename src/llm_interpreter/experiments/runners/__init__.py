"""
Experiment runner implementations.
"""

import platform
if platform.system() == "Linux":
    from llm_interpreter.experiments.runners.vllm_experiment import VLLMRunner
#fi
from llm_interpreter.experiments.runners.gpt_experiment import GPTRunner, GPT_MODEL_ENUM
from llm_interpreter.experiments.runners.gemini_experiment import GeminiRunner
from llm_interpreter.experiments.runners.ollama_experiment import OllamaRunner
from llm_interpreter.experiments.runners.batch_experiment import BatchRunner
from llm_interpreter.experiments.runners.random_guesser_experiment import (
    RandomGuesserRunner,
)

if platform.system() == "Linux":
    __all__ = [
        "VLLMRunner",
        "GPTRunner",
        "GeminiRunner",
        "GPT_MODEL_ENUM",
        "OllamaRunner",
        "BatchRunner",
        "RandomGuesserRunner",
    ]
else:
    __all__ = [
        "GPTRunner",
        "GeminiRunner",
        "GPT_MODEL_ENUM",
        "OllamaRunner",
        "BatchRunner",
        "RandomGuesserRunner",
    ]
#fi
