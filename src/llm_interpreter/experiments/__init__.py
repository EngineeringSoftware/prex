"""
Package for experiment-related code, including experiment runners and experiment functions.
"""

import platform

if platform.system() == "Linux":
    from llm_interpreter.experiments.runners import (
        VLLMRunner,
        GPTRunner,
        GeminiRunner,
        OllamaRunner,
    )
else:
    from llm_interpreter.experiments.runners import (
        GPTRunner,
        GeminiRunner,
        OllamaRunner,
    )    
from llm_interpreter.experiments.base import ExperimentArgs
from llm_interpreter.experiments.prompts import PROMPT_STRATEGY
from llm_interpreter.experiments.functions import *  # noqa: F403


if platform.system() == "Linux":
    __all__ = [
        "VLLMRunner",
        "GPTRunner",
        "GeminiRunner",
        "ExperimentArgs",
        "PROMPT_STRATEGY",
        "OllamaRunner",
        # All experiment functions will be included via __all__ from functions.py
    ]
else:
    __all__ = [
        "GPTRunner",
        "GeminiRunner",
        "ExperimentArgs",
        "PROMPT_STRATEGY",
        "OllamaRunner",
        # All experiment functions will be included via __all__ from functions.py
    ]
