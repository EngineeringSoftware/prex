"""
For any results analysis code for the llm-interpreter paper (ICLR 2026).

Methods should be named experiment_<method_name> and annotated
with  @subcommand(category=Category.RESULT).

"""

import seutil as su

from llm_interpreter.utils import (
    subcommand,
    Category,
)
from llm_interpreter.experiments.prompts import PROMPT_STRATEGY
from llm_interpreter.results.results_analysis import ResultsAnalyzer


logger = su.log.get_logger(__name__, su.log.INFO)


@subcommand(category=Category.RESULT)
def result_translate_mk_imp_sos():
    """
    Collect the models' results of the translate-mk IMP SOS semantics.
    """
    analyzer = ResultsAnalyzer(
        task="translate",
        setup_name="mk",
        exp_name="IMP-SOS",
        model_list=[
            "gpt-4o-mini",
            "qwen2.5-coder:32b",
            "qwq:32b",
            "qwq:32b-fp16",
            "gemini-2.5-pro-preview-05-06",  # ADD
            "qwen2.5-coder:14b",  # ADD
            "qwen2.5-coder:14b-instruct-fp16",  # ADD
            "qwen2.5-coder:32b-instruct-fp16",  # ADD
        ],
    )
    analyzer.analyze_results(prompt_strategies=[PROMPT_STRATEGY.DA])


@subcommand(category=Category.RESULT)
def result_translate_uk_imp_sos():
    """
    Collect the models' results of the translate-uk IMP SOS semantics.
    """
    analyzer = ResultsAnalyzer(
        task="translate",
        setup_name="uk",
        exp_name="IMP-SOS",
        model_list=[
            # "gpt-4o-mini",
            # "gemini-2.5-pro-preview-05-06",
            "qwq:32b-fp16",
            "qwen2.5-coder:14b-instruct-fp16",
            "qwen2.5-coder:32b-instruct-fp16",
        ],
    )
    analyzer.analyze_results(prompt_strategies=[PROMPT_STRATEGY.DA])


@subcommand(category=Category.RESULT)
def result_ig_uk_imp_sos_ebnf():
    """
    Collect the models' results of the ig-uk IMP-SOS-EBNF.
    """
    analyzer = ResultsAnalyzer(
        task="ig",
        setup_name="uk",
        exp_name="IMP-SOS-EBNF",
        model_list=[
            "Qwen-Qwen2.5-Coder-14B-Instruct",
            "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct",
            "gpt-4o-mini",
            "o3-mini",
        ],
    )
    analyzer.analyze_results(prompt_strategies=[PROMPT_STRATEGY.DA])


@subcommand(category=Category.RESULT)
def result_ig_uk_imp_sos_antlr():
    """
    Collect the models' results of the ig-uk IMP-SOS-ANTLR.
    """
    analyzer = ResultsAnalyzer(
        task="ig",
        setup_name="uk",
        exp_name="IMP-SOS-ANTLR",
        model_list=[
            "Qwen-Qwen2.5-Coder-14B-Instruct",
            "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct",
            "gpt-4o-mini",
            "o3-mini",
        ],
    )
    analyzer.analyze_results(prompt_strategies=[PROMPT_STRATEGY.DA])


@subcommand(category=Category.RESULT)
def result_iga_uk_imp_sos():
    """
    Collect the models' results of the iga-uk IMP-SOS.
    """
    analyzer = ResultsAnalyzer(
        task="iga",
        setup_name="uk",
        exp_name="IMP-SOS",
        model_list=[
            "Qwen-Qwen2.5-Coder-14B-Instruct",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct",
            "gpt-4o-mini",
            # "o3-mini",
        ],
    )
    analyzer.analyze_results(prompt_strategies=[PROMPT_STRATEGY.DA])


@subcommand(category=Category.RESULT)
def result_iga_mk_imp_sos():
    """
    Collect the models' results of the iga-mk IMP-SOS.
    """
    analyzer = ResultsAnalyzer(
        task="iga",
        setup_name="mk",
        exp_name="IMP-SOS",
        model_list=[
            "Qwen-Qwen2.5-Coder-14B-Instruct",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct",
            # "gpt-4o-mini",
            # "o3-mini",
        ],
    )
    analyzer.analyze_results(prompt_strategies=[PROMPT_STRATEGY.DA])


@subcommand(category=Category.RESULT)
def result_igaf_uk_imp_sos():
    analyzer = ResultsAnalyzer(
        task="igaf",
        setup_name="uk",
        exp_name="IMP-SOS",
        model_list=[
            "Qwen-Qwen2.5-Coder-14B-Instruct",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct",
            "gpt-4o-mini",
            # "o3-mini",
        ],
    )
    analyzer.analyze_results(prompt_strategies=[PROMPT_STRATEGY.DA])


@subcommand(category=Category.RESULT)
def result_igaf_mk_imp_sos():
    analyzer = ResultsAnalyzer(
        task="igaf",
        setup_name="mk",
        exp_name="IMP-SOS",
        model_list=[
            "Qwen-Qwen2.5-Coder-14B-Instruct",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct",
            # "gpt-4o-mini",
            # "o3-mini",
        ],
    )
    analyzer.analyze_results(prompt_strategies=[PROMPT_STRATEGY.DA])
