"""
For any experiment for the llm-interpreter paper (ICLR 2026).

Methods should be named experiment_<method_name> and annotated
with  @subcommand(category=Category.EXPERIMENT).

Example:
@subcommand(category=Category.EXPERIMENT)
def experiment_gpt_4o_translate_mk_imp_sos():
    model_name = "gpt-4o-mini"
    ...
"""

import platform
import seutil as su

from llm_interpreter.utils import (
    subcommand,
    Category,
)
from llm_interpreter.experiments.args import ExperimentArgs
from llm_interpreter.experiments.prompts import PROMPT_STRATEGY
if platform.system() == "Linux":
    from llm_interpreter.experiments.runners import (
        GPTRunner,
        GeminiRunner,
        OllamaRunner,
        VLLMRunner,
    )
else:
    from llm_interpreter.experiments.runners import (
        GPTRunner,
        GeminiRunner,
        OllamaRunner,
    )
#fi 

from llm_interpreter.macros import Macros
from llm_interpreter.experiments.runners.gpt_experiment import GPT_MODEL_ENUM

logger = su.log.get_logger(__name__, su.log.INFO)


@subcommand(category=Category.EXPERIMENT)
def gpt_4o_translate_mk_imp_sos():
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="translate",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-gpt-4o.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_translate_mk_imp_sos():
    """
    Translate IMP-SOS programs to the semantic-equivalent Python programs under mutated semantics.
    """
    model_name = "gemini-2.5-pro-preview-05-06"
    exp_args = ExperimentArgs(
        task="translate",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gemini_runner.do_experiment()


if platform.system() == "Linux":
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_ig_uk_imp_sos_ebnf():
        """
        Generate interpreter written different PLs for the IMP-SOS-EBNF language
        """
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="ig",
            setup_name="uk",
            expr_name="IMP-SOS-EBNF",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "ig/uk/imp-sos/config-qwen25-coder-14b-tokyo-multi-sampling.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()


    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_ig_uk_imp_sos_antlr():
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="ig",
            setup_name="uk",
            expr_name="IMP-SOS-ANTLR",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "ig/uk/imp-sos/config-qwen25-coder-14b-tokyo-multi-sampling.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()


    @subcommand(category=Category.EXPERIMENT)
    def deepseek_coder_v2_lite_ig_uk_imp_sos_antlr():
        model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
        exp_args = ExperimentArgs(
            task="ig",
            setup_name="uk",
            expr_name="IMP-SOS-ANTLR",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "ig/uk/imp-sos/config-deepseek-coderv2-lite-tokyo-multi-sampling.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()


    @subcommand(category=Category.EXPERIMENT)
    def deepseek_coder_v2_lite_ig_uk_imp_sos_ebnf():
        model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
        exp_args = ExperimentArgs(
            task="ig",
            setup_name="uk",
            expr_name="IMP-SOS-EBNF",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "ig/uk/imp-sos/config-deepseek-coderv2-lite-tokyo-multi-sampling.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

        
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_iga_uk_imp_sos():
        """
        Generate interpreter using ANTLR visitors with ANTLR generated parser
        """
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="iga",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "iga/uk/imp-sos/config-qwen25-coder-14b-tokyo-multi-sampling.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()


    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_iga_mk_imp_sos():
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="iga",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "iga/mk/imp-sos/config-qwen25-coder-14b-tokyo-multi-sampling.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()


    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_igaf_mk_imp_sos():
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="igaf",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "igaf/config-qwen25-coder-14b-tokyo-multi-sampling.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()


    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_igaf_uk_imp_sos():
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="igaf",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "igaf/config-qwen25-coder-14b-tokyo-multi-sampling.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()


#fi

@subcommand(category=Category.EXPERIMENT)
def gpt_4o_mini_ig_uk_imp_sos():
    """
    Generate an interpreter written in 4 programming languages for the IMP-SOS language under ANTLR and EBNF grammars
    """
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="ig",
        setup_name="uk",
        expr_name="IMP-SOS-ANTLR",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "ig/uk/imp-sos/config-gpt-4o-mini-multi-sampling.yaml",
        args=exp_args,
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
    )
    gpt_runner.do_experiment()

    # EBNF
    exp_args.expr_name = "IMP-SOS-EBNF"
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def o3_mini_ig_uk_imp_sos():
    """
    Generate an interpreter written in 4 programming languages for the IMP-SOS language under ANTLR and EBNF grammars
    """
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="ig",
        setup_name="uk",
        expr_name="IMP-SOS-ANTLR",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "ig/uk/imp-sos/config-o3-mini-multi-sampling.yaml",
        args=exp_args,
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
    )
    # gpt_runner.do_experiment()

    # EBNF
    exp_args.expr_name = "IMP-SOS-EBNF"
    gpt_runner.do_experiment()




@subcommand(category=Category.EXPERIMENT)
def gpt_4o_mini_iga_uk_imp_sos():
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="iga",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "iga/uk/imp-sos/config-gpt-4o-mini-multi-sampling.yaml",
        args=exp_args,
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI_SNAPSHOT,
        use_azure=False,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gpt_4o_mini_igaf_uk_imp_sos():
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="igaf",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "iga/uk/imp-sos/config-gpt-4o-mini-multi-sampling.yaml",
        args=exp_args,
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI_SNAPSHOT,
        use_azure=False,
    )
    gpt_runner.do_experiment()


# ollama qwq32b translate mk


@subcommand(category=Category.EXPERIMENT)
def ollama_qwq_32b_translate_mk_imp_sos():
    exp_args = ExperimentArgs(
        task="translate",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name="qwq:32b-fp16",
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    ollama_runner = OllamaRunner(
        model_config_file=Macros.model_config_dir
        / "translate/mk/imp-sos/config-ollama-qwq-32b.yaml",
        args=exp_args,
    )
    ollama_runner.do_experiment()


# ollama qwen2.5-coder:32b translate mk


@subcommand(category=Category.EXPERIMENT)
def ollama_qwen2_5_coder_32b_translate_mk_imp_sos():
    exp_args = ExperimentArgs(
        task="translate",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name="qwen2.5-coder:32b-instruct-fp16",
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    ollama_runner = OllamaRunner(
        model_config_file=Macros.model_config_dir
        / "translate/mk/imp-sos/config-ollama-qwen2.5-coder-32b.yaml",
        args=exp_args,
    )
    ollama_runner.do_experiment()


# ollama qwen2.5-coder:14b translate mk


@subcommand(category=Category.EXPERIMENT)
def ollama_qwen2_5_coder_14b_translate_mk_imp_sos():
    exp_args = ExperimentArgs(
        task="translate",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name="qwen2.5-coder:14b-instruct-fp16",
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    ollama_runner = OllamaRunner(
        model_config_file=Macros.model_config_dir
        / "translate/mk/imp-sos/config-ollama-qwen2.5-coder-14b.yaml",
        args=exp_args,
    )
    ollama_runner.do_experiment()


# ollama qwq32b translate uk


@subcommand(category=Category.EXPERIMENT)
def ollama_qwq_32b_translate_uk_imp_sos():
    exp_args = ExperimentArgs(
        task="translate",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name="qwq:32b-fp16",
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    ollama_runner = OllamaRunner(
        model_config_file=Macros.model_config_dir
        / "translate/uk/imp-sos/config-ollama-qwq-32b.yaml",
        args=exp_args,
    )
    ollama_runner.do_experiment()


# ollama qwen2.5-coder:32b translate uk


@subcommand(category=Category.EXPERIMENT)
def ollama_qwen2_5_coder_32b_translate_uk_imp_sos():
    exp_args = ExperimentArgs(
        task="translate",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name="qwen2.5-coder:32b-instruct-fp16",
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    ollama_runner = OllamaRunner(
        model_config_file=Macros.model_config_dir
        / "translate/uk/imp-sos/config-ollama-qwen2.5-coder-32b.yaml",
        args=exp_args,
    )
    ollama_runner.do_experiment()


# ollama qwen2.5-coder:14b translate uk


@subcommand(category=Category.EXPERIMENT)
def ollama_qwen2_5_coder_14b_translate_uk_imp_sos():
    exp_args = ExperimentArgs(
        task="translate",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name="qwen2.5-coder:14b-instruct-fp16",
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    ollama_runner = OllamaRunner(
        model_config_file=Macros.model_config_dir
        / "translate/uk/imp-sos/config-ollama-qwen2.5-coder-14b.yaml",
        args=exp_args,
    )
    ollama_runner.do_experiment()
