"""
For any experiment that generates results/data for the paper.

Methods should be named experiment_<method_name> and annotated
with  @subcommand(category=Category.EXPERIMENT)
"""

import os
import glob
import time
import random
import platform
import seutil as su
from tqdm import tqdm

from llm_interpreter.macros import Macros
from llm_interpreter.language import Language, IMP
from llm_interpreter.compiler_runners import KFramework
import numpy as np
from datasets import load_dataset, Dataset
from collections import defaultdict

if platform.system() == "Linux":
    from llm_interpreter.experiments.runners import (
        VLLMRunner,
        GPTRunner,
        GeminiRunner,
        BatchRunner,
        RandomGuesserRunner,
    )
else:
    from llm_interpreter.experiments.runners import (
        GPTRunner,
        GeminiRunner,
        RandomGuesserRunner,
    )

from llm_interpreter.experiments.runners.gpt_experiment import GPT_MODEL_ENUM
from llm_interpreter.experiments.prompts import PROMPT_STRATEGY
from llm_interpreter.experiments.args import ExperimentArgs
from llm_interpreter.utils import (
    subcommand,
    Category,
    write_to_tmp,
)
from llm_interpreter.tacc_runners.tacc_helper import TACCCoordinator
from llm_interpreter.experiments.intp_functions import *  # noqa: F403

logger = su.log.get_logger(__name__, su.log.INFO)


@subcommand(category=Category.EXPERIMENT)
def experiment_evaluate_imp_with_k(imp_program_dir: str, output_file: str = None):
    """Executes all IMP programs (`*.imp`) in the directory `imp_program_dir` using the K-framework and writes their evaluation statuses to the file `output_file`." """
    imp_lang1 = IMP("IMP1", Language.SEMANTICS_TYPE.K)
    write_to_tmp(imp_lang1.get_semantics(), "IMP1.k")
    k_framework = KFramework()
    k_framework.compile_k_specification(
        k_file=f"{Macros.tmp_dir}/IMP1.k", output_dir=f"{Macros.tmp_dir}/IMP1"
    )

    imp_programs: list[str] = glob.glob(f"{imp_program_dir}/*.imp")
    imp_program_statuses: list = []
    valid_cnt, invalid_cnt = 0, 0
    for program in tqdm(
        imp_programs, total=len(imp_programs), desc="Evaluating IMP programs with K"
    ):
        program_status: dict = {}
        program_status["name"] = os.path.basename(program)
        try:
            output: str = k_framework.run_program(program_file=program)
            k_result = KFramework.parse_k_framework_output(output)
            program_status["K-evaluatable"] = str(True)
            program_status["trace"] = k_result.get_execution_trace()
            program_status["final_state"] = k_result.get_final_state()
            valid_cnt += 1
        except Exception as e:
            logger.warning(e)
            program_status["K-evaluatable"] = str(False)
            invalid_cnt += 1
        # yrt
        imp_program_statuses.append(program_status)
    # rof
    if output_file:
        su.io.dump(output_file, imp_program_statuses)
    else:
        print(imp_program_statuses)
    logger.info(
        f"Number of valid program is {valid_cnt}; invalid program is {invalid_cnt}"
    )
    # fi


# fed


@subcommand(category=Category.EXPERIMENT)
def experiment_generate_imp_arithmetic_expr_rule_order_test(
    num_expr: int, max_num_terms: int
):
    imp_lang1 = IMP("IMPBASE", Language.SEMANTICS_TYPE.K)
    imp_lang1.build_visitors()
    expr_list: list = []
    for i in range(num_expr):
        random.seed(time.time())
        expr_data: dict = {}
        [expr, state, op_count] = imp_lang1.generate_arithmetic_expr(
            random.randint(2, max_num_terms), random.randint(1, max_num_terms // 2)
        )
        expr_data["expr"] = expr
        expr_data["state"] = state
        # Have to wrap expr into assign_stmt since expr are
        # not valid entry points into the parser
        assign_expr = f"{random.choice(list(state.keys()))} = {expr};"
        rules = imp_lang1.get_rule(assign_expr, state)
        # We pop assignment rule Rule 4
        rules.pop()
        expr_data["rules"] = rules
        expr_list.append(expr_data)
    # rof
    su.io.dump(Macros.pytest_oracles_dir / "imp_arithmetic_exprs.jsonl", expr_list)


# fed


@subcommand(category=Category.EXPERIMENT)
def experiment_generate_imp_arithmetic_expr_op_count_test(
    num_expr: int, max_num_terms: int
):
    imp_lang1 = IMP("IMPBASE", Language.SEMANTICS_TYPE.K)
    imp_lang1.build_visitors()
    expr_list: list = []
    for i in range(num_expr):
        random.seed(time.time())
        expr_data: dict = {}
        [expr, state, op_count] = imp_lang1.generate_arithmetic_expr(
            random.randint(2, max_num_terms), random.randint(1, max_num_terms // 2)
        )
        expr_data["expr"] = expr
        expr_data["state"] = state
        expr_data["rule_count"] = op_count
        expr_list.append(expr_data)
    # rof
    su.io.dump(
        Macros.pytest_oracles_dir / "imp_arithmetic_expr_op_count.jsonl", expr_list
    )


@subcommand(category=Category.EXPERIMENT)
def experiment_generate_imp_boolean_expr_op_count_test(
    num_expr: int, max_num_terms: int
):
    imp_lang1 = IMP("IMPBASE", Language.SEMANTICS_TYPE.K)
    imp_lang1.build_visitors()
    expr_list: list = []
    for i in range(num_expr):
        random.seed(time.time())
        expr_data: dict = {}
        [expr, expr_result, state, rule_count] = imp_lang1.generate_boolean_expr(
            random.randint(2, max_num_terms)
        )
        expr_data["expr"] = expr
        expr_data["state"] = state
        expr_data["rule_count"] = rule_count
        expr_data["result"] = str(expr_result)
        expr_list.append(expr_data)
    # rof
    su.io.dump(Macros.pytest_oracles_dir / "imp_boolean_expr_op_count.jsonl", expr_list)


# fed


## MODEL RUNNER EXPERIMENTS


# --
# nl2rule
# --
@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_nl2rule_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_nl2rule_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_o3_mini_uk_nl2rule_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_o3_mini_mk_nl2rule_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_mini_uk_nl2rule_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_mini_mk_nl2rule_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mini_uk_nl2rule_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mini_mk_nl2rule_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def gemini_uk_nl2rule_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers = 4,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gemini_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def gemini_mk_nl2rule_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers = 4,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gemini_runner.do_experiment()
#fed

if platform.system() == "Linux":
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_nl2rule_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed
    
#fi

# --
# nl2rule_K
# --
@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_nl2rule_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_nl2rule_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_o3_mini_uk_nl2rule_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_o3_mini_mk_nl2rule_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_mini_uk_nl2rule_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_mini_mk_nl2rule_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mini_uk_nl2rule_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mini_mk_nl2rule_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def gemini_uk_nl2rule_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers = 4,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "Standard_NumRule5_RandomSampleFalse"
    )
    gemini_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def gemini_mk_nl2rule_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="nl2rule",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers = 4,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "nl2rule_K",
        hf_split = "NonStandard_NumRule5_RandomSampleFalse"
    )
    gemini_runner.do_experiment()
#fed

if platform.system() == "Linux":
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "Standard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_nl2rule_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="nl2rule",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "nl2rule_K",
            hf_split = "NonStandard_NumRule5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed
    
#fi

# --
# rule2nl
# --
@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_rule2nl_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "Standard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_rule2nl_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_o3_mini_uk_rule2nl_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "Standard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_o3_mini_mk_rule2nl_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_mini_uk_rule2nl_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "Standard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_mini_mk_rule2nl_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mini_uk_rule2nl_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "Standard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mini_mk_rule2nl_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def gemini_uk_rule2nl_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers = 4,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "Standard_NumDescription5_RandomSampleFalse",
    )
    gemini_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def gemini_mk_rule2nl_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers = 4,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gemini_runner.do_experiment()

if platform.system() == "Linux":
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_rule2nl_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed
    
#fi

#fed

# --
# rule2nl_K
# --
@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_rule2nl_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "Standard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_rule2nl_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_o3_mini_uk_rule2nl_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "Standard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_o3_mini_mk_rule2nl_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_mini_uk_rule2nl_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "Standard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_mini_mk_rule2nl_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mini_uk_rule2nl_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "Standard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mini_mk_rule2nl_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gpt_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def gemini_uk_rule2nl_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers = 4,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "Standard_NumDescription5_RandomSampleFalse"
    )
    gemini_runner.do_experiment()
#fed

@subcommand(category=Category.EXPERIMENT)
def gemini_mk_rule2nl_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="rule2nl",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / "etp"
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers = 4,
        use_hf = True,
        hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
        hf_config = "rule2nl_K",
        hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
    )
    gemini_runner.do_experiment()

if platform.system() == "Linux":
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "Standard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_rule2nl_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="rule2nl",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            args=exp_args,
            use_hf = True,
            hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
            hf_config = "rule2nl_K",
            hf_split = "NonStandard_NumDescription5_RandomSampleFalse"
        )
        vllm_runner.do_experiment()
    #fed
    
#fi

#fed

# ---
# pcp tasks
# ---

@subcommand(category=Category.EXPERIMENT)
def random_guesser_uk_pcp_imp_sos(seed: int = 42):
    """PCP baseline: sample from the empirical uk dataset label distribution."""
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=RandomGuesserRunner.MODEL_NAME,
        prompt_strategy=PROMPT_STRATEGY.DA, # prompt strategy doesn't matter anyway because it doesn't care about the prompt
        random_seed=seed,
    )
    RandomGuesserRunner(args=exp_args).do_experiment()


@subcommand(category=Category.EXPERIMENT)
def random_guesser_uk_pcp_imp_k(seed: int = 42):
    """PCP baseline: sample from the empirical uk dataset label distribution."""
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=RandomGuesserRunner.MODEL_NAME,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    RandomGuesserRunner(args=exp_args).do_experiment()


@subcommand(category=Category.EXPERIMENT)
def random_guesser_mk_pcp_imp_sos(seed: int = 42):
    """PCP baseline on mk setup: same label prior as the uk dataset."""
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=RandomGuesserRunner.MODEL_NAME,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    RandomGuesserRunner(args=exp_args).do_experiment()


@subcommand(category=Category.EXPERIMENT)
def random_guesser_mk_pcp_imp_k(seed: int = 42):
    """PCP baseline on mk setup: same label prior as the uk dataset."""
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=RandomGuesserRunner.MODEL_NAME,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    RandomGuesserRunner(args=exp_args).do_experiment()


if platform.system() == "Linux":
    # qwen2.5-coder 3b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_uk_pcp_imp_sos(seed: int = 42, shard: int = 0, total_shards: int = 1):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
            shard=shard,
            total_shards=total_shards,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_uk_pcp_cot_imp_sos(seed: int = 42, shard: int = 0, total_shards: int = 1):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
            shard=shard,
            total_shards=total_shards,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_mk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # qwen2.5-coder 7b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_uk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_uk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_mk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_uk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_mk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_uk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_mk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # ministral 3 3b
    @subcommand(category=Category.EXPERIMENT)
    def ministral_3b_uk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-ministral-3-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_3b_uk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-ministral-3-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_3b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-ministral-3-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_3b_mk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-ministral-3-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_3b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-ministral-3-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_3b_uk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-ministral-3-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_3b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-ministral-3-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_3b_mk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-ministral-3-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # ministral 3 8b
    @subcommand(category=Category.EXPERIMENT)
    def ministral_8b_uk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-8B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-ministral-3-8b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_8b_uk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-8B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-ministral-3-8b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_8b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-8B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-ministral-3-8b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_8b_uk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-8B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-ministral-3-8b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_8b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-8B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-ministral-3-8b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_8b_mk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-8B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-ministral-3-8b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_8b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-8B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-ministral-3-8b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_8b_mk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-8B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-ministral-3-8b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # ministral 3 14b
    @subcommand(category=Category.EXPERIMENT)
    def ministral_14b_uk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-14B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-ministral-3-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_14b_uk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-14B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-ministral-3-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_14b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-14B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-ministral-3-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_14b_uk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-14B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-ministral-3-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_14b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-14B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-ministral-3-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_14b_mk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="mistralai/Ministral-3-14B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-ministral-3-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_14b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-14B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-ministral-3-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def ministral_14b_mk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="mistralai/Ministral-3-14B-Instruct-2512-BF16",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-ministral-3-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # qwen2.5-coder 14b

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # qwen-coder-2.5-32b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_pcp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_pcp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_pcp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_pcp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_pcp_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_pcp_cot_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_pcp_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_pcp_cot_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_pcp_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_pcp_cot_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_pcp_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_pcp_cot_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_pcp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_pcp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/uk/imp-k/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_pcp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_pcp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="pcp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / "pcp/mk/imp-k/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
# fi


# OpenAI-like models
# pcp
@subcommand(category=Category.EXPERIMENT)
def openai_o3_uk_pcp_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_uk_pcp_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_mk_pcp_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_mk_pcp_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_pcp_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_pcp_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_pcp_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_pcp_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_pcp_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_pcp_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_pcp_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_pcp_imp_cot_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.COT,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_pcp_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


# op
@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_nk_op_imp_sos(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name="claude-sonnet-4-6",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_uk_op_imp_sos(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name="claude-sonnet-4-6",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_uk_op_imp_k(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name="claude-sonnet-4-6",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_mk_op_imp_sos(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name="claude-sonnet-4-6",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_mk_op_imp_k(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name="claude-sonnet-4-6",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()
    
@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_nk_op_imp_sos_fuzzer_generated(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name="claude-sonnet-4-6",
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_uk_op_imp_sos_fuzzer_generated(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name="claude-sonnet-4-6",
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_uk_op_imp_k_fuzzer_generated(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name="claude-sonnet-4-6",
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_mk_op_imp_sos_fuzzer_generated(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name="claude-sonnet-4-6",
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_sonnet_4_6_mk_op_imp_k_fuzzer_generated(seed: int = 1):
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name="claude-sonnet-4-6",
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def openai_o3_nk_op_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_uk_op_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_uk_op_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_mk_op_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_mk_op_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_nk_op_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_nk_op_imp_sos_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        use_azure=False,
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_nk_op_imp_sos_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        use_azure=False,
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_nk_op_imp_sos_fuzzer_generated_reduced(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated_reduced",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        use_azure=False,
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_op_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_op_imp_sos_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_op_imp_sos_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_op_imp_sos_fuzzer_generated_reduced(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated_reduced",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_op_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_op_imp_k_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_op_imp_k_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_op_imp_k_fuzzer_generated_reduced(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="fuzzer_generated_reduced",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_op_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_op_imp_sos_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_op_imp_sos_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_op_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_op_imp_k_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_op_imp_k_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_regular_nk_op_imp_sos(seed: int = 1):
    model_name = "gpt-5.4"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_regular_mk_op_imp_sos(seed: int = 1):
    model_name = "gpt-5.4"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_nk_op_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_nk_op_imp_sos_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        use_azure=False,
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_nk_op_imp_sos_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        use_azure=False,
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_nk_op_imp_sos_fuzzer_generated_reduced(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated_reduced",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        use_azure=False,
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_op_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_op_imp_sos_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_op_imp_sos_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_op_imp_sos_fuzzer_generated_reduced(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated_reduced",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_op_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_op_imp_k_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_op_imp_k_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_op_imp_k_fuzzer_generated_reduced(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="fuzzer_generated_reduced",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_op_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_op_imp_sos_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_op_imp_sos_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_op_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_op_imp_k_synthetic_cpp(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="synthetic_cpp",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_op_imp_k_fuzzer_generated(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="fuzzer_generated",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_nk_op_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_nk_op_imp_sos_fuzzer_generated_reduced(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        dataset_name="fuzzer_generated_reduced",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_op_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_op_imp_sos_fuzzer_generated_reduced(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        dataset_name="fuzzer_generated_reduced",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_op_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_op_imp_k_fuzzer_generated_reduced(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        dataset_name="fuzzer_generated_reduced",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_op_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_op_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_op_imp_sos_unseen_gpt4o_1_token(
    seed: int = 1,
    hf_repo_id: str = "LambdaadbmaL/PredStateTest",
    hf_config: str = "predstate",
    hf_split: str = "imp_sos_mk_unseen_gpt4o_1_token_human_written",
    hf_token: str = os.environ.get("HF_TOKEN")
):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk-unseen-gpt4o-1-token",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / "mk"
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure=False,
        use_hf=True,
        hf_repo_id=hf_repo_id,
        hf_config=hf_config,
        hf_split=hf_split,
        hf_token=hf_token,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_op_imp_k_unseen_gpt4o_1_token(
    seed: int = 1,
    hf_repo_id: str = "LambdaadbmaL/PredStateTest",
    hf_config: str = "predstate",
    hf_split: str = "imp_k_mk_unseen_gpt4o_1_token_human_written",
    hf_token: str = os.environ.get("HF_TOKEN")
):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk-unseen-gpt4o-1-token",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / "mk"
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure=False,
        use_hf=True,
        hf_repo_id=hf_repo_id,
        hf_config=hf_config,
        hf_split=hf_split,
        hf_token=hf_token,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_op_imp_sos_unseen_caucasian_albanian(
    seed: int = 1,
    hf_repo_id: str = "LambdaadbmaL/PredStateTest",
    hf_config: str = "predstate",
    hf_split: str = "imp_sos_mk_unseen_caucasian_albanian_human_written",
    hf_token: str = os.environ.get("HF_TOKEN")
):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk-unseen-caucasian-albanian",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / "mk"
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure=False,
        use_hf=True,
        hf_repo_id=hf_repo_id,
        hf_config=hf_config,
        hf_split=hf_split,
        hf_token=hf_token,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_op_imp_k_unseen_caucasian_albanian(
    seed: int = 1,
    hf_repo_id: str = "LambdaadbmaL/PredStateTest",
    hf_config: str = "predstate",
    hf_split: str = "imp_k_mk_unseen_caucasian_albanian_human_written",
    hf_token: str = os.environ.get("HF_TOKEN")
):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk-unseen-caucasian-albanian",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / "mk"
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure=False,
        use_hf=True,
        hf_repo_id=hf_repo_id,
        hf_config=hf_config,
        hf_split=hf_split,
        hf_token=hf_token,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


# srp
@subcommand(category=Category.EXPERIMENT)
def batch_gpt_5_4_mini_uk_srp_imp_sos(seed: int = 1):
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name="gpt-5.4-mini",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_gpt_5_4_mini_uk_srp_imp_k(seed: int = 1):
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name="gpt-5.4-mini",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_gpt_5_4_mini_mk_srp_imp_sos(seed: int = 1):
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name="gpt-5.4-mini",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_gpt_5_4_mini_mk_srp_imp_k(seed: int = 1):
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name="gpt-5.4-mini",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def openai_o3_uk_srp_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_uk_srp_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_mk_srp_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_mk_srp_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_srp_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_srp_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_srp_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_srp_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_srp_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_srp_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_srp_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_srp_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_srp_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_srp_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_srp_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_srp_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


# etp
@subcommand(category=Category.EXPERIMENT)
def batch_gpt_5_4_mini_uk_etp_imp_sos(seed: int = 1):
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name="gpt-5.4-mini",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_gpt_5_4_mini_uk_etp_imp_k(seed: int = 1):
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name="gpt-5.4-mini",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_gpt_5_4_mini_mk_etp_imp_sos(seed: int = 1):
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name="gpt-5.4-mini",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def batch_gpt_5_4_mini_mk_etp_imp_k(seed: int = 1):
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name="gpt-5.4-mini",
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    batch_runner = BatchRunner(
        max_tokens = 32768,
        args=exp_args
    )
    batch_runner.do_experiment()

@subcommand(category=Category.EXPERIMENT)
def openai_o3_uk_etp_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_uk_etp_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_mk_etp_imp_sos(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_o3_mk_etp_imp_k(seed: int = 1):
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_etp_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_uk_etp_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_etp_imp_sos(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt5_mk_etp_imp_k(seed: int = 1):
    model_name = "gpt-5-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_etp_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_uk_etp_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_etp_imp_sos(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt_5_4_mk_etp_imp_k(seed: int = 1):
    model_name = "gpt-5.4-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-gpt-5-mini.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_5_4_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_etp_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_uk_etp_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_etp_imp_sos(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def openai_gpt4o_mini_mk_etp_imp_k(seed: int = 1):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        use_azure=False,
        args=exp_args,
    )
    gpt_runner.do_experiment()
    exp_args.prompt_strategy = PROMPT_STRATEGY.COT
    gpt_runner.args = exp_args
    gpt_runner.do_experiment()


# Gemini
# pcp task
@subcommand(category=Category.EXPERIMENT)
def gemini_uk_pcp_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_uk_pcp_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_pcp_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_pcp_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="pcp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


##
# op task
##
@subcommand(category=Category.EXPERIMENT)
def gemini_uk_op_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_uk_op_imp_sos_synthetic_cpp(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        dataset_name="synthetic_cpp",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=2,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_uk_op_imp_sos_fuzzer_generated(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-SOS",
        dataset_name="fuzzer_generated",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=2,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_uk_op_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_uk_op_imp_k_synthetic_cpp(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        dataset_name="synthetic_cpp",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=2,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_uk_op_imp_k_fuzzer_generated(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="uk",
        expr_name="IMP-K",
        dataset_name="fuzzer_generated",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=2,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_nk_op_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_nk_op_imp_sos_synthetic_cpp(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        dataset_name="synthetic_cpp",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_nk_op_imp_sos_fuzzer_generated(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="nk",
        expr_name="IMP-SOS",
        dataset_name="fuzzer_generated",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_op_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_op_imp_sos_synthetic_cpp(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        dataset_name="synthetic_cpp",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_op_imp_sos_fuzzer_generated(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-SOS",
        dataset_name="fuzzer_generated",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_op_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_op_imp_k_synthetic_cpp(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        dataset_name="synthetic_cpp",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_op_imp_k_fuzzer_generated(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="op",
        setup_name="mk",
        expr_name="IMP-K",
        dataset_name="fuzzer_generated",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


# srp
@subcommand(category=Category.EXPERIMENT)
def gemini_uk_srp_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_uk_srp_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_srp_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_srp_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="srp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


# etp
@subcommand(category=Category.EXPERIMENT)
def gemini_uk_etp_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_uk_etp_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="uk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gpt_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=4,
    )
    gpt_runner.do_experiment()

#fed


@subcommand(category=Category.EXPERIMENT)
def dataset_gen_unseen_one_token_gpt4o(
    hf_src_repo_id = "EngineeringSoftware/PLSemanticsBench",
    hf_src_config = "predstate-IMP-SOS-uk-human-written",
    hf_dst_repo_id = "LambdaadbmaL/PredStateTest",
    hf_dst_config = "predstate",
    hf_dst_split = "imp_sos_mk_unseen_gpt4o_1_token_human_written",
    hf_token = os.environ.get("HF_TOKEN")
):
    import llm_interpreter.data.dataset_process as dsp
    imp = IMP("IMPTEST", Language.SEMANTICS_TYPE.SOS, **dsp.DatasetProcessor.SEMANTICS_MUTATIONS["unseen-gpt4o-1-token"])
    dataset = load_dataset(hf_src_repo_id, name=hf_src_config, token=hf_token)['train']
    new_dataset = []
    for dt in dataset:
        dt['mutated-program'] = imp.translate_unmutated_program(dt['program'])
        if dt['semantics-type'] == "SOS":
            dt['syntax'] = imp.get_syntax()
        dt['semantics'] = imp.get_semantics()
        dt['mutated-pattern'] = "unseen-gpt4o-1-token"
        dt['mutated'] = True
        new_dataset.append(dt)
    Dataset.from_list(new_dataset).push_to_hub(hf_dst_repo_id, config_name=hf_dst_config, split=hf_dst_split, token=hf_token)


@subcommand(category=Category.EXPERIMENT)
def dataset_gen_unseen_caucasian_albanian_gpt4o(
    hf_src_repo_id = "EngineeringSoftware/PLSemanticsBench",
    hf_src_config = "predstate-IMP-SOS-uk-human-written",
    hf_dst_repo_id = "LambdaadbmaL/PredStateTest",
    hf_dst_config = "predstate",
    hf_dst_split = "imp_sos_mk_unseen_caucasian_albanian_human_written",
    hf_token = os.environ.get("HF_TOKEN")
):
    import llm_interpreter.data.dataset_process as dsp
    imp = IMP("IMPTEST", Language.SEMANTICS_TYPE.SOS, **dsp.DatasetProcessor.SEMANTICS_MUTATIONS["unseen"])
    dataset = load_dataset(hf_src_repo_id, name=hf_src_config, token=hf_token)['train']
    new_dataset = []
    for dt in dataset:
        dt['mutated-program'] = imp.translate_unmutated_program(dt['program'])
        if dt['semantics-type'] == "SOS":
            dt['syntax'] = imp.get_syntax()
        dt['semantics'] = imp.get_semantics()
        dt['mutated-pattern'] = "unseen"
        dt['mutated'] = True
        new_dataset.append(dt)
    Dataset.from_list(new_dataset).push_to_hub(hf_dst_repo_id, config_name=hf_dst_config, split=hf_dst_split, token=hf_token)

@subcommand(category=Category.EXPERIMENT)
def laod_from_hf_push_to_hf(
    hf_src_repo_id = "EngineeringSoftware/PLSemanticsBench",
    hf_src_config = "predstate-IMP-nk-human-written",
    hf_dst_repo_id = "LambdaadbmaL/PLSemanticsBench",
    hf_dst_config = "predstate",
    hf_dst_split = "nk_human_written",
    hf_token = os.environ.get("HF_TOKEN")
):
    dataset = load_dataset(hf_src_repo_id, name=hf_src_config, token=hf_token)['train']
    dataset.push_to_hub(hf_dst_repo_id, config_name=hf_dst_config, split=hf_dst_split, token=hf_token)

@subcommand(category=Category.EXPERIMENT)
def dataset_token_counter(
    field_name: str = "mutated-program",
    group_by_field: str = "mutation-pattern",
    hf_repo_id = "EngineeringSoftware/PLSemanticsBench",
    hf_config = "predstate-IMP-K-mk-human-written",
    hf_split = None,
    hf_token = os.environ.get("HF_TOKEN")
):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
    )    
    dataset = load_dataset(hf_repo_id, name=hf_config, token=hf_token)['train']
    dataset_tokens = defaultdict(list)
    for dt in dataset:
        dataset_tokens[str(dt[group_by_field])].append(gpt_runner._num_tokens_from_string(dt[field_name]))
    for group, tokens in dataset_tokens.items():
        print(f"Group: {group}")
        print(f"Average tokens: {np.mean(tokens)}")
        print(f"Max tokens: {np.max(tokens)}")
        print(f"Min tokens: {np.min(tokens)}")
        print(f"Median tokens: {np.median(tokens)}")
        print(f"Std tokens: {np.std(tokens)}")


@subcommand(category=Category.EXPERIMENT)
def dataset_assembled_token_counter(
    mutation_pattern: str = None,
    hf_repo_id = "EngineeringSoftware/PLSemanticsBench",
    hf_config = "predstate-IMP-K-uk-human-written",
    hf_token = os.environ.get("HF_TOKEN")
):
    model_name = "gpt-4o-mini"
    exp_args = ExperimentArgs(
        task="op" if hf_config.startswith("predstate") else "srp" if hf_config.startswith("predrule") else "etp" if hf_config.startswith("predtrace") else None,
        setup_name="mk" if "-mk-" in hf_config else "uk" if "-uk-" in hf_config else "nk",
        expr_name="IMP-K" if "-IMP-K-" in hf_config else "IMP-SOS" if "-IMP-SOS-" in hf_config else None,
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.GPT_4o_MINI,
        args=exp_args,
        use_azure = False,
        use_hf = True,
        hf_train_split = True,
        hf_repo_id = hf_repo_id,
        hf_config = hf_config,
        hf_token = hf_token,
    )    
    _, _, _, input_tokens, output_tokens = gpt_runner.num_dataset_input_and_output_tokens(mutation_pattern=mutation_pattern)
    # print the average, max, min, median, std of input and output tokens
    return {
        "average_input_tokens": round(np.mean(input_tokens)),
        "max_input_tokens": round(np.max(input_tokens)),
        "min_input_tokens": round(np.min(input_tokens)),
        "median_input_tokens": round(np.median(input_tokens)),
        "std_input_tokens": round(np.std(input_tokens)),
        "average_output_tokens": round(np.mean(output_tokens)),
        "max_output_tokens": round(np.max(output_tokens)),
        "min_output_tokens": round(np.min(output_tokens)),
        "median_output_tokens": round(np.median(output_tokens)),
        "std_output_tokens": round(np.std(output_tokens)),
    }


@subcommand(category=Category.EXPERIMENT)
def dataset_assembled_token_counter_all(
    hf_repo_id = "EngineeringSoftware/PLSemanticsBench",
    hf_token = os.environ.get("HF_TOKEN")
):
    tasks = ["predstate", "predrule", "predtrace"]
    setups = ["uk", "mk"]
    exprs = ["IMP-SOS", "IMP-K"]
    dataset_names = ["human-written", "llm-translated", "fuzzer-generated"]
    mutation_patterns = ["Standard", "KeywordSwap", "KeywordObf"]
    results = []
    for task in tasks:
        for mutation_pattern in mutation_patterns:
            for setup in setups:
                if (setup == "mk" and mutation_pattern == 'Standard') or (setup == "nk" and mutation_pattern != 'Standard') or (setup == "uk" and mutation_pattern != 'Standard'):
                    continue
                for expr in exprs:
                    if task == "predstate":
                        for dataset_name in dataset_names:
                            print(f"Task: {task}, Setup: {setup}, Expr: {expr}, Dataset: {dataset_name}, Mutation Pattern: {mutation_pattern}")
                            hf_config = f"{task}-{expr}-{setup}-{dataset_name}"
                            results.append(dataset_assembled_token_counter(mutation_pattern=mutation_pattern, hf_repo_id=hf_repo_id, hf_config=hf_config, hf_token=hf_token) | {
                                "task": task,
                                "mutation_pattern": mutation_pattern,
                                "setup": setup,
                                "expr": expr,
                                "dataset_name": dataset_name,
                            })
                    else:
                        print(f"Task: {task}, Setup: {setup}, Expr: {expr}, Dataset: human-written, Mutation Pattern: {mutation_pattern}")
                        hf_config = f"{task}-{expr}-{setup}-human-written"
                        results.append(dataset_assembled_token_counter(mutation_pattern=mutation_pattern, hf_repo_id=hf_repo_id, hf_config=hf_config, hf_token=hf_token) | {
                            "task": task,
                            "mutation_pattern": mutation_pattern,
                            "setup": setup,
                            "expr": expr,
                            "dataset_name": "human-written",
                        })
    su.io.dump(Macros.results_dir / "dataset_assembled_token_counter_all.json", results)

@subcommand(category=Category.EXPERIMENT)
def cost_estimation(
    model_name: str,
    task: str,
    setup: str,
    expr: str,
    strategy: str = "da",
    dataset_name: str = "human_written",
    use_hf = True,
    hf_repo_id = "LambdaadbmaL/PLSemanticsBench",
    hf_config = "nl2rule",
    hf_split = "Standard_NumRule5_RandomSampleFalse"

):
    models: list = ["o3-mini", "gpt-5-mini", "gpt-4o-mini", "gemini-2.5-pro"]
    gpt_model_dict: dict = {
        "o3-mini": GPT_MODEL_ENUM.O3_MINI,
        "gpt-5-mini": GPT_MODEL_ENUM.GPT_5_MINI,
        "gpt-4o-mini": GPT_MODEL_ENUM.GPT_4o_MINI,
    }
    tasks: list = ["pcp", "op", "srp", "etp", "nl2rule", "rule2nl"]
    setups: list = ["uk", "mk"]
    exprs: list = ["IMP-SOS", "IMP-K"]
    strategies: list = ["da", "cot"]
    if model_name not in models:
        raise NotImplementedError(f"Model name must be one of {models}")
    # fi
    if task not in tasks:
        raise NotImplementedError(f"Task must be one of {tasks}")
    # fi
    if setup not in setups:
        raise NotImplementedError(f"Setup must be one of {setups}")
    # fi
    if expr not in exprs:
        raise NotImplementedError(f"Expr must be one of {exprs}")
    # fi
    if strategy not in strategies and strategy:
        raise NotImplementedError(f"Strategy must be one of {strategies}")
    # fi
    exp_args = ExperimentArgs(
        task=task,
        setup_name=setup,
        expr_name=expr,
        dataset_name=dataset_name,
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA if strategy == "da" else PROMPT_STRATEGY.COT,
        random_seed=1,
    )
    runner = None
    match model_name:
        case "gemini-2.5-pro":
            runner = GeminiRunner(
                model_config_file=Macros.model_config_dir
                / "etp"
                / exp_args.setup_name
                / exp_args.expr_name.lower()
                / f"config-{model_name}.yaml",
                args=exp_args,
                max_workers=4,
                use_hf = use_hf,
                hf_repo_id = hf_repo_id,
                hf_config = hf_config,
                hf_split = hf_split,
            )
        case _:
            runner = GPTRunner(
                model_config_file=Macros.model_config_dir
                / exp_args.task
                / exp_args.setup_name
                / exp_args.expr_name.lower()
                / f"config-{model_name}.yaml",
                gpt_model=gpt_model_dict[model_name],
                args=exp_args,
                use_hf = use_hf,
                hf_repo_id = hf_repo_id,
                hf_config = hf_config,
                hf_split = hf_split,
            )
    print(runner.get_cost_estimation_for_task())


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_etp_imp_sos(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_mk_etp_imp_k(seed: int = 1):
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="etp",
        setup_name="mk",
        expr_name="IMP-K",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
        random_seed=seed,
    )
    gemini_runner = GeminiRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        args=exp_args,
        max_workers=8,
    )
    gemini_runner.do_experiment()


# ---
# op tasks
# ---

if platform.system() == "Linux":
    # qwen-coder-3b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_nk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
        # da
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_uk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_uk_op_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_mk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    def qwen_coder_3b_mk_op_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen-coder-7b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_nk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_uk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_mk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen-coder-14b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_nk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        # da
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_op_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_op_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen-coder-32b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_nk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        # da
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_nk_op_imp_sos_temp(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        # da
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h1002.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()


    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_op_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_op_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen3-coder-30b-a3b
    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_nk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        # da
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_op_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_op_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwq-32b
    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_nk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        # da
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_nk_op_imp_sos_synthetic_cpp(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        # da
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            dataset_name="synthetic_cpp",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_nk_op_imp_sos_fuzzer_generated(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        # da
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            dataset_name="fuzzer_generated",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_op_imp_sos_synthetic_cpp(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            dataset_name="synthetic_cpp",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_op_imp_sos_fuzzer_generated(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            dataset_name="fuzzer_generated",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_op_imp_k(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_op_imp_k_synthetic_cpp(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            dataset_name="synthetic_cpp",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_op_imp_k_fuzzer_generated(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            dataset_name="fuzzer_generated",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_op_imp_sos(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_op_imp_sos_synthetic_cpp(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            dataset_name="synthetic_cpp",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_op_imp_sos_fuzzer_generated(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            dataset_name="fuzzer_generated",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_op_imp_k(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_op_imp_k_synthetic_cpp(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            dataset_name="synthetic_cpp",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_op_imp_k_fuzzer_generated(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            dataset_name="fuzzer_generated",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # deepseek-qwen-14b
    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_nk_op_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_op_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_op_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_op_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_op_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # deepseek-qwen-32b
    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_nk_op_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_op_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_op_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_op_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_op_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # llama3.3-70b
    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_nk_op_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        # da
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_op_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_op_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_op_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_op_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_nk_op_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="nk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_op_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_op_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_op_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_op_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="op",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    # ---
    # srp tasks
    # ---

    # qwen-coder-3b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_uk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_mk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen-coder-7b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_uk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_mk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen-coder-14b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_srp_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_srp_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen-coder-32b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_srp_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_srp_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen3-coder-30b-a3b
    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment(shard=shard, total_shards=total_shards)
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        #vllm_runner.do_experiment(shard=shard, total_shards=total_shards)

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_srp_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment(shard=shard, total_shards=total_shards)
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        #vllm_runner.do_experiment(shard=shard, total_shards=total_shards)

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_srp_imp_k(seed: int = 42):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment(shard=shard, total_shards=total_shards)
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        #vllm_runner.do_experiment()

    # qwq-32b
    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_uk_srp_imp_k(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_srp_imp_sos(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwq_32b_mk_srp_imp_k(seed: int = 42):
        model_name = "Qwen/QwQ-32B"
        ## uk
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # deepseek-qwen-14b
    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_srp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_srp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_srp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_srp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # deepseek-qwen-32b
    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_srp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_srp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_srp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_srp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    # llama3.3-70b
    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_srp_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_srp_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_srp_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_srp_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_srp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_srp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_srp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_srp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="srp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    ##
    # etp tasks
    ##

    # qwen-coder-3b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_uk_etp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_3b_mk_etp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-3b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen-coder-7b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_uk_etp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_7b_mk_etp_imp_sos(seed: int = 42):
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-7b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        vllm_runner.args = exp_args
        vllm_runner.do_experiment()

    # qwen-coder-14b
    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_etp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_uk_etp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_etp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_14b_mk_etp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-k/config-qwen25-coder-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_etp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_uk_etp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_etp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_coder_32b_mk_etp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_etp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_uk_etp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_etp_cot_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen3_coder_30b_a3b_mk_etp_cot_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            prompt_strategy=PROMPT_STRATEGY.COT,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-k/config-qwen25-coder-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_qwq_32b_uk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_qwq_32b_uk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_qwq_32b_mk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def qwen_qwq_32b_mk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="Qwen/QwQ-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-k/config-qwq-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_uk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-k/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_14b_mk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-k/config-deepseek-qwen-14b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_uk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/uk/imp-k/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_etp_imp_sos(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-sos/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_qwen_32b_mk_etp_imp_k(seed: int = 42):
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / "etp/mk/imp-k/config-deepseek-qwen-32b-h100.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_etp_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_uk_etp_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_etp_imp_sos(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def llama3_70b_mk_etp_imp_k(seed: int = 42):
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()
        # cot
        exp_args.prompt_strategy = PROMPT_STRATEGY.COT
        gpt_runner.args = exp_args
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_etp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_uk_etp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="uk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_etp_imp_sos(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    @subcommand(category=Category.EXPERIMENT)
    def deepseek_llama3_70b_mk_etp_imp_k(seed: int = 42):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        exp_args = ExperimentArgs(
            task="etp",
            setup_name="mk",
            expr_name="IMP-K",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
            random_seed=seed,
        )
        gpt_runner = GPTRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-k/config-deepseek-llama-33-70b-4h100.yaml",
            gpt_model=GPT_MODEL_ENUM.DeepSeek_LLAMA_3_70B,
            args=exp_args,
        )
        gpt_runner.do_experiment()

    ## -------------------
    ## TACC EXPERIMENTS
    ## -------------------
    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_pcp_uk_imp_sos_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-pcp-uk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_pcp_uk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-pcp-uk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_pcp_mk_imp_sos_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-pcp-mk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_pcp_mk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-pcp-mk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_op_uk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-op-uk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_op_qwq_synthetic_cpp_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir
            / "tacc_runners/task-op-qwq-synthetic-cpp.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_op_qwq_fuzzer_generated_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir
            / "tacc_runners/task-op-qwq-fuzzer-generated.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_op_qwq_fuzzer_generated_reduced_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir
            / "tacc_runners/task-op-qwq-fuzzer-generated-reduced.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_op_uk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-op-uk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_op_nk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-op-nk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_llama_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-llama.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_op_mk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-op-mk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_op_mk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-op-mk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_srp_uk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-srp-uk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_srp_uk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-srp-uk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_srp_mk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-srp-mk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_srp_mk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-srp-mk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_etp_uk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-etp-uk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_etp_uk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-etp-uk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_etp_mk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-etp-mk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_etp_mk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-etp-mk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_rebuttal_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/rebuttal.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_nl2rule_uk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-nl2rule-uk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_nl2rule_mk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-nl2rule-mk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_rule2nl_uk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-rule2nl-uk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_rule2nl_mk_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-rule2nl-mk-IMP-SOS.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_nl2rule_uk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-nl2rule-uk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_nl2rule_mk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-nl2rule-mk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_rule2nl_uk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-rule2nl-uk-IMP-K.yaml",
        )
        tacc_runner.run()

    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_rule2nl_mk_imp_k_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-rule2nl-mk-IMP-K.yaml",
        )
        tacc_runner.run()
    
    @subcommand(category=Category.EXPERIMENT)
    def launch_tacc_nl2ule_rule2nl_qwen3_coder_30b_a3b_exps():
        tacc_runner = TACCCoordinator(
            tasks_spec_file=Macros.main_dir / "tacc_runners/task-nl2rule-rule2nl-qwen3-coder-30b-a3b.yaml",
        )
        tacc_runner.run()
# fi

# AST prompt
@subcommand(category=Category.EXPERIMENT)
def openai_o3_mk_ast_imp_sos():
    model_name = "o3-mini"
    exp_args = ExperimentArgs(
        task="ast",
        setup_name="mk",
        expr_name="IMP-SOS",
        model_name=model_name,
        prompt_strategy=PROMPT_STRATEGY.DA,
    )
    gpt_runner = GPTRunner(
        model_config_file=Macros.model_config_dir
        / exp_args.task
        / exp_args.setup_name
        / exp_args.expr_name.lower()
        / f"config-{model_name}.yaml",
        gpt_model=GPT_MODEL_ENUM.O3_MINI,
        args=exp_args,
    )
    gpt_runner.do_experiment()


@subcommand(category=Category.EXPERIMENT)
def gemini_ast_mk_imp_sos():
    model_name = "gemini-2.5-pro"
    exp_args = ExperimentArgs(
        task="ast",
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


@subcommand(category=Category.EXPERIMENT)
def do_experiment_n_times(n: int, exp_name: str):
    if exp_name in globals():
        for i in range(1, n + 1):
            globals()[exp_name](i)
        # rof
    # fi


# fed
