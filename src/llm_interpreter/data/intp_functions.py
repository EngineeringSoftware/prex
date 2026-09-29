import platform
import seutil as su
from typing import List
from llm_interpreter.utils import (
    subcommand,
    Category,
)

from llm_interpreter.macros import Macros
from llm_interpreter.experiments.prompts import PROMPT_STRATEGY
from llm_interpreter.experiments.args import ExperimentArgs
from llm_interpreter.data.repo_collect import ReposCollector
from llm_interpreter.data.data_collect import DataCollector
from llm_interpreter.data.dataset_process import DatasetProcessor
from llm_interpreter.language import Language
from llm_interpreter.data.synthetic_dataset_process import SyntheticDataProcess
if platform.system() == "Linux":
    from llm_interpreter.experiments.runners import (
        VLLMRunner,
    )
#fi

logger = su.log.get_logger(__name__, su.log.INFO)


@subcommand(category=Category.DATA)
def collect_cpp_competition_repos(source: List[str]):
    """
    Download the github repo with the CPP solutions to the competition.
    --source [cpp_codeforces,cpp_atcoder,cpp_leetcode]
    """
    collector = ReposCollector()
    collector.download_repo(
        [Macros.doc_dir / f"{src}_links.txt" for src in source],
        Macros.downloads_dir / "cpp_competition_repos",
    )


@subcommand(category=Category.DATA)
def collect_cpp_code_from_github():
    """
    Collect the C++ code from the downloaded github repos.
    """
    collector = DataCollector()
    collector.collect_cpp_code()


@subcommand(category=Category.DATA)
def collect_cpp_code_from_codeforces():
    """
    Collect the C++ code from the codeforces dataset.
    """
    collector = DataCollector()
    collector.collect_submissions_with_tests()

def collect_cpp_code_token_distribution():
    """
    Collect the token distribution of the C++ code collected from the github repos.
    """
    collector = DataCollector()
    collector.get_cpp_data_token_distribution()


if platform.system() == "Linux":
    @subcommand(category=Category.DATA)
    def translate_cpp_code_to_imp():
        """
        Build synthetic IMP data from the C++ code collected from the huggingface open-r1.
        This will use the translation prompt to convert C++ code to IMP.
        """
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        exp_args = ExperimentArgs(
            task="data-gen",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"{exp_args.task}/{exp_args.setup_name}/imp-sos/config-qwen25-coder-32b.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()


    @subcommand(category=Category.DATA)
    def translate_csmith_code_to_imp():
        """
        Build synthetic IMP data from the C++ code collected from the huggingface open-r1.
        This will use the translation prompt to convert C++ code to IMP.
        """
        model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        exp_args = ExperimentArgs(
            task="dgc",
            setup_name="uk",
            expr_name="IMP-SOS",
            model_name=model_name,
            prompt_strategy=PROMPT_STRATEGY.DA,
        )
        vllm_runner = VLLMRunner(
            model_config_file=Macros.model_config_dir
            / f"data-gen/{exp_args.setup_name}/imp-sos/config-qwen25-coder-14b-tokyo.yaml",
            args=exp_args,
        )
        vllm_runner.do_experiment()
#fi

@subcommand(category=Category.DATA)
def filter_executable_imp_programs():
    """
    Data cleaning: filter the executable IMP programs from the synthetic dataset.
    """
    synthetic_data_processor = SyntheticDataProcess("IMP")
    synthetic_data_processor.collect_imp_programs()


@subcommand(category=Category.DATA)
def transform_old_imp_programs():
    synthetic_data_processor = SyntheticDataProcess("IMP")
    synthetic_data_processor.transform_imp_programs()


@subcommand(category=Category.DATA)
def collect_csmith_programs():
    synthetic_data_processor = SyntheticDataProcess("IMP")
    synthetic_data_processor.collect_csmith_programs()


@subcommand(category=Category.DATA)
def collect_translated_csmith_programs():
    synthetic_data_processor = SyntheticDataProcess("IMP")
    synthetic_data_processor.collect_translated_csmith_programs()

    
@subcommand(category=Category.DATA)
def make_ig_uk_imp_sos_dataset():
    syntax_types = ["EBNF", "ANTLR"]
    impl_languages = ["Python", "Java", "Go", "Rust"]

    data_processor = DatasetProcessor(PL="IMP")
    for syntax_type in syntax_types:
        data_processor.create_ig_dataset(
            PL="IMP",
            task="ig",
            setup_name="uk",
            syntax_type=syntax_type,
            semantics_type=Language.SEMANTICS_TYPE.SOS,
            impl_langs=impl_languages,
            target_dir=Macros.data_dir / "dataset",
        )


@subcommand(category=Category.DATA)
def make_ig_mk_imp_sos_dataset():
    syntax_types = ["EBNF", "ANTLR"]
    impl_languages = ["Python", "Java", "Go", "Rust"]

    data_processor = DatasetProcessor(PL="IMP")
    for syntax_type in syntax_types:
        data_processor.create_ig_dataset(
            PL="IMP",
            task="ig",
            setup_name="mk",
            syntax_type=syntax_type,
            semantics_type=Language.SEMANTICS_TYPE.SOS,
            impl_langs=impl_languages,
            target_dir=Macros.data_dir / "dataset",
        )


@subcommand(category=Category.DATA)
def make_iga_uk_imp_sos_dataset():
    impl_languages = ["Python", "Java"]

    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_iga_dataset(
        PL="IMP",
        task="iga",
        setup_name="uk",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        impl_langs=impl_languages,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_iga_mk_imp_sos_dataset():
    impl_languages = ["Python", "Java"]

    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_iga_dataset(
        PL="IMP",
        task="iga",
        setup_name="mk",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        impl_langs=impl_languages,
        target_dir=Macros.data_dir / "dataset",
    )
