from __future__ import annotations
from typing import Union, List, Any, Dict, Tuple, Sequence, Optional
from pathlib import Path
import seutil as su
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import traceback
import os
import random
import re
from dataclasses import dataclass



from llm_interpreter.macros import Macros
from llm_interpreter.language import Language, IMP
from llm_interpreter.language.visitors import IMPFuzzerVisitor
from llm_interpreter.utils import (
    write_to_tmp,
    read_from_txt_file,
    generate_xml_trace,
    push_jsonl_split_to_hf,
)
from llm_interpreter.compiler_runners import KFramework, KResult
from llm_interpreter.compiler_runners.k_framework import ExecutionState
from llm_interpreter.language.language import RuleMeta

SEED_VALUE = 42
random.seed(SEED_VALUE)

PCP_SEMANTIC_ERROR_SUFFIXES = (
    "divide_by_zero",
    "modulo_zero",
    "break_outside_loop",
    "continue_outside_loop",
    "var_use_before_declare",
)

# Excluded from PCP datasets for now because full prompt (syntax + semantics + program)
# exceeds the 32K token context window used by Qwen models (~38K for fuzz_321.imp)
PCP_EXCLUDED_PROGRAMS = frozenset({"fuzz_321.imp"})

logger = su.log.get_logger(__name__, su.log.INFO)


def list_program_files(raw_programs_dir: Union[str, Path], pl: str) -> List[Path]:
    """List program files in a flat or nested corpus directory."""
    root = Path(raw_programs_dir)
    ext = pl.lower()
    direct = sorted(root.glob(f"*.{ext}"))
    if direct:
        return direct
    return sorted(root.rglob(f"*.{ext}"))


def infer_pcp_program_source(src_filename: str) -> str:
    """
    Infer corpus source from a PCP program filename.
    """
    stem = Path(src_filename).stem
    for suffix in PCP_SEMANTIC_ERROR_SUFFIXES:
        token = f"_{suffix}"
        if stem.endswith(token):
            stem = stem[: -len(token)]
            break
    if stem.startswith("fuzz_"):
        return "fuzzer_generated"
    if stem.startswith("pgm_"):
        return "synthetic_cpp"
    return "human_written"


@dataclass
class SRPData:
    line_number: int = 0
    prior_state: Dict[str, Any] = None
    rules: List[str] = None
    cleaned_stmt: str = None
    control_stack: List[str] = None


@dataclass
class ASTData:
    line_number: int = 0
    statement: str = None
    mutated_statement: str = None


class DatasetProcessor:
    SEMANTICS_MUTATIONS = {
        "addSub_mulDiv_negateRelation": {
            "PLUS_OP": "-",
            "MINUS_OP": "+",
            "MUL_OP": "/",
            "DIV_OP": "*",
            "LT_OP": ">",
            "GT_OP": "<",
            "LTEQ_OP": ">=",
            "GTEQ_OP": "<=",
            "EQ_OP": "!=",
            "NEQ_OP": "==",
            "AND_OP": "||",
            "OR_OP": "&&",
        },
        "unseen": {
            "PLUS_OP": "𐕐",
            "MINUS_OP": "𐕙",
            "MUL_OP": "𐕊",
            "DIV_OP": "𐕏",
            "MOD_OP": "𐕖",
            "ASSIGN_OP": "𐕂",
            "LT_OP": "𐔳",
            "GT_OP": "𐕃",
            "LTEQ_OP": "𐔷",
            "GTEQ_OP": "𐕛",
            "EQ_OP": "𐕟",
            "NEQ_OP": "𐕀",
            "AND_OP": "𐕜",
            "OR_OP": "𐔻",
            "NOT_OP": "𐔰",
            "BREAK": "𐔾",
            "IF": "𐔸",
            "ELSE": "𐕎",
            "WHILE": "𐕕",
            "HALT": "𐔱",
            "CONTINUE": "𐔲",
            "LOOP": "𐕣",
            "LE": "𐕠",
            "ERROR": "𐕞",
        },
        "unseen-gpt4o-1-token": {
            "PLUS_OP": "★",
            "MINUS_OP": "▲",
            "MUL_OP": "$",
            "DIV_OP": "■",
            "MOD_OP": "▼",
            "ASSIGN_OP": "€",
            "LT_OP": "●",
            "GT_OP": "◎",
            "LTEQ_OP": "▽",
            "GTEQ_OP": "❤",
            "EQ_OP": "◆",
            "NEQ_OP": "►",
            "AND_OP": "◇",
            "OR_OP": "△",
            "NOT_OP": "○",
            "BREAK": "✓",
            "IF": "▷",
            "ELSE": "¥",
            "WHILE": "₪",
            "HALT": "₹",
            "CONTINUE": "✔",
            "LOOP": "£",
            "LE": "▶",
            "ERROR": "□",
        },
    }

    OP_UNIT_MUTATIONS = {
        "addSub": {
            "PLUS_OP": "-",
            "MINUS_OP": "+",
        },
        "mulDiv": {
            "MUL_OP": "/",
            "DIV_OP": "*",
        },
        "negateRelation": {
            "LT_OP": ">",
            "GT_OP": "<",
            "LTEQ_OP": ">=",
            "GTEQ_OP": "<=",
            "EQ_OP": "!=",
            "NEQ_OP": "==",
            "AND_OP": "||",
            "OR_OP": "&&",
        },
    }

    SEMANTIC_INVALID_RULES_IMP = {
        "SOS": {
            f"{IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO.value}": "Rule 19",
            f"{IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO.value}": "Rule 23",
            f"{IMPFuzzerVisitor.FUZZING_TYPES.BREAK_OUTSIDE_LOOP.value}": "Rule 73",
            f"{IMPFuzzerVisitor.FUZZING_TYPES.CONTINUE_OUTSIDE_LOOP.value}": "Rule 76",
            f"{IMPFuzzerVisitor.FUZZING_TYPES.VAR_USE_BEFORE_DECLARE.value}": "Rule 2", # We don't include Rule 6 for assignment since we don't generate such invalid variants 
        },
        "K": {
            f"{IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO.value}": "Rule 7",
            f"{IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO.value}": "Rule 9",
            f"{IMPFuzzerVisitor.FUZZING_TYPES.BREAK_OUTSIDE_LOOP.value}": "Rule 34",
            f"{IMPFuzzerVisitor.FUZZING_TYPES.CONTINUE_OUTSIDE_LOOP.value}": "Rule 31",
            f"{IMPFuzzerVisitor.FUZZING_TYPES.VAR_USE_BEFORE_DECLARE.value}": "Rule 2",
        },
    }

    KEYWORD_OBF_LEGACY = "unseen"
    KEYWORD_OBF_GPT4O_1_TOKEN = "unseen-gpt4o-1-token"

    @staticmethod
    def pcp_keyword_obf_uses_gpt4o_1_token(model_name: str) -> bool:
        """
        All models use single-token KeywordObf symbols at prompt time.
        """
        return any(
            tag in model_name
            for tag in ("Coder-3B", "Coder-7B", "Coder-14B", "Coder-32B")
        ) or "DeepSeek-R1-Distill-Qwen" in model_name

    @classmethod
    def keyword_obf_mutations(cls, model_name: str) -> Dict[str, str]:
        key = (
            cls.KEYWORD_OBF_GPT4O_1_TOKEN
            if cls.pcp_keyword_obf_uses_gpt4o_1_token(model_name)
            else cls.KEYWORD_OBF_LEGACY
        )
        return cls.SEMANTICS_MUTATIONS[key]

    @classmethod
    def remap_mutation_symbols(cls, text: str, from_key: str, to_key: str) -> str:
        if from_key == to_key:
            return text
        src = cls.SEMANTICS_MUTATIONS[from_key]
        dst = cls.SEMANTICS_MUTATIONS[to_key]
        pairs = sorted(
            ((src[k], dst[k]) for k in src if k in dst and src[k] != dst[k]),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for old, new in pairs:
            text = text.replace(old, new)
        return text

    def __init__(self, PL: str):
        self.super_pl = PL
        self.k_framework = None

    def create_dataset_unmutated_semantics(
        self,
        PL: str,
        semantics_type: Language.SEMANTICS_TYPE,
        task: str,
        setup_name: str,
        raw_programs_dir: Union[str, Path],
        target_dir: Union[str, Path],
        dataset_name: str = "human_written",
        use_k_framework: bool = True,
    ):
        """
        Create the dataset of programs for the given UNMUTATED programming language and the semantics type.
        """
        self.task = task
        self.setup_name = setup_name
        if task == "pcp" and use_k_framework:
            logger.info("Skipping K framework setup for pcp task.")
            use_k_framework = False
        if self.super_pl == "IMP":
            if use_k_framework:
                execution_k = IMP(
                    name=f"{self.super_pl}_{semantics_type.value}",
                    semantics_type=Language.SEMANTICS_TYPE.K,
                )
                self._setup_k_framework(program_lang=execution_k)
            else:
                execution_k = None
            program_language = IMP(
                name=f"{self.super_pl}_{semantics_type.value}",
                semantics_type=semantics_type,
            )
        else:
            raise NotImplementedError(f"Language {self.super_pl} is not supported yet.")
        data_list = self.create_dataset(
            raw_programs_dir, PL, semantics_type, execution_k, program_language, [], []
        )
        logger.info(f"In total {len(data_list)} unmutated data.")
        # save data
        exp_name = f"{PL}-{semantics_type}"
        su.io.dump(
            Path(target_dir)
            / (
                f"dataset-{task}-{setup_name}-{exp_name}.jsonl"
                if dataset_name == "human_written"
                else f"dataset-{task}-{setup_name}-{exp_name}-{dataset_name}.jsonl"
            ),
            data_list,
        )

    def create_dataset_mutated_semantics(
        self,
        PL: str,
        semantics_type: Language.SEMANTICS_TYPE,
        task: str,
        setup_name: str,
        raw_programs_dir: Union[str, Path],
        target_dir: Union[str, Path],
        include_caucasian_albanian: bool = True,
        dataset_name: str = "human_written",
        use_k_framework: bool = True,
    ):
        """
        Create the dataset of programs for given MUTATED programming language and the semantics type.
        """
        self.task = task
        self.setup_name = setup_name
        if task == "pcp" and use_k_framework:
            logger.info("Skipping K framework setup for pcp task.")
            use_k_framework = False
        mutated_data_list = []
        mutation_patterns = self._get_mutation_patterns(include_caucasian_albanian)
        for patterns in mutation_patterns:
            # prepare semantics
            semantics_mutations = {}
            for mutate_pattern in patterns:
                semantics_mutations.update(self.SEMANTICS_MUTATIONS[mutate_pattern])
            #
            execution_k, program_language = self._get_mutated_language(
                mutate_patterns=patterns,
                semantics_type=semantics_type,
                semantics_mutations=semantics_mutations,
                use_k_framework=use_k_framework,
            )
            mutated_data_list = self.create_dataset(
                raw_programs_dir,
                PL,
                semantics_type,
                execution_k,
                program_language,
                patterns,
                mutated_data_list,
            )
        logger.info(
            f"In total {len(mutated_data_list)} mutated data for {len(mutation_patterns)} mutation patterns."
        )
        exp_name = f"{PL}-{semantics_type}"
        # save data
        su.io.dump(
            Path(target_dir)
            / (
                f"dataset-{task}-{setup_name}-{exp_name}.jsonl"
                if dataset_name == "human_written"
                else f"dataset-{task}-{setup_name}-{exp_name}-{dataset_name}.jsonl"
            ),
            mutated_data_list,
        )

    def create_dataset(
        self,
        raw_programs_dir: Union[str, Path],
        PL: str,
        semantics_type: Language.SEMANTICS_TYPE,
        execution_k: Language,
        program_language: Language,
        patterns: List[str] = [],
        data_list: List = [],
    ) -> List:
        """
        Create the dataset for the given programming language and semantics type.
        """
        if PL == "IMP":
            original_program_language = IMP(
                name=f"{self.super_pl}_{semantics_type.value}",
                semantics_type=semantics_type,
            )
        else:
            raise NotImplementedError(f"Language {PL} is not supported yet.")
        program_language.build_visitors()
        all_program_files = list_program_files(raw_programs_dir, PL)
        with tqdm(
            total=len(all_program_files),
            desc="Processing dataset",
        ) as pbar:
            for file_path in all_program_files:
                file_name = file_path.name
                if self.task == "pcp" and file_name in PCP_EXCLUDED_PROGRAMS:
                    logger.info(
                        f"Skipping {file_name}: excluded for now because the PCP "
                        f"prompt exceeds the model token limit."
                    )
                    pbar.update(1)
                    continue
                raw_program = su.io.load(file_path, fmt=su.io.Fmt.txt)
                dt = {}
                mutation_pattern = "_".join(patterns)
                if mutation_pattern:
                    data_id = f"{self.task}-{self.setup_name}-{semantics_type.value}-{mutation_pattern}-{file_name}"
                else:
                    data_id = f"{self.task}-{self.setup_name}-{semantics_type.value}-{file_name}"
                #
                dt["id"] = data_id
                dt["program"] = self._transform_program(
                    raw_program=raw_program,
                    program_language=original_program_language,
                )
                dt["src-filename"] = str(file_name)
                dt["semantics-type"] = semantics_type.value
                try:
                    dt["syntax"] = program_language.get_syntax()
                except NotImplementedError as err:
                    print(err)
                    dt["syntax"] = ""
                # yrt
                dt["semantics"] = program_language.get_semantics()
                dt["language"] = "IMP"
                if patterns:
                    dt["mutated-program"] = self._transform_program(
                        raw_program=raw_program,
                        program_language=program_language,
                    )
                    tmp_name = (
                        f"mutated-{semantics_type.value}-{mutation_pattern}-{file_name}"
                    )
                    write_to_tmp(dt["mutated-program"], tmp_name)
                    file_path = Macros.tmp_dir / tmp_name
                else:
                    dt["mutated-program"] = dt["program"]
                #
                dt["mutated"] = dt["mutated-program"] != dt["program"]
                dt["mutation-pattern"] = (
                    "Standard"
                    if len(mutation_pattern) == 0
                    else (
                        "KeywordSwap"
                        if "addSub_mulDiv_negateRelation" in mutation_pattern
                        else "KeywordObf"
                    )
                )
                try:
                    if self.task == "pcp":
                        self._process_pep_dt(file_path, dt)
                        dt["dataset-source"] = infer_pcp_program_source(file_name)
                    elif self.task == "op":
                        self._process_op_dt(
                            file_path, dt, execution_k, semantics_type, False
                        )
                        del dt["exec-trace"]
                        del dt["final-state"]
                    elif self.task == "srp":
                        self._process_srp_dt(
                            file_path, dt, execution_k, semantics_type, True
                        )
                        del dt["exec-trace"]
                        del dt["final-state"]
                    elif self.task == "etp":
                        self._process_etp_dt(
                            file_path,
                            dt,
                            execution_k,
                            semantics_type,
                            program_language,
                            False,
                        )
                        del dt["exec-trace"]
                        del dt["final-state"]
                    elif self.task == "ast":
                        self._process_ast_dt(file_path, dt, execution_k, semantics_type)
                    else:
                        raise NotImplementedError(
                            f"Task {self.task} is not supported yet."
                        )
                    # fi
                    data_list.append(dt)
                except Exception as err:
                    print(f"Omitting {file_name} due to error {err}")
                # yrt
                pbar.update(1)
            # for
        # with
        return data_list

    def create_ig_dataset(
        self,
        PL: str,
        task: str,
        setup_name: str,
        syntax_type: str,
        semantics_type: Language.SEMANTICS_TYPE,
        impl_langs: List[str],
        target_dir: Path,
    ):
        print(f"Creating the {task}_{setup_name}_{PL} dataset...")
        data_list = []

        if PL != "IMP":
            raise NotImplementedError(f"Language {PL} is not supported yet.")
        if syntax_type not in ["EBNF", "ANTLR"]:
            raise ValueError(f"Syntax type {syntax_type} is not supported.")
        if semantics_type != Language.SEMANTICS_TYPE.SOS:
            raise ValueError(f"Semantics type {semantics_type} is not supported.")

        if setup_name == "uk":
            program_language = IMP(
                name=f"{self.super_pl}_{semantics_type.value}",
                semantics_type=semantics_type,
            )
        elif setup_name == "mk":
            # program_language = IMP(
            #     name=f"{self.super_pl}_{semantics_type.value}_mk",
            #     semantics_type=semantics_type,
            # )
            raise NotImplementedError(f"Setup name {setup_name} is not supported yet.")
        else:
            raise ValueError(f"Setup name {setup_name} is not supported.")

        for impl_lang in impl_langs:
            dt = {}
            dt["language"] = "MyLang"
            dt["syntax-type"] = syntax_type
            dt["semantics-type"] = semantics_type.value
            dt["impl-language"] = impl_lang
            dt["syntax"] = (
                program_language.get_ebnf_syntax()
                if syntax_type == "EBNF"
                else program_language.get_antlr4_syntax()
            )
            dt["semantics"] = program_language.get_semantics()

            data_list.append(dt)

        su.io.dump(
            target_dir
            / f"dataset-{task}-{setup_name}-{PL}-{semantics_type.value}-{syntax_type}.jsonl",
            data_list,
        )

    def create_iga_dataset(
        self,
        PL: str,
        task: str,
        setup_name: str,
        semantics_type: Language.SEMANTICS_TYPE,
        impl_langs: List[str],
        target_dir: Path,
    ):
        print(f"Creating the {task}_{setup_name}_{PL} dataset...")
        data_list = []

        if PL != "IMP":
            raise NotImplementedError(f"Language {PL} is not supported yet.")
        if semantics_type != Language.SEMANTICS_TYPE.SOS:
            raise ValueError(f"Semantics type {semantics_type} is not supported.")

        if setup_name == "uk":
            pls = {
                "uk": IMP(
                    name=f"{self.super_pl}_{semantics_type.value}",
                    semantics_type=semantics_type,
                )
            }
        elif setup_name == "mk":
            pls = {
                "ks": IMP(
                    name=f"{self.super_pl}_{semantics_type.value}_ks",
                    semantics_type=semantics_type,
                    **self.SEMANTICS_MUTATIONS["addSub_mulDiv_negateRelation"],
                ),
                "ko": IMP(
                    name=f"{self.super_pl}_{semantics_type.value}_ko",
                    semantics_type=semantics_type,
                    **self.SEMANTICS_MUTATIONS["unseen"],
                ),
            }
        else:
            raise ValueError(f"Setup name {setup_name} is not supported.")

        for impl_lang in impl_langs:
            for mutation, program_language in pls.items():
                if impl_lang == "Python":
                    fext, main_file, antlr_dlang = "py", "main.py", "Python3"
                    ftemplate = "EvalVisitor-python.template"
                    if mutation == "uk":
                        antlr_dir = Macros.antlr_imp_python_uk_dir
                        valid_pgm_dir = Macros.valid_imp_uk_dir
                        invalid_pgm_dir = Macros.invalid_imp_uk_dir
                    elif mutation == "ks":
                        antlr_dir = Macros.antlr_imp_python_ks_dir
                        valid_pgm_dir = Macros.valid_imp_mk_ks_dir
                        invalid_pgm_dir = Macros.invalid_imp_mk_ks_dir
                    elif mutation == "ko":
                        antlr_dir = Macros.antlr_imp_python_ko_dir
                        valid_pgm_dir = Macros.valid_imp_mk_ko_dir
                        invalid_pgm_dir = Macros.invalid_imp_mk_ko_dir
                elif impl_lang == "Java":
                    fext, main_file, antlr_dlang = "java", "Main.java", "Java"
                    ftemplate = "EvalVisitor-java.template"
                    if mutation == "uk":
                        antlr_dir = Macros.antlr_imp_java_uk_dir
                        valid_pgm_dir = Macros.valid_imp_uk_dir
                        invalid_pgm_dir = Macros.invalid_imp_uk_dir
                    elif mutation == "ks":
                        antlr_dir = Macros.antlr_imp_java_ks_dir
                        valid_pgm_dir = Macros.valid_imp_mk_ks_dir
                        invalid_pgm_dir = Macros.invalid_imp_mk_ks_dir
                    elif mutation == "ko":
                        antlr_dir = Macros.antlr_imp_java_ko_dir
                        valid_pgm_dir = Macros.valid_imp_mk_ko_dir
                        invalid_pgm_dir = Macros.invalid_imp_mk_ko_dir
                else:
                    raise ValueError(
                        f"Implementation language {impl_lang} is not supported."
                    )

                # if antlr_dir does not exist, create it
                if not antlr_dir.exists():
                    antlr_dir.mkdir(parents=True)

                dt = {}
                syntax = program_language.get_antlr4_syntax()
                vis_template = su.io.load(
                    Macros.antlr_template_dir / ftemplate, fmt=su.io.Fmt.txt
                )
                main_template = su.io.load(
                    Macros.antlr_template_dir / main_file, fmt=su.io.Fmt.txt
                )

                su.io.dump(antlr_dir / "MyLang.g4", syntax, fmt=su.io.Fmt.txt)
                su.io.dump(
                    antlr_dir / "EvalVisitor.template", vis_template, fmt=su.io.Fmt.txt
                )
                su.io.dump(antlr_dir / main_file, main_template, fmt=su.io.Fmt.txt)

                # HACK: for some reason antlr4 will report error if the version
                #       is not specified
                su.bash.run(
                    f"antlr4 -v 4.13.0 -Dlanguage={antlr_dlang} -visitor {antlr_dir / 'MyLang.g4'}",
                    check_returncode=0,
                )

                dt["language"] = "MyLang"
                dt["impl-language"] = impl_lang
                dt["semantics-type"] = semantics_type.value
                if mutation != "uk":
                    dt["mutation-pattern"] = mutation
                dt["syntax"] = syntax
                dt["semantics"] = program_language.get_semantics()

                dt["states-field"] = "states"
                dt["base-visitor-code"] = su.io.load(
                    antlr_dir / f"MyLangVisitor.{fext}", fmt=su.io.Fmt.txt
                )
                dt["parser-code"] = su.io.load(
                    antlr_dir / f"MyLangParser.{fext}", fmt=su.io.Fmt.txt
                )
                dt["extend-visitor-name"] = "EvalVisitor"
                dt["extend-visitor-code"] = su.io.load(
                    antlr_dir / "EvalVisitor.template", fmt=su.io.Fmt.txt
                )
                dt["main-code"] = su.io.load(antlr_dir / main_file, fmt=su.io.Fmt.txt)

                for i, valid_example in enumerate(Macros.valid_examples):
                    ans_file = valid_example.replace(".imp", ".txt")
                    dt[f"valid-example-{i}"] = su.io.load(
                        valid_pgm_dir / valid_example, fmt=su.io.Fmt.txt
                    )
                    dt[f"valid-output-{i}"] = su.io.load(
                        Macros.valid_imp_ans_dir / ans_file, fmt=su.io.Fmt.txt
                    )

                for i, invalid_example in enumerate(Macros.invalid_examples):
                    for type, output in Macros.invalid_outputs.items():
                        if type in invalid_example:
                            dt[f"invalid-example-{i}"] = su.io.load(
                                invalid_pgm_dir / invalid_example, fmt=su.io.Fmt.txt
                            )
                            dt[f"invalid-output-{i}"] = output
                            break

                data_list.append(dt)

        su.io.dump(
            target_dir
            / f"dataset-{task}-{setup_name}-{PL}-{semantics_type.value}.jsonl",
            data_list,
        )

    def _get_mutation_patterns(
        self, include_caucasian_albanian: bool = False
    ) -> List[List[str]]:
        """
        Get all combinations of the mutation patterns.
        """
        all_patterns = list(self.SEMANTICS_MUTATIONS.keys())
        all_patterns.remove("unseen")
        # Keep generation on the legacy unseen alphabet ("unseen") and
        # exclude the alternate one-token variant from default combinations.
        if "unseen-gpt4o-1-token" in all_patterns:
            all_patterns.remove("unseen-gpt4o-1-token")
        all_patterns_comb = unique_subsets(all_patterns)
        if include_caucasian_albanian:
            all_patterns_comb.append(("unseen",))
        # fi
        return all_patterns_comb

    def _get_runtime_result(
        self,
        program_file: str,
        program_language: Language,
        semantics_type: Language.SEMANTICS_TYPE,
        gen_rules_for_srp: bool = False,
    ) -> KResult:
        """
        Run the program using K framework and return the result.
        """
        if self.k_framework is None:
            raise RuntimeError("K framework is not set up.")
        output: str = self.k_framework.run_program(program_file=program_file)
        program_content = read_from_txt_file(program_file)
        k_result = self.k_framework.parse_k_framework_output(
            program_content, output, program_language, semantics_type, gen_rules_for_srp
        )
        return k_result

    def _setup_k_framework(self, program_lang: Language) -> KFramework:
        """
        Setup K framework for the given programming language.
        """
        program_lang.build_visitors()
        language_name = program_lang.get_name()
        if self.super_pl == "IMP":
            write_to_tmp(program_lang.get_semantics(), f"{language_name}.k")
            self.k_framework = KFramework()
            self.k_framework.compile_k_specification(
                k_file=f"{Macros.tmp_dir}/{language_name}.k",
                output_dir=f"{Macros.tmp_dir}/{language_name}",
            )
        else:
            raise NotImplementedError(f"Language {self.super_pl} is not supported yet.")

    def _sanitize_language_name(self, language_name: str) -> str:
        """
        Convert a language name into a safe identifier for ANTLR/K artifacts.
        """
        sanitized = re.sub(r"[^0-9A-Za-z_]", "_", language_name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            sanitized = "LANG"
        if sanitized[0].isdigit():
            sanitized = f"LANG_{sanitized}"
        return sanitized

    def _get_mutated_language(
        self,
        mutate_patterns: List[str],
        semantics_type: Language.SEMANTICS_TYPE,
        semantics_mutations: Dict[str, str],
        use_k_framework: bool = True,
    ) -> Language:
        """
        Construct the Mutated Language object with the mutation pattern.
        Return:
            K Language with the mutated semantics for execution.
            Language object with the mutated semantics.
        """
        pattern_name = "_".join([mp for mp in mutate_patterns])
        language_name = self._sanitize_language_name(
            f"{self.super_pl}_{semantics_type}_{pattern_name}"
        )
        if self.super_pl == "IMP":
            if use_k_framework:
                execution_k = IMP(
                    name=language_name,
                    semantics_type=Language.SEMANTICS_TYPE.K,
                    **semantics_mutations,
                )
                self._setup_k_framework(program_lang=execution_k)
            else:
                execution_k = None
            program_language = IMP(
                name=language_name,
                semantics_type=semantics_type,
                **semantics_mutations,
            )
            return execution_k, program_language
        else:
            raise NotImplementedError(f"Language {self.super_pl} is not supported yet.")

    def _transform_program(
        self,
        raw_program: str,
        program_language: Language,
    ) -> str:
        """
        Use AST parser to transform the raw program to keep the program semantics.

        Returns:
            The transformed program.
        """
        return program_language.translate_unmutated_program(program=raw_program)

    def _process_pep_dt(self, file_path: Path | str, dt: Dict[str, Any]):
        """
        Process the pep task data.
        """
        is_semantically_valid: bool = True
        base_file_name: str = os.path.basename(file_path)
        for fuzzing_type in IMPFuzzerVisitor.FUZZING_TYPES:
            if fuzzing_type.value in base_file_name:
                dt["semantic-error-type"] = fuzzing_type.value
                dt["semantic-error-rule"] = DatasetProcessor.SEMANTIC_INVALID_RULES_IMP[
                    dt["semantics-type"]
                ][fuzzing_type.value]
                dt["ans"] = "##error##"
                is_semantically_valid = False
                break
            # fi
        # rof
        if is_semantically_valid:
            dt["semantic-error-type"] = "None"
            dt["semantic-error-rule"] = "None"
            dt["ans"] = "##success##"
        # fi
        return

    def _process_op_dt(
        self,
        file_path: Path | str,
        dt: Dict[str, Any],
        execution_k: Language,
        semantics_type: Language.SEMANTICS_TYPE,
        gen_rules_for_srp: bool = False,
    ):
        """
        Process the op task data.
        """
        # execute the program under the language
        try:
            k_result: KResult = self._get_runtime_result(
                file_path, execution_k, semantics_type, gen_rules_for_srp
            )
        except Exception as e:
            logger.warning(
                f"Error in executing program with k framework: {traceback.format_exc()} \n{e}"
            )
            k_result = None
        #
        if k_result:
            dt["K-evaluatable"] = True
            dt["exec-trace"] = k_result.get_execution_trace()
            dt["final-state"] = k_result.get_final_state()
            final_state = k_result.get_final_state()
            sorted_keys = sorted(final_state.keys())
            gt: list = ["<answer>"]
            for key in sorted_keys:
                gt.append(f"  <{key}>{final_state[key]}</{key}>")
            # rof
            gt.append("</answer>")
            dt["ground-truth"] = "\n".join(gt)
        else:
            dt["K-evaluatable"] = False
            dt["exec-trace"] = None
            dt["final-state"] = None
            dt["ground-truth"] = None
        return

    def _process_srp_dt(
        self,
        file_path: Path | str,
        dt: Dict[str, Any],
        execution_k: Language,
        semantics_type: Language.SEMANTICS_TYPE,
        gen_rules_for_srp: bool = True,
    ):
        """
        Process the SRP task data.
        """
        self._process_op_dt(
            file_path, dt, execution_k, semantics_type, gen_rules_for_srp
        )
        exec_trace = dt["exec-trace"]
        sampled_exec_trace: List[Tuple[int, Any]] = random_sample_list(exec_trace)
        if not sampled_exec_trace:
            raise ValueError("Sampled execution trace is empty.")
        #
        # Make the dataset
        data_list = []
        for idx, exec_state in sampled_exec_trace:
            srp_dt = SRPData()
            if idx == 0:
                srp_dt.prior_state = {}
            else:
                srp_dt.prior_state = exec_trace[idx - 1].state

            # Always add this variable (needed for exiting loop for continue)
            srp_dt.prior_state["ble"] = 0
            [srp_dt.cleaned_stmt, srp_dt.line_number] = (
                self._process_annotated_stmt_for_srp(exec_state.stmt)
            )
            srp_dt.rules = exec_state.stmt_rule
            if exec_state.control_stack != "ε":
                srp_dt.control_stack = srp_dt.cleaned_stmt
            else:
                srp_dt.control_stack = exec_state.control_stack
            # fi
            data_list.append(srp_dt)
        #
        dt["sampled-statements"] = data_list
        gt: list = ["<ans>"]
        for i, data in enumerate(data_list):
            gt.append(f'  <answer id="{i + 1}">')
            for rule in data.rules:
                gt.append(f"    <rule>{rule[5:]}</rule>")
            # rof
            gt.append("  </answer>")
        # rof
        gt.append("</ans>")
        dt["ground-truth"] = "\n".join(gt)
        return

    def _process_annotated_stmt_for_srp(self, stmt: str):
        stmt_lines: List[str] = stmt.split("\n")
        annotated_line: int = -1
        for idx, line in enumerate(stmt_lines):
            if "@" in line:
                if annotated_line == -1:
                    annotated_line = idx + 1
                    stmt_lines[idx] = line.replace("@ ", "")
                else:
                    raise ValueError(f"Multiple annotated lines found in {stmt}")
                # fi
            # fi
        # rof
        if annotated_line == -1:
            raise ValueError(f"No annotated/marked lines in {stmt}")
        # fi
        return ("\n".join(stmt_lines), annotated_line)

    # fed

    def _process_etp_dt(
        self,
        file_path: Path | str,
        dt: Dict[str, Any],
        execution_k: Language,
        semantics_type: Language.SEMANTICS_TYPE,
        program_language: Language,
        gen_rules_for_srp: bool = False,
    ):
        """
        Process the SRP task data.
        """
        self._process_op_dt(
            file_path, dt, execution_k, semantics_type, gen_rules_for_srp
        )
        program: str = dt["mutated-program"]
        rules: list = []
        states: list = []
        max_loop_depth: int = -1
        max_if_depth: int = -1
        if semantics_type == Language.SEMANTICS_TYPE.SOS:
            [rules, states, max_loop_depth, max_if_depth] = program_language.get_rule(
                program, {}, program_language.get_semantics_type(), False
            )
            states = states[1:]
            states.append(dt["final-state"])
            for state in states:
                if "ble" in state:
                    del state["ble"]
                # fi
            # rof
        else:
            exec_trace = dt["exec-trace"]
            for idx, element in enumerate(exec_trace):
                for rule in element.rule:
                    rules.append(rule)
                    if "Rule 21" in element.rule:
                        if rule != "Rule 21":
                            states.append(exec_trace[idx - 1].state)
                        else:
                            states.append(element.state)
                        # fi
                    else:
                        states.append(element.state)
                    # fi
                # rof
            # rof
        # fi
        exec_list = []
        for rule, state in zip(rules, states):
            exec_list.append((rule, state))
        # rof

        gold_output = generate_xml_trace(exec_list)
        dt["max-loop-depth"] = max_loop_depth
        dt["max-if-depth"] = max_if_depth
        dt["ground-truth"] = gold_output
        return

    def _process_ast_dt(
        self,
        file_path: Path | str,
        dt: Dict[str, Any],
        execution_k: Language,
        semantics_type: Language.SEMANTICS_TYPE,
    ):
        """
        Process the AST task data.
        """
        self._process_op_dt(file_path, dt, execution_k, semantics_type)
        exec_trace = dt["exec-trace"]
        sampled_exec_trace: List[Tuple[int, Any]] = random_sample_list(exec_trace)
        if not sampled_exec_trace:
            raise ValueError("Sampled execution trace is empty.")
        #
        # Make the dataset
        data_list = []
        statements = dt["mutated-program"].split("\n")
        for idx, exec_state in sampled_exec_trace:
            data_list.append(exec_state.line_number)
        #
        dt["statement"] = statements[int(random.choice(data_list)) - 1]
        return

    def create_dataset_notation_comprehension_unmutated(
        self,
        PL: str,
        semantics_type: Language.SEMANTICS_TYPE,
        task: str,
        setup_name: str,
        num_samples: int,
        random_mode: bool,
        target_dir: Union[str, Path],
        push_to_hub: bool = False,
        repo_id: str = None,
        split: str = None,
    ):
        """
        Create the dataset for unmutated notation comprehension task.
        """
        self.task = task
        self.setup_name = setup_name
        if self.super_pl == "IMP":
            program_language = IMP(
                name=f"{self.super_pl}_{semantics_type.value}",
                semantics_type=semantics_type,
            )
        else:
            raise NotImplementedError(f"Language {self.super_pl} is not supported yet.")
        data_list = self.create_dataset_notation_comprehension(
            PL=PL,
            semantics_type=semantics_type,
            task=task,
            setup_name=setup_name,
            num_samples=num_samples,
            random_mode=random_mode,
            program_language=program_language,
        )
        if push_to_hub:
            push_jsonl_split_to_hf(
                repo_id=repo_id,
                config_name=f"{task}" if semantics_type == Language.SEMANTICS_TYPE.SOS else f"{task}_K",
                split=split,
                data_list=data_list,
                allowed_splits=[
                    "Standard_NumRule5_RandomSampleFalse", 
                    "NonStandard_NumRule5_RandomSampleFalse",
                    "Standard_NumDescription5_RandomSampleFalse", 
                    "NonStandard_NumDescription5_RandomSampleFalse",
                    "Standard_NumRule5_RandomSampleTrue",
                    "NonStandard_NumRule5_RandomSampleTrue",
                    "Standard_NumDescription5_RandomSampleTrue",
                    "NonStandard_NumDescription5_RandomSampleTrue",
                ],
            )
        else:
            su.io.dump(
                Path(target_dir)
                / f"dataset-{task}-{setup_name}-{PL}-{semantics_type.value}-{random_mode}.jsonl",
                data_list,
            )

    def create_dataset_notation_comprehension_mutated(
        self,
        PL: str,
        semantics_type: Language.SEMANTICS_TYPE,
        task: str,
        setup_name: str,
        num_samples: int,
        random_mode: bool,
        target_dir: Union[str, Path],
        push_to_hub: bool = False,
        repo_id: str = None,
        split: str = None,
    ):
        """
        Create the dataset of programs for notation comprehension task.
        """
        self.task = task
        self.setup_name = setup_name
        mutated_data_list = []
        mutation_patterns = self._get_mutation_patterns(True)
        for patterns in mutation_patterns:
            # prepare semantics
            semantics_mutations = {}
            for mutate_pattern in patterns:
                semantics_mutations.update(self.SEMANTICS_MUTATIONS[mutate_pattern])
            #
            _, program_language = self._get_mutated_language(
                mutate_patterns=patterns,
                semantics_type=semantics_type,
                semantics_mutations=semantics_mutations,
                use_k_framework=False,
            )
            data_list = self.create_dataset_notation_comprehension(
                PL=PL,
                semantics_type=semantics_type,
                task=task,
                setup_name=setup_name,
                num_samples=num_samples,
                random_mode=random_mode,
                program_language=program_language,
                patterns=patterns,
            )
            mutated_data_list.extend(data_list)
        if push_to_hub:
            push_jsonl_split_to_hf(
                repo_id=repo_id,
                config_name=f"{task}" if semantics_type == Language.SEMANTICS_TYPE.SOS else f"{task}_K",
                split=split,
                data_list=mutated_data_list,
                allowed_splits=[
                    "Standard_NumRule5_RandomSampleFalse", 
                    "NonStandard_NumRule5_RandomSampleFalse",
                    "Standard_NumDescription5_RandomSampleFalse", 
                    "NonStandard_NumDescription5_RandomSampleFalse",
                    "Standard_NumRule5_RandomSampleTrue",
                    "NonStandard_NumRule5_RandomSampleTrue",
                    "Standard_NumDescription5_RandomSampleTrue",
                    "NonStandard_NumDescription5_RandomSampleTrue",
                ],
            )
        else:
            su.io.dump(
                Path(target_dir)
                / f"dataset-{task}-{setup_name}-{PL}-{semantics_type.value}-{random_mode}.jsonl",
                mutated_data_list,
            )

    def create_dataset_notation_comprehension(
        self,
        PL: str,
        semantics_type: Language.SEMANTICS_TYPE,
        task: str,
        setup_name: str,
        num_samples: int,
        random_mode: bool,
        program_language: Language,
        patterns: List[str] = [],
    ):
        """
        Create the dataset of programs for notation comprehension task.
        """
        if semantics_type == Language.SEMANTICS_TYPE.SOS:
            rule_meta = program_language.build_rule_meta_sos()
        else:
            rule_meta = program_language.build_rule_meta_k()
        sampler = NearMissSampler(rule_meta, rng=random.Random(123))
        data_list = []
        for i in range(num_samples):
            data = {}
            data["language"] = PL
            data["syntax"] = program_language.get_syntax()
            data["semantics-glossary"] = program_language.get_semantics_glossary(semantics_type)
            data["mutated"] = False if setup_name == "uk" else True
            data["semantics-type"] = semantics_type.value
            data["mutation-pattern"] = 'None' if setup_name == "uk" else "_".join(patterns)
            if task == "nl2rule":
                data["num_rules"] = 5
            elif task == "rule2nl":
                data["num_descriptions"] = 5
            # fi
            data["task"] = task
            data["setup-name"] = setup_name
            data["random-mode"] = random_mode
            correct_rule_id = random.choice(list(rule_meta.keys()))
            if task == "nl2rule":
                data = data | build_question_nl_to_rule(correct_rule_id, sampler, random_mode=random_mode, seed=0)
            elif task == "rule2nl":
                data = data | build_question_rule_to_nl(correct_rule_id, sampler, random_mode=random_mode, seed=0)
            else:
                raise ValueError(f"Task {task} is not supported yet.")
            data_list.append(data)
        #rof
        return data_list


###
# Helper functions
###


def random_sample_list(
    exec_list: List[ExecutionState], min_n: int = 2, max_n: int = 10
) -> List[Tuple[int, ExecutionState]]:
    """
    Randomly selects a subset of items from a list, prioritizing rule diversity.

    Groups items by their "rule" value, then samples one item from a diverse
    set of rules, respecting min_n and max_n constraints on the number of
    unique rules sampled.

    Args:
        input_list: A list of ExecutionState objects
        min_n: The minimum number of unique rules desired in the sample.
               If fewer unique rules exist, items from all unique rules
               will be sampled.
        max_n: The maximum number of unique rules to sample from.

    Returns:
        A list of tuples, where each tuple contains the original index
        and the corresponding item from the input_list. Returns an empty
        list if the input list is empty.
    """

    num_items = len(exec_list)

    if num_items == 0:
        logger.warning("The execution list is empty. Returning an empty list.")
        return []

    rules_to_indices = defaultdict(list)

    for idx, item in enumerate(exec_list):
        if len(item.stmt_rule) == 0:
            continue
        # fi
        rules_to_indices[str(item.stmt_rule)].append(idx)
    #
    unique_rules_size = len(rules_to_indices)
    unique_rules = list(rules_to_indices.keys())
    rules_to_sample_from = []
    if unique_rules_size < min_n:
        logger.warning(
            f"Found only {unique_rules_size} unique rules, which is less than "
            f"min_n ({min_n}). Sampling one item from each unique rule found."
        )
        rules_to_sample_from = unique_rules
    elif unique_rules_size > max_n:
        rules_to_sample_from = random.sample(unique_rules, max_n)
    else:
        rules_to_sample_from = unique_rules

    # Sample one index per selected rule
    sampled_indices = []
    for rule in rules_to_sample_from:
        candidate_indices = rules_to_indices[rule]
        if candidate_indices:  # Ensure the list is not empty
            sampled_indices.append(random.choice(candidate_indices))
    sampled_indices.sort()
    # Create the result list by pairing sampled indices with their items
    sampled_list = [(index, exec_list[index]) for index in sampled_indices]

    return sampled_list


def unique_subsets(items: List[Any]) -> List[List[Any]]:
    subsets = []
    for i in range(1, len(items) + 1):
        subsets.extend(combinations(items, i))
    return subsets



# -----------------------------
# Core sampler
# -----------------------------

class NearMissSampler:
    """
    Samples unique distractors with priority:
      1) same Construct
      2) same Category
      3) anywhere else (fallback)

    Can optionally sample distractors completely randomly.
    """

    def __init__(
        self,
        meta_by_rule_id: Dict[str, RuleMeta],
        *,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.meta = meta_by_rule_id
        self.rng = rng or random.Random()

        # Pre-index for fast candidate retrieval
        self._by_construct: Dict[str, List[str]] = {}
        self._by_category: Dict[str, List[str]] = {}

        for rid, m in self.meta.items():
            self._by_construct.setdefault(m.construct, []).append(rid)
            self._by_category.setdefault(m.category, []).append(rid)

    def sample_distractors(
        self,
        correct_rule_id: str,
        *,
        k: int = 4,
        random_mode: bool = False,
        # If you want to *also* enforce that distractors come from distinct constructs/categories:
        enforce_unique_construct: bool = False,
        enforce_unique_category: bool = False,
    ) -> List[str]:
        """
        Returns k unique distractor rule_ids, excluding correct_rule_id.
        If random_mode=True, samples uniformly from all other rules.
        Otherwise uses near-miss priority: Construct -> Category -> Global.
        """
        if correct_rule_id not in self.meta:
            raise KeyError(f"Unknown rule_id: {correct_rule_id}")

        all_ids = list(self.meta.keys())
        pool_all = [rid for rid in all_ids if rid != correct_rule_id]

        if k > len(pool_all):
            raise ValueError(f"Requested k={k} distractors but only {len(pool_all)} available.")

        if random_mode:
            self.rng.shuffle(pool_all)
            return pool_all[:k]

        cm = self.meta[correct_rule_id]
        pool_construct = [rid for rid in self._by_construct.get(cm.construct, []) if rid != correct_rule_id]
        pool_category = [rid for rid in self._by_category.get(cm.category, []) if rid != correct_rule_id]

        # Remove construct hits from category pool to avoid duplicates and keep the priority clean
        set_construct = set(pool_construct)
        pool_category = [rid for rid in pool_category if rid not in set_construct]

        # Fallback pool: everything else not already in earlier pools
        set_early = set_construct | set(pool_category) | {correct_rule_id}
        pool_fallback = [rid for rid in all_ids if rid not in set_early]

        # Shuffle within each tier (so sampling is random but tiered)
        self.rng.shuffle(pool_construct)
        self.rng.shuffle(pool_category)
        self.rng.shuffle(pool_fallback)

        chosen: List[str] = []
        used_constructs = {cm.construct} if enforce_unique_construct else set()
        used_categories = {cm.category} if enforce_unique_category else set()

        def try_take_from(pool: Sequence[str]) -> None:
            nonlocal chosen
            for rid in pool:
                if len(chosen) >= k:
                    return
                m = self.meta[rid]
                if enforce_unique_construct and m.construct in used_constructs:
                    continue
                if enforce_unique_category and m.category in used_categories:
                    continue
                chosen.append(rid)
                if enforce_unique_construct:
                    used_constructs.add(m.construct)
                if enforce_unique_category:
                    used_categories.add(m.category)

        try_take_from(pool_construct)
        try_take_from(pool_category)
        try_take_from(pool_fallback)

        # If uniqueness constraints were too strict, relax them automatically (best-effort)
        if len(chosen) < k:
            remaining = [rid for rid in pool_all if rid not in set(chosen)]
            self.rng.shuffle(remaining)
            chosen.extend(remaining[: (k - len(chosen))])

        return chosen[:k]


# -----------------------------
# Convenience: build a 5-choice question
# -----------------------------

def make_mcq_options(
    correct_id: str,
    distractors: List[str],
    *,
    rng: Optional[random.Random] = None
) -> Tuple[List[str], int]:
    """
    Returns (options, correct_index), where options is length 1+len(distractors).
    Shuffles options and returns the index of correct_id.
    """
    rng = rng or random.Random()
    options = [correct_id] + list(distractors)
    rng.shuffle(options)
    return options, options.index(correct_id)


def build_question_nl_to_rule(
    correct_rule_id: str,
    sampler: NearMissSampler,
    *,
    random_mode: bool = False,
    seed: Optional[int] = None,
) -> Dict:
    rng = random.Random(seed)
    # Use the same rng for sampler + shuffling to make runs reproducible
    sampler_local = NearMissSampler(sampler.meta, rng=rng)

    distractors = sampler_local.sample_distractors(
        correct_rule_id,
        k=4,
        random_mode=random_mode,
    )
    option_ids, answer_index = make_mcq_options(correct_rule_id, distractors, rng=rng)
    return {
        "task": "nl2rule",
        "question": sampler.meta[correct_rule_id].nl_description,
        "options": [f"Rule {i} := {{{sampler.meta[rid].rule_text}}}" for i,rid in enumerate(option_ids)],
        "option_rule_ids": [sampler.meta[rid].rule_id for rid in option_ids],          # keep IDs for analysis/debugging
        "answer_index": answer_index,
        "answer_rule_id": sampler.meta[correct_rule_id].rule_id,
        "sampling_mode": "random" if random_mode else "near_miss",
    }


def build_question_rule_to_nl(
    correct_rule_id: str,
    sampler: NearMissSampler,
    *,
    random_mode: bool = False,
    seed: Optional[int] = None,
) -> Dict:
    rng = random.Random(seed)
    sampler_local = NearMissSampler(sampler.meta, rng=rng)

    distractors = sampler_local.sample_distractors(
        correct_rule_id,
        k=4,
        random_mode=random_mode,
    )
    option_ids, answer_index = make_mcq_options(correct_rule_id, distractors, rng=rng)
    return {
        "task": "rule2nl",
        "question": f"Rule := {{{sampler.meta[correct_rule_id].rule_text}}}",
        "options": [f"Description {i} := {sampler.meta[rid].nl_description}" for i,rid in enumerate(option_ids)],
        "option_rule_ids": [sampler.meta[rid].rule_id for rid in option_ids],
        "answer_index": answer_index,
        "answer_rule_id": sampler.meta[correct_rule_id].rule_id,
        "sampling_mode": "random" if random_mode else "near_miss",
    }


