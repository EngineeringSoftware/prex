import seutil as su
import json
import textwrap

from typing import List
from llm_interpreter.experiments.args import ExperimentArgs
from llm_interpreter.experiments.prompts import (
    PROMPT_STRATEGY,
    pep_with_semantics_sos_prompt,
    pep_with_semantics_sos_cot_prompt,
    pep_with_semantics_k_prompt,
    pep_with_semantics_k_cot_prompt,
    op_with_semantics_sos_prompt,
    op_with_semantics_sos_cot_prompt,
    op_with_semantics_k_prompt,
    op_with_semantics_k_cot_prompt,
    op_with_semantics_five_shot_sos_prompt,
    op_with_semantics_five_shot_k_prompt,
    op_no_semantics_prompt,
    op_no_semantics_cot_prompt,
    srp_stmt_prompt,
    srp_zero_shot_sos_prompt,
    srp_zero_shot_sos_cot_prompt,
    srp_zero_shot_k_prompt,
    srp_zero_shot_k_cot_prompt,
    etp_imp_sos_prompt,
    etp_imp_sos_cot_prompt,
    etp_imp_k_prompt,
    etp_imp_k_cot_prompt,
    python_code_translation_prompt,
    ig_imp_sos_prompt,
    iga_prompt,
    igaf_timeout_prompt,
    igaf_build_fail_prompt,
    igaf_valid_wr_prompt,
    igaf_valid_wa_prompt,
    igaf_invalid_wr_prompt,
    igaf_invalid_wa_prompt,
    igaf_prefix,
    igaf_suffix,
    ast_prompt,
    formal_semantics_notation_comprehension_prompt_sos_rule_2_nl,
    formal_semantics_notation_comprehension_prompt_sos_nl_2_rule,
    formal_semantics_notation_comprehension_prompt_k_rule_2_nl,
    formal_semantics_notation_comprehension_prompt_k_nl_2_rule,
)
from llm_interpreter.data.data_generator_prompts import (
    translation_c_to_imp_prompt,
    dgc_prompt,
)
from llm_interpreter.data.dataset_process import DatasetProcessor
from llm_interpreter.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)

PROMPTS_DICT: dict = {
    # PCP
    "pcp-IMP-SOS-da": pep_with_semantics_sos_prompt,
    "pcp-IMP-SOS-cot": pep_with_semantics_sos_cot_prompt,
    "pcp-IMP-K-da": pep_with_semantics_k_prompt,
    "pcp-IMP-K-cot": pep_with_semantics_k_cot_prompt,
    # POP
    "op-IMP-SOS-da": op_with_semantics_sos_prompt,
    "op-IMP-SOS-cot": op_with_semantics_sos_cot_prompt,
    "op-IMP-K-da": op_with_semantics_k_prompt,
    "op-IMP-K-cot": op_with_semantics_k_cot_prompt,
    # SRP
    "srp-IMP-SOS-da": srp_zero_shot_sos_prompt,
    "srp-IMP-SOS-cot": srp_zero_shot_sos_cot_prompt,
    "srp-IMP-K-da": srp_zero_shot_k_prompt,
    "srp-IMP-K-cot": srp_zero_shot_k_cot_prompt,
    # ETP
    "etp-IMP-SOS-da": etp_imp_sos_prompt,
    "etp-IMP-SOS-cot": etp_imp_sos_cot_prompt,
    "etp-IMP-K-da": etp_imp_k_prompt,
    "etp-IMP-K-cot": etp_imp_k_cot_prompt,
    # Formal semantics notation comprehension
    "rule2nl-IMP-SOS": formal_semantics_notation_comprehension_prompt_sos_rule_2_nl,
    "nl2rule-IMP-SOS": formal_semantics_notation_comprehension_prompt_sos_nl_2_rule,
    "rule2nl-IMP-K": formal_semantics_notation_comprehension_prompt_k_rule_2_nl,
    "nl2rule-IMP-K": formal_semantics_notation_comprehension_prompt_k_nl_2_rule,
}


def make_rule2nl_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    prompt_template = PROMPTS_DICT[
        f"{args.task}-{args.expr_name}"
    ]
    #
    if (
        args.setup_name == "mk"
        and "addSub_mulDiv_negateRelation" in dt["mutation-pattern"]
    ):
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=textwrap.indent(dt["syntax"], "    "),
            semantics_glossary=dt["semantics-glossary"],
            num_descriptions=len(dt["options"]),
            descriptions=textwrap.indent("\n\n".join(dt["options"]), "    "),
            rule=dt["question"],
            ERROR="ERROR",
            HALT="halt",
        )
    elif args.setup_name == "mk" and "unseen" in dt["mutation-pattern"]:
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=textwrap.indent(dt["syntax"], "    "),
            semantics_glossary=dt["semantics-glossary"],
            num_descriptions=len(dt["options"]),
            descriptions=textwrap.indent("\n\n".join(dt["options"]), "    "),
            rule=dt["question"],
            ERROR=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["ERROR"],
            HALT=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["HALT"],
        )
    else:
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=textwrap.indent(dt["syntax"], "    "),
            semantics_glossary=dt["semantics-glossary"],
            num_descriptions=len(dt["options"]),
            descriptions=textwrap.indent("\n\n".join(dt["options"]), "    "),
            rule=dt["question"],
            ERROR="ERROR",
            HALT="halt",
        )
    #
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_nl2rule_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    prompt_template = PROMPTS_DICT[
        f"{args.task}-{args.expr_name}"
    ]
    #
    if (
        args.setup_name == "mk"
        and "addSub_mulDiv_negateRelation" in dt["mutation-pattern"]
    ):
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=textwrap.indent(dt["syntax"], "    "),
            semantics_glossary=dt["semantics-glossary"],
            num_rules=len(dt["options"]),
            rules=textwrap.indent("\n\n".join(dt["options"]), "    "),
            description=dt["question"],
            ERROR="ERROR",
            HALT="halt",
        )
    elif args.setup_name == "mk" and "unseen" in dt["mutation-pattern"]:
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=textwrap.indent(dt["syntax"], "    "),
            semantics_glossary=dt["semantics-glossary"],
            num_rules=len(dt["options"]),
            rules=textwrap.indent("\n\n".join(dt["options"]), "    "),
            description=dt["question"],
            ERROR=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["ERROR"],
            HALT=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["HALT"],
        )
    else:
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=textwrap.indent(dt["syntax"], "    "),
            semantics_glossary=dt["semantics-glossary"],
            num_rules=len(dt["options"]),
            rules=textwrap.indent("\n\n".join(dt["options"]), "    "),
            description=dt["question"],
            ERROR="ERROR",
            HALT="halt",
        )
    #
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_pep_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    program = dt["program"] if args.setup_name != "mk" else dt["mutated-program"]
    semantics = dt["semantics"]
    if (
        args.setup_name == "mk"
        and dt["mutation-pattern"] == "KeywordObf"
        and DatasetProcessor.pcp_keyword_obf_uses_gpt4o_1_token(args.model_name)
    ):
        program = DatasetProcessor.remap_mutation_symbols(
            program,
            DatasetProcessor.KEYWORD_OBF_LEGACY,
            DatasetProcessor.KEYWORD_OBF_GPT4O_1_TOKEN,
        )
        semantics = DatasetProcessor.remap_mutation_symbols(
            semantics,
            DatasetProcessor.KEYWORD_OBF_LEGACY,
            DatasetProcessor.KEYWORD_OBF_GPT4O_1_TOKEN,
        )
    prompt_template = PROMPTS_DICT[
        f"{args.task}-{args.expr_name}-{args.prompt_strategy}"
    ]
    if (
        args.setup_name == "mk"
        and dt["mutation-pattern"] == "KeywordSwap"
    ):
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=semantics,
            program=program,
            ERROR="ERROR",
            HALT="halt",
            CONTINUE="continue",
        )
    elif (
        args.setup_name == "mk"
        and dt["mutation-pattern"] == "KeywordObf"
    ):
        obf = DatasetProcessor.keyword_obf_mutations(args.model_name)
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=semantics,
            program=program,
            ERROR=obf["ERROR"],
            HALT=obf["HALT"],
            CONTINUE=obf["CONTINUE"],
        )
    else:
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=semantics,
            program=program,
            ERROR="ERROR",
            HALT="halt",
            CONTINUE="continue",
        )
    #
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_op_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    program = dt["program"] if args.setup_name != "mk" else dt["mutated-program"]
    if args.setup_name != "nk":
        prompt_template = PROMPTS_DICT[
            f"{args.task}-{args.expr_name}-{args.prompt_strategy}"
        ]
    elif args.setup_name == "nk":
        if args.prompt_strategy == PROMPT_STRATEGY.DA and "SOS" in args.expr_name:
            prompt_template = op_no_semantics_prompt
        elif args.prompt_strategy == PROMPT_STRATEGY.COT and "SOS" in args.expr_name:
            prompt_template = op_no_semantics_cot_prompt
    #
    if (
        args.setup_name == "mk" and dt["mutation-pattern"] == "KeywordSwap"
    ):
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=dt["semantics"],
            program=program,
            ERROR="ERROR",
            HALT="halt",
            ASSIGN_OP="=",
            ADD_OP="-",
        )
    elif args.setup_name == "mk" and dt["mutation-pattern"] == "KeywordObf":
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=dt["semantics"],
            program=dt["mutated-program"],
            ERROR=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["ERROR"],
            HALT=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["HALT"],
            ASSIGN_OP=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["ASSIGN_OP"],
            ADD_OP=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["PLUS_OP"],
        )
    elif args.setup_name == "uk" or args.setup_name == "nk":
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=dt["semantics"],
            program=program,
            ERROR="ERROR",
            HALT="halt",
            ASSIGN_OP="=",
            ADD_OP="+",
        )
    else:
        raise ValueError(f"Unknown mutation pattern: {dt['mutation-pattern']}")
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_srp_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    """
    Prepare the prompt for the SRP task of a given program.
    """
    program = dt["program"] if args.setup_name != "mk" else dt["mutated-program"]
    prompt_template = PROMPTS_DICT[
        f"{args.task}-{args.expr_name}-{args.prompt_strategy}"
    ]
    if (
        args.setup_name == "mk"
        and dt["mutation-pattern"] == "KeywordSwap"
    ):
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=dt["semantics"],
            questions=_prepare_srp_questions(dt, program),
            LTEQ_OP=DatasetProcessor.SEMANTICS_MUTATIONS[
                "addSub_mulDiv_negateRelation"
            ]["LTEQ_OP"],
            NOT_OP="!",
            ERROR="ERROR",
            HALT="halt",
            WHILE="while",
            LOOP="loop",
            IF="if",
            ELSE="else",
        )
    elif args.setup_name == "mk" and dt["mutation-pattern"] == "KeywordObf":
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=dt["semantics"],
            questions=_prepare_srp_questions(dt, program),
            LTEQ_OP=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["LTEQ_OP"],
            NOT_OP=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["NOT_OP"],
            ERROR=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["ERROR"],
            HALT=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["HALT"],
            WHILE=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["WHILE"],
            LOOP=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["LOOP"],
            IF=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["IF"],
            ELSE=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["ELSE"],
        )
    elif args.setup_name == "uk":
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=dt["semantics"],
            questions=_prepare_srp_questions(dt, program),
            LTEQ_OP="<=",
            NOT_OP="!",
            ERROR="ERROR",
            HALT="halt",
            WHILE="while",
            LOOP="loop",
            IF="if",
            ELSE="else",
        )
    else:
        raise ValueError(f"Unknown mutation pattern: {dt['mutation-pattern']}")
    #
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_etp_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    program = dt["program"] if args.setup_name != "mk" else dt["mutated-program"]
    prompt_template = PROMPTS_DICT[
        f"{args.task}-{args.expr_name}-{args.prompt_strategy}"
    ]
    if (
        args.setup_name == "mk"
        and dt["mutation-pattern"] == "KeywordSwap"
    ):
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=dt["semantics"],
            program=program,
            ERROR="ERROR",
            HALT="halt",
            WHILE="while",
            ASSIGN_OP="=",
            LT_OP=">",
            PLUS_OP="-",
            IF="if",
            ELSE="else",
        )
    elif args.setup_name == "mk" and dt["mutation-pattern"] == "KeywordObf":
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=dt["semantics"],
            program=program,
            ERROR=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["ERROR"],
            HALT=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["HALT"],
            WHILE=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["WHILE"],
            ASSIGN_OP=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["ASSIGN_OP"],
            LT_OP=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["LT_OP"],
            PLUS_OP=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["PLUS_OP"],
            IF=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["IF"],
            ELSE=DatasetProcessor.SEMANTICS_MUTATIONS["unseen"]["ELSE"],
        )
    elif args.setup_name == "uk":
        prompt = prompt_template.format(
            language=dt["language"],
            syntax=dt["syntax"],
            semantics=dt["semantics"],
            program=program,
            ERROR="ERROR",
            HALT="halt",
            WHILE="while",
            ASSIGN_OP="=",
            LT_OP="<",
            PLUS_OP="+",
            IF="if",
            ELSE="else",
        )
    else:
        raise ValueError(f"Unknown mutation pattern: {dt['mutation-pattern']}")
    #
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_ast_prompt(dt: dict) -> List[dict]:
    # statement = dt["statement"] if args.setup_name != "mk" else dt["mutated-statement"]
    prompt = ast_prompt.format(
        language=dt["language"],
        syntax=dt["syntax"],
        semantics=dt["semantics"],
        statement=dt["statement"],
    )
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_translate_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    program = dt["program"] if args.setup_name != "mk" else dt["mutated-program"]
    prompt = python_code_translation_prompt.format(
        language=dt["language"],
        syntax=dt["syntax"],
        semantics=dt["semantics"],
        program=program,
    )
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_ig_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    prompt = ig_imp_sos_prompt.format(
        syntax_type=dt["syntax-type"],
        language=dt["language"],
        syntax=dt["syntax"],
        semantics=dt["semantics"],
        impl_language=dt["impl-language"],
        additional_notes=dt.get("additional-notes", ""),
    )
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_iga_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    mutation_pattern = dt.get("mutation-pattern", "uk")
    prompt = iga_prompt.format(
        syntax=dt["syntax"],
        semantics=dt["semantics"],
        language=dt["language"],
        impl_language=dt["impl-language"],
        base_visitor_code=dt["base-visitor-code"],
        parser_code=dt["parser-code"],
        extend_visitor_name=dt["extend-visitor-name"],
        extend_visitor_code=dt["extend-visitor-code"],
        states_field=dt["states-field"],
        main_code=dt["main-code"],
        valid_example_0=dt["valid-example-0"],
        valid_output_0=dt["valid-output-0"],
        invalid_example_0=dt["invalid-example-0"],
        invalid_output_0=dt["invalid-output-0"],
        additional_notes=dt.get("additional-notes", ""),
    )
    su.io.dump(
        Macros.tmp_dir
        / "prompts"
        / f"iga-prompt-{dt['impl-language']}-{mutation_pattern}.txt",
        prompt,
    )
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_igaf_prompt(args: ExperimentArgs, dt: dict) -> List[dict]:
    idx = dt["idx"]
    impl_lang = dt["impl-language"]
    feedback = dt["feedback"]
    mutation_pattern = feedback["mutation_pattern"]
    intp_program = feedback["intp_program"]

    # TODO: maybe find a better way than read from tmp
    previous_prompt = su.io.load(
        Macros.tmp_dir / "prompts" / f"iga-prompt-{impl_lang}-{mutation_pattern}.txt",
        fmt=su.io.Fmt.txt,
    )
    previous_response = f"```{impl_lang}\n{intp_program}\n```"
    new_prompt = _prepare_igaf_feedback_prompt(feedback, mutation_pattern)

    if new_prompt == "":
        logger.warning(f"No feedback for: {impl_lang}-{mutation_pattern}-{idx}")
        return None

    chat = [
        {"role": "user", "content": previous_prompt},
        {"role": "assistant", "content": previous_response},
        {"role": "user", "content": new_prompt},
    ]
    su.io.dump(
        Macros.tmp_dir
        / "prompts"
        / f"igaf-prompt-{impl_lang}-{mutation_pattern}-{idx}.txt",
        chat,
    )
    return chat


def make_dataset_generation_prompt(dt: dict) -> List[dict]:
    """
    Prepare the prompt for dataset generation tasks.
    This function is a placeholder and can be extended based on specific requirements.
    """
    # Currently, this function does not generate any specific prompt.
    # It can be customized later as needed.
    import random

    random.seed(42)  # For reproducibility
    random_test = random.choice(dt["publicTests"])
    prompt = translation_c_to_imp_prompt.format(
        code=dt["sourceCode"],
        test=random_test["input"],
    )
    chat = [{"role": "user", "content": prompt}]
    return chat


def make_dgc_prompt(dt: dict) -> List[dict]:
    prompt = dgc_prompt.format(
        code=dt["program"],
    )
    chat = [{"role": "user", "content": prompt}]
    su.io.dump(
        Macros.tmp_dir / "prompts" / "dgc.txt",
        chat,
    )
    return chat


###
# Helper functions
###


def _prepare_igaf_feedback_prompt(feedback: dict, mutation_pattern: str) -> str:
    prompt = ""

    for i, ve in enumerate(Macros.valid_examples):
        state = feedback[f"valid_example_{i}"]
        assert ve == state["program_name"], (
            f"VE_{i} program name mismatch: {ve} vs {state['program_name']}"
        )
        if "Build failed" in prompt:
            break
        prompt += _determine_ve_state(i, state, mutation_pattern)

    for i, ive in enumerate(Macros.invalid_examples):
        state = feedback[f"invalid_example_{i}"]
        assert ive == state["program_name"], (
            f"IVE_{i} program name mismatch: {ive} vs {state['program_name']}"
        )
        if "Build failed" in prompt:
            break
        prompt += _determine_ive_state(i, state, mutation_pattern)

    if prompt != "":
        prompt = igaf_prefix + prompt + igaf_suffix

    return prompt


def _determine_ve_state(idx: int, state: dict, mutation_pattern: str) -> str:
    prompt = ""

    pgm_name = state["program_name"]
    exit_code = state["exit_code"]
    stdout = state["stdout"]
    stderr = state["stderr"]

    if mutation_pattern == "uk":
        valid_imp_dir = Macros.valid_imp_uk_dir
    elif mutation_pattern == "ks":
        valid_imp_dir = Macros.valid_imp_mk_ks_dir
    elif mutation_pattern == "ko":
        valid_imp_dir = Macros.valid_imp_mk_ko_dir
    else:
        raise ValueError(f"Unknown mutation pattern: {mutation_pattern}")

    ans_name = pgm_name.replace(".imp", ".txt")
    pgm_code = su.io.load(valid_imp_dir / pgm_name, fmt=su.io.Fmt.txt)
    ans = su.io.load(Macros.valid_imp_ans_dir / ans_name, fmt=su.io.Fmt.txt)

    try:
        stdout = json.loads(
            stdout,
        )
    except json.JSONDecodeError:
        stdout = stdout.strip()
    expected_output = json.loads(ans)

    if state["is_timeout"]:
        prompt = igaf_timeout_prompt.format(
            pgm_type="valid",
            idx=idx + 1,
            code=pgm_code,
            stdout=stdout,
            stderr=stderr,
            expected_output=expected_output,
        )
    elif state["is_build_failed"]:
        prompt = igaf_build_fail_prompt.format(
            stdout=stdout,
            stderr=stderr,
        )
    elif exit_code != 0:
        prompt = igaf_valid_wr_prompt.format(
            idx=idx + 1,
            code=pgm_code,
            stdout=stdout,
            stderr=stderr,
            expected_output=expected_output,
        )
    elif stdout != expected_output:
        prompt = igaf_valid_wa_prompt.format(
            idx=idx + 1,
            code=pgm_code,
            stdout=stdout,
            expected_output=expected_output,
        )
    else:
        assert exit_code == 0 and stdout == expected_output, (
            "Should pass the test case."
        )

    return prompt


def _determine_ive_state(idx: int, state: dict, mutation_pattern: str) -> str:
    prompt = ""

    pgm_name = state["program_name"]
    exit_code = state["exit_code"]
    stdout = state["stdout"]
    stderr = state["stderr"]

    if mutation_pattern == "uk":
        invalid_imp_dir = Macros.invalid_imp_uk_dir
    elif mutation_pattern == "ks":
        invalid_imp_dir = Macros.invalid_imp_mk_ks_dir
    elif mutation_pattern == "ko":
        invalid_imp_dir = Macros.invalid_imp_mk_ko_dir
    else:
        raise ValueError(f"Unknown mutation pattern: {mutation_pattern}")

    pgm_code = su.io.load(invalid_imp_dir / pgm_name, fmt=su.io.Fmt.txt)
    for type, output in Macros.invalid_outputs.items():
        if type in pgm_name:
            expected_output = output
            break

    if state["is_timeout"]:
        prompt = igaf_timeout_prompt.format(
            pgm_type="invalid",
            idx=idx + 1,
            code=pgm_code,
            stdout=stdout,
            stderr=stderr,
            expected_output=expected_output,
        )
    elif state["is_build_failed"]:
        prompt = igaf_build_fail_prompt.format(
            stdout=stdout,
            stderr=stderr,
        )
    elif exit_code == 0:
        prompt = igaf_invalid_wr_prompt.format(
            idx=idx + 1,
            code=pgm_code,
            stdout=stdout,
            expected_output=expected_output,
        )
    elif stdout.strip() != expected_output.strip():
        prompt = igaf_invalid_wa_prompt.format(
            idx=idx + 1,
            code=pgm_code,
            stdout=stdout,
            stderr=stderr,
            expected_output=expected_output,
        )
    else:
        assert exit_code != 0 and stdout.strip() == expected_output.strip(), (
            "Should report correct error."
        )

    return prompt


def _prepare_srp_questions(dt: dict, program: str) -> str:
    """Prepare the prompt for the SRP task of a given program."""
    srp_data = dt["sampled-statements"]
    question_prompt = ""
    for i, srp_dt in enumerate(srp_data):
        question_prompt += srp_stmt_prompt.format(
            index=i + 1,
            line_number=int(srp_dt["line_number"]),
            program_state=srp_dt["prior_state"],
            statement=srp_dt["cleaned_stmt"],
            control_stack=srp_dt["control_stack"],
        )
        question_prompt += "\n\n"
    #
    return question_prompt


def _add_linenum_to_program(program: str) -> str:
    """Add line numbers to the program."""
    lines = program.strip().split("\n")
    numbered_lines = [f"{i + 1}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)
