"""
For processing/cleaning dataset or collect dataset stats.

Methods should be named result_<method_name> and annotated
with  @subcommand(category=Category.DATA)
"""

import os
import platform
import random
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import seutil as su
from antlr4 import CommonTokenStream, InputStream
from datasets import load_dataset
from tqdm import tqdm

from llm_interpreter.antlr4_parsers import IMPBASELexer, IMPBASEParser
from llm_interpreter.compiler_runners import KFramework, KFrameworkError
from llm_interpreter.data.data_analysis import (
    DataAnalysis,
    collect_dataset_code_complexity_metrics,
)
from llm_interpreter.data.dataset_process import DatasetProcessor
from llm_interpreter.data.intp_functions import *  # noqa: F403
from llm_interpreter.language import IMP, Language
from llm_interpreter.language.metrics import (
    DepDegreeMetric,
    ExtendedCyclomaticMetric,
    HalsteadMetric,
)
from llm_interpreter.language.visitors import IMPFuzzerVisitor
from llm_interpreter.macros import Macros
from llm_interpreter.utils import Category, subcommand, write_to_dir, write_to_tmp

if platform.system() == "Linux":
    from transformers import AutoTokenizer
# fi


logger = su.log.get_logger(__name__, su.log.INFO)

PCP_SEMANTIC_ERROR_SUFFIXES = (
    "divide_by_zero",
    "modulo_zero",
    "break_outside_loop",
    "continue_outside_loop",
    "var_use_before_declare",
)

PCP_FUZZ_REPORTS_DIR = Macros.data_dir / "imp" / "pcp_fuzz_reports"
PCP_PROGRAMS_DIR = Macros.data_dir / "imp" / "extended-valid-invalid-dataset"

K_SEMANTIC_ERROR_RULES = tuple(
    DatasetProcessor.SEMANTIC_INVALID_RULES_IMP["K"].values()
)

DEFAULT_ALREADY_FUZZED_INVALID_DIRS = (
    Macros.data_dir / "imp" / "invalid_imp_programs" / "human_written",
    Macros.data_dir / "imp" / "invalid_imp_programs" / "fuzzer_generated",
    Macros.data_dir / "imp" / "invalid_imp_programs" / "synthetic_cpp",
    Macros.data_dir / "imp" / "invalid_imp_programs",
)

_PCP_FUZZ_K_FRAMEWORK: Optional[KFramework] = None
_PCP_FUZZ_IMP_LANG: Optional[IMP] = None


@dataclass
class FuzzProgramOutcome:
    valid_src_filename: str
    fuzz_method: str  # visitor | fallback | failed
    invalid_src_filename: Optional[str] = None
    fuzzing_type: Optional[str] = None
    visitor_failures: List[Dict[str, str]] = field(default_factory=list)
    fallback_failures: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None
    fuzzed_program: Optional[str] = None


@dataclass
class FuzzBatchResult:
    success_count: int
    failure_count: int
    visitor_count: int
    fallback_count: int
    outcomes: List[FuzzProgramOutcome]
    report_path: Optional[Path] = None


@dataclass
class FuzzByCategorySuccess:
    valid_src_filename: str
    invalid_src_filename: str
    category: str
    seed: int
    target_site_ids: List[int]
    output_path: str


@dataclass
class AssignSite:
    site_id: int
    depth: int
    in_loop: bool
    in_if: bool


@dataclass
class FlowControlSite:
    kind: str
    depth: int
    in_loop: bool
    line_number: int


@dataclass
class FuzzByCategoryFailure:
    valid_src_filename: str
    category: str
    reason: str
    attempts: int
    attempt_errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FuzzByCategoryResult:
    category: str
    requested_count: int
    success_count: int
    failure_count: int
    skipped_already_fuzzed_count: int
    successes: List[FuzzByCategorySuccess]
    failures: List[FuzzByCategoryFailure]
    report_path: Optional[Path] = None
    valid_split: Optional[str] = None


@dataclass
class PCPDataSplit:
    name: str
    valid_dir: Path
    output_dir: Path


@dataclass
class FuzzByCategoryAllSplitsResult:
    category: str
    requested_count_total: int
    split_results: List[FuzzByCategoryResult]
    report_path: Optional[Path] = None

    @property
    def success_count(self) -> int:
        return sum(result.success_count for result in self.split_results)

    @property
    def failure_count(self) -> int:
        return sum(result.failure_count for result in self.split_results)

    @property
    def skipped_already_fuzzed_count(self) -> int:
        return sum(result.skipped_already_fuzzed_count for result in self.split_results)


def get_pcp_data_splits() -> List[PCPDataSplit]:
    imp_root = (Macros.data_dir / "imp").resolve()
    return [
        PCPDataSplit(
            name="human_written",
            valid_dir=(imp_root / "valid_imp_programs" / "human_written").resolve(),
            output_dir=(imp_root / "invalid_imp_programs" / "human_written").resolve(),
        ),
        PCPDataSplit(
            name="fuzzer_generated",
            valid_dir=(imp_root / "fuzzer_generated").resolve(),
            output_dir=(
                imp_root / "invalid_imp_programs" / "fuzzer_generated"
            ).resolve(),
        ),
        PCPDataSplit(
            name="synthetic_cpp",
            valid_dir=(imp_root / "valid_imp_programs" / "synthetic_cpp").resolve(),
            output_dir=(imp_root / "invalid_imp_programs" / "synthetic_cpp").resolve(),
        ),
    ]


@dataclass
class PCPCategoryFuzzCandidate:
    file_path: Path
    split: PCPDataSplit
    output_dir: Path


def _collect_pcp_category_fuzz_candidates(
    splits: List[PCPDataSplit],
    category_value: str,
    *,
    skip_already_fuzzed: bool,
    already_fuzzed_invalid_dirs: List[Path],
    output_dir_base: Optional[Path],
) -> tuple[List[PCPCategoryFuzzCandidate], Dict[str, int]]:
    skipped_by_split: Dict[str, int] = {split.name: 0 for split in splits}
    skipped_already_fuzzed: set[str] = set()
    if skip_already_fuzzed:
        skipped_already_fuzzed = _collect_valid_files_already_fuzzed_for_category(
            already_fuzzed_invalid_dirs,
            category_value,
        )

    candidates: List[PCPCategoryFuzzCandidate] = []
    for split in splits:
        split_output_dir = (
            output_dir_base / split.name
            if output_dir_base is not None
            else split.output_dir
        )
        for file_path in sorted(split.valid_dir.glob("*.imp")):
            if file_path.name in skipped_already_fuzzed:
                skipped_by_split[split.name] += 1
                continue
            candidates.append(
                PCPCategoryFuzzCandidate(
                    file_path=file_path,
                    split=split,
                    output_dir=split_output_dir,
                )
            )
    return candidates, skipped_by_split


def _normalize_fuzz_program_filename(name: str) -> str:
    filename = name.strip().strip("{}").strip("'\"")
    if not filename:
        raise ValueError("Program filename cannot be empty.")
    if not filename.endswith(".imp"):
        filename = f"{filename}.imp"
    return Path(filename).name


def _resolve_existing_program_path(program_name: str) -> Optional[Path]:
    raw = program_name.strip().strip("{}").strip("'\"")
    if not raw:
        return None

    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates.append(Path.cwd() / raw)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _infer_pcp_split_for_valid_program_path(
    file_path: Path,
    splits: List[PCPDataSplit],
) -> PCPDataSplit:
    resolved = file_path.resolve()
    for split in splits:
        split_valid_dir = split.valid_dir.resolve()
        if resolved == split_valid_dir or split_valid_dir in resolved.parents:
            return split

    for split in splits:
        if split.name == resolved.parent.name:
            return split

    split_dirs = "\n  - ".join(str(split.valid_dir.resolve()) for split in splits)
    raise FileNotFoundError(
        f"Valid program {resolved} is not under any known PCP split directory. "
        f"Expected one of:\n  - {split_dirs}"
    )


def _normalize_fuzz_programs_arg(
    programs: Optional[Union[str, List[str]]],
) -> Optional[List[str]]:
    if programs is None:
        return None
    if isinstance(programs, str):
        program_names = [part.strip() for part in programs.split(",")]
    else:
        program_names = [part.strip() for name in programs for part in name.split(",")]
    program_names = [name for name in program_names if name]
    if not program_names:
        raise ValueError("At least one program filename must be provided.")
    return program_names


def _resolve_pcp_program_candidate(
    program_name: str,
    splits: List[PCPDataSplit],
    *,
    output_dir_base: Optional[Path],
) -> PCPCategoryFuzzCandidate:
    explicit_path = _resolve_existing_program_path(program_name)
    if explicit_path is not None:
        split = _infer_pcp_split_for_valid_program_path(explicit_path, splits)
        split_output_dir = (
            output_dir_base / split.name
            if output_dir_base is not None
            else split.output_dir
        )
        return PCPCategoryFuzzCandidate(
            file_path=explicit_path,
            split=split,
            output_dir=split_output_dir.resolve(),
        )

    filename = _normalize_fuzz_program_filename(program_name)
    matches: List[tuple[PCPDataSplit, Path]] = []
    for split in splits:
        file_path = (split.valid_dir / filename).resolve()
        if file_path.is_file():
            matches.append((split, file_path))

    if not matches:
        searched = []
        for split in splits:
            searched.append(f"{split.name}: {split.valid_dir.resolve()}")
        searched_text = "\n  - ".join(searched)
        raise FileNotFoundError(
            f"Valid program {filename} was not found in any PCP split. Searched:\n"
            f"  - {searched_text}"
        )
    if len(matches) > 1:
        located_in = ", ".join(split.name for split, _ in matches)
        raise ValueError(
            f"Ambiguous program {filename}: found in multiple PCP splits ({located_in})."
        )

    split, file_path = matches[0]
    split_output_dir = (
        output_dir_base / split.name
        if output_dir_base is not None
        else split.output_dir
    )
    return PCPCategoryFuzzCandidate(
        file_path=file_path,
        split=split,
        output_dir=split_output_dir.resolve(),
    )


def _resolve_pcp_program_candidates(
    program_names: List[str],
    splits: List[PCPDataSplit],
    *,
    output_dir_base: Optional[Path],
) -> List[PCPCategoryFuzzCandidate]:
    if not program_names:
        raise ValueError("At least one program filename must be provided.")
    return [
        _resolve_pcp_program_candidate(name, splits, output_dir_base=output_dir_base)
        for name in program_names
    ]


def _resolve_program_files_in_valid_dir(
    program_names: List[str],
    valid_dir: Path,
) -> List[Path]:
    if not program_names:
        raise ValueError("At least one program filename must be provided.")
    resolved_valid_dir = valid_dir.resolve()
    resolved: List[Path] = []
    for program_name in program_names:
        explicit_path = _resolve_existing_program_path(program_name)
        if explicit_path is not None:
            resolved.append(explicit_path)
            continue

        filename = _normalize_fuzz_program_filename(program_name)
        file_path = (resolved_valid_dir / filename).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Valid program {filename} was not found in {resolved_valid_dir}"
            )
        resolved.append(file_path)
    return resolved


def _run_fuzz_by_category_all_pcp_splits(
    *,
    category_value: str,
    fuzzing_category: IMPFuzzerVisitor.FUZZING_TYPES,
    n: int,
    seed: int,
    skip_already_fuzzed: bool,
    already_fuzzed_invalid_dirs: List[Path],
    max_attempts_per_program: int,
    output_dir_base: Optional[Path],
    combined_report_path: Path,
    imp_lang: IMP,
    k_framework: KFramework,
    k_validate_lang: IMP,
    program_names: Optional[List[str]] = None,
) -> FuzzByCategoryAllSplitsResult:
    splits = get_pcp_data_splits()
    skipped_by_split: Dict[str, int] = {split.name: 0 for split in splits}
    available_unfuzzed_count_total: Optional[int] = None
    if program_names:
        selected_candidates = _resolve_pcp_program_candidates(
            program_names,
            splits,
            output_dir_base=output_dir_base,
        )
        if n != len(selected_candidates):
            logger.info(
                f"Ignoring n={n}; fuzzing {len(selected_candidates)} explicitly "
                f"specified program(s)."
            )
    else:
        candidates, skipped_by_split = _collect_pcp_category_fuzz_candidates(
            splits,
            category_value,
            skip_already_fuzzed=skip_already_fuzzed,
            already_fuzzed_invalid_dirs=already_fuzzed_invalid_dirs,
            output_dir_base=output_dir_base,
        )
        available_unfuzzed_count_total = len(candidates)

        rng = random.Random(seed)
        rng.shuffle(candidates)
        selected_candidates = candidates[:n]
        if len(selected_candidates) < n:
            logger.warning(
                f"Requested {n} programs but only {len(selected_candidates)} unfuzzed "
                f"candidates are available across all PCP splits."
            )

    split_successes: Dict[str, List[FuzzByCategorySuccess]] = {
        split.name: [] for split in splits
    }
    split_failures: Dict[str, List[FuzzByCategoryFailure]] = {
        split.name: [] for split in splits
    }
    selected_by_split: Dict[str, int] = {split.name: 0 for split in splits}

    for idx, candidate in enumerate(
        tqdm(
            selected_candidates,
            desc=f"Fuzzing {category_value} (all PCP splits)",
        )
    ):
        selected_by_split[candidate.split.name] += 1
        su.io.mkdir(candidate.output_dir, parents=True)
        valid_filename = candidate.file_path.name
        imp_program = _clean_imp_program_lines(
            su.io.load(candidate.file_path, fmt=su.io.Fmt.txt)
        )
        (
            fuzzed_program,
            winning_seed,
            target_site_ids,
            attempt_errors,
            failure_reason,
        ) = _fuzz_and_validate_program_with_category(
            imp_lang,
            imp_program,
            fuzzing_category,
            k_framework,
            k_validate_lang,
            base_seed=seed + idx * max_attempts_per_program,
            max_attempts=max_attempts_per_program,
        )

        if fuzzed_program is None or winning_seed is None or target_site_ids is None:
            split_failures[candidate.split.name].append(
                FuzzByCategoryFailure(
                    valid_src_filename=valid_filename,
                    category=category_value,
                    reason=failure_reason or "Fuzzing and K validation failed.",
                    attempts=max_attempts_per_program,
                    attempt_errors=attempt_errors,
                )
            )
            continue

        invalid_filename = _invalid_filename_for_category(
            valid_filename, category_value
        )
        output_path = candidate.output_dir / invalid_filename
        su.io.dump(output_path, fuzzed_program, fmt=su.io.Fmt.txt)
        split_successes[candidate.split.name].append(
            FuzzByCategorySuccess(
                valid_src_filename=valid_filename,
                invalid_src_filename=invalid_filename,
                category=category_value,
                seed=winning_seed,
                target_site_ids=target_site_ids,
                output_path=str(output_path),
            )
        )

    split_results: List[FuzzByCategoryResult] = []
    combined_report_records: List[Dict[str, Any]] = []
    for split in splits:
        split_output_dir = (
            output_dir_base / split.name
            if output_dir_base is not None
            else split.output_dir
        )
        split_report_path = combined_report_path.with_name(
            f"{combined_report_path.stem}-{split.name}{combined_report_path.suffix}"
        )
        report_records = {
            "category": category_value,
            "valid-dir": str(split.valid_dir),
            "output-dir": str(split_output_dir),
            "requested-count": selected_by_split[split.name],
            "selected-count": selected_by_split[split.name],
            "success-count": len(split_successes[split.name]),
            "failure-count": len(split_failures[split.name]),
            "skipped-already-fuzzed-count": skipped_by_split[split.name],
            "skip-already-fuzzed": skip_already_fuzzed,
            "explicit-programs": list(program_names) if program_names else [],
            "already-fuzzed-invalid-dirs": [
                str(path) for path in already_fuzzed_invalid_dirs
            ],
            "successes": [asdict(item) for item in split_successes[split.name]],
            "failures": [asdict(item) for item in split_failures[split.name]],
        }
        su.io.dump(split_report_path, [report_records])
        summary_path = split_report_path.with_suffix(".summary.json")
        su.io.dump(summary_path, report_records, fmt=su.io.Fmt.jsonNoSort)

        split_result = FuzzByCategoryResult(
            category=category_value,
            requested_count=selected_by_split[split.name],
            success_count=len(split_successes[split.name]),
            failure_count=len(split_failures[split.name]),
            skipped_already_fuzzed_count=skipped_by_split[split.name],
            successes=split_successes[split.name],
            failures=split_failures[split.name],
            report_path=split_report_path,
            valid_split=split.name,
        )
        split_results.append(split_result)
        combined_report_records.append(
            {
                "split": split.name,
                "valid-dir": str(split.valid_dir),
                "output-dir": str(split_output_dir),
                "report-path": str(split_report_path),
                "requested-count": split_result.requested_count,
                "selected-count": selected_by_split[split.name],
                "success-count": split_result.success_count,
                "failure-count": split_result.failure_count,
                "skipped-already-fuzzed-count": split_result.skipped_already_fuzzed_count,
            }
        )
        if split_result.requested_count:
            _print_fuzz_by_category_summary(split_result)
            print(f"  Summary: {summary_path}")

    su.io.dump(combined_report_path, combined_report_records)
    combined_summary_path = combined_report_path.with_suffix(".summary.json")
    su.io.dump(
        combined_summary_path,
        {
            "category": category_value,
            "requested-count-total": len(selected_candidates),
            "selected-count-total": len(selected_candidates),
            "available-unfuzzed-count-total": available_unfuzzed_count_total,
            "explicit-programs": list(program_names) if program_names else [],
            "splits": combined_report_records,
            "total-success-count": sum(
                record["success-count"] for record in combined_report_records
            ),
            "total-failure-count": sum(
                record["failure-count"] for record in combined_report_records
            ),
            "total-skipped-already-fuzzed-count": sum(
                record["skipped-already-fuzzed-count"]
                for record in combined_report_records
            ),
        },
        fmt=su.io.Fmt.jsonNoSort,
    )

    all_splits_result = FuzzByCategoryAllSplitsResult(
        category=category_value,
        requested_count_total=n,
        split_results=split_results,
        report_path=combined_report_path,
    )
    _print_fuzz_by_category_all_splits_summary(all_splits_result)
    print(f"  Combined summary: {combined_summary_path}")
    return all_splits_result


def _clean_imp_program_lines(imp_program: str) -> str:
    cleaned_lines = []
    for line in imp_program.splitlines():
        stripped_line = line.strip()
        if not cleaned_lines and not stripped_line:
            continue
        if stripped_line.startswith("//"):
            continue
        if "//" in line:
            cleaned_lines.append(line.split("//")[0].rstrip())
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines) + "\n"


def _fallback_fuzz_program(
    imp_program: str,
    fuzzing_type: IMPFuzzerVisitor.FUZZING_TYPES,
) -> str:
    """
    Deterministic semantic-invalid injection when the ANTLR fuzzer visitor fails.
    """
    lines = [line for line in imp_program.splitlines() if line.strip()]
    if fuzzing_type == IMPFuzzerVisitor.FUZZING_TYPES.CONTINUE_OUTSIDE_LOOP:
        return "continue;\n" + imp_program
    if fuzzing_type == IMPFuzzerVisitor.FUZZING_TYPES.BREAK_OUTSIDE_LOOP:
        return "break;\n" + imp_program
    if fuzzing_type == IMPFuzzerVisitor.FUZZING_TYPES.VAR_USE_BEFORE_DECLARE:
        first_stmt = 0
        for idx, line in enumerate(lines):
            if line.strip().startswith("int "):
                continue
            first_stmt = idx
            break
        lines.insert(first_stmt, "ans = (z + 1);")
        return "\n".join(lines) + "\n"
    if fuzzing_type in (
        IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO,
        IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO,
    ):
        op = (
            "/"
            if fuzzing_type == IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO
            else "%"
        )
        for idx, line in enumerate(lines):
            if "=" in line and not line.strip().startswith("int "):
                lines[idx] = line.split("=", 1)[0] + f"= (0 {op} 1);"
                return "\n".join(lines) + "\n"
        lines.append(f"ans = (1 {op} 0);")
        return "\n".join(lines) + "\n"
    return imp_program


def _fuzz_program_to_semantically_invalid(
    imp_lang: IMP,
    imp_program: str,
    *,
    seed: int,
) -> FuzzProgramOutcome:
    """
    Try each semantic-error fuzzing rule until one succeeds for this program.
    Records whether the ANTLR visitor or the text fallback was used.
    """
    imp_program_norm = imp_lang.translate_unmutated_program(imp_program)
    fuzzing_types = list(IMPFuzzerVisitor.FUZZING_TYPES)
    rng = random.Random(seed)
    rng.shuffle(fuzzing_types)
    visitor_failures: List[Dict[str, str]] = []
    last_err: Optional[Exception] = None
    for fuzzing_type in fuzzing_types:
        try:
            random.seed(rng.randint(0, 2**31 - 1))
            input_stream = InputStream(imp_program_norm)
            lexer = IMPBASELexer(input_stream)
            token_stream = CommonTokenStream(lexer)
            parser = IMPBASEParser(token_stream)
            tree = parser.program()
            fuzzer_visitor = IMPFuzzerVisitor(fuzzing_types=[fuzzing_type])
            fuzzed_program = fuzzer_visitor.visit(tree)
            selected_type = fuzzer_visitor.get_selected_fuzzing_type()
            return FuzzProgramOutcome(
                valid_src_filename="",
                invalid_src_filename=None,
                fuzzing_type=selected_type.value,
                fuzz_method="visitor",
                visitor_failures=visitor_failures,
                fuzzed_program=fuzzed_program,
            )
        except Exception as err:
            visitor_failures.append(
                {"fuzzing-type": fuzzing_type.value, "error": str(err)}
            )
            last_err = err
    fallback_failures: List[Dict[str, str]] = []
    for fuzzing_type in fuzzing_types:
        try:
            fuzzed_program = _fallback_fuzz_program(imp_program_norm, fuzzing_type)
            return FuzzProgramOutcome(
                valid_src_filename="",
                invalid_src_filename=None,
                fuzzing_type=fuzzing_type.value,
                fuzz_method="fallback",
                visitor_failures=visitor_failures,
                fuzzed_program=fuzzed_program,
            )
        except Exception as err:
            fallback_failures.append(
                {"fuzzing-type": fuzzing_type.value, "error": str(err)}
            )
            last_err = err
    error_msg = str(last_err) if last_err is not None else "No fuzzing types available."
    return FuzzProgramOutcome(
        valid_src_filename="",
        fuzz_method="failed",
        visitor_failures=visitor_failures,
        error=error_msg,
        fallback_failures=fallback_failures,
    )


def _default_fuzz_report_path(imp_data_dir: Path, output_dir: Path) -> Path:
    PCP_FUZZ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return PCP_FUZZ_REPORTS_DIR / f"pcp-fuzz-report-{imp_data_dir.name}.jsonl"


def _outcome_to_report_record(
    outcome: FuzzProgramOutcome,
    *,
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    record = asdict(outcome)
    record["input-dir"] = str(input_dir)
    record["output-dir"] = str(output_dir)
    record.pop("fuzzed_program", None)
    return record


def _print_fuzz_batch_summary(batch: FuzzBatchResult, *, input_dir: Path) -> None:
    total = batch.success_count + batch.failure_count
    print(f"\nFuzz summary for {input_dir}:")
    print(f"  Total programs: {total}")
    print(f"  Success: {batch.success_count}")
    print(f"  Failed: {batch.failure_count}")
    print(f"  Visitor fuzzer: {batch.visitor_count}")
    print(f"  Fallback fuzzer: {batch.fallback_count}")
    if batch.report_path is not None:
        print(f"  Report: {batch.report_path}")
    if batch.fallback_count:
        fallback_files = [
            o.valid_src_filename for o in batch.outcomes if o.fuzz_method == "fallback"
        ]
        print("  Fallback files:")
        for name in fallback_files:
            print(f"    - {name}")


def _fuzz_imp_programs_in_dir(
    imp_data_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    report_path: Optional[Path] = None,
) -> FuzzBatchResult:
    """
    Generate one semantically invalid variant per valid program in imp_data_dir.
    Writes a JSONL report describing visitor vs fallback usage per file.
    """
    imp_lang = IMP("IMP_UNMUTATED", Language.SEMANTICS_TYPE.K)
    su.io.mkdir(output_dir, parents=True)
    if report_path is None:
        report_path = _default_fuzz_report_path(imp_data_dir, output_dir)

    success_cnt = 0
    failure_cnt = 0
    visitor_cnt = 0
    fallback_cnt = 0
    outcomes: List[FuzzProgramOutcome] = []
    program_files = sorted(imp_data_dir.glob("*.imp"))
    report_records: List[Dict[str, Any]] = []

    for idx, file_path in enumerate(
        tqdm(program_files, desc=f"Fuzzing {imp_data_dir.name}")
    ):
        file_name_only = file_path.name
        imp_program = _clean_imp_program_lines(su.io.load(file_path, fmt=su.io.Fmt.txt))
        outcome = _fuzz_program_to_semantically_invalid(
            imp_lang,
            imp_program,
            seed=seed + idx,
        )
        outcome.valid_src_filename = file_name_only

        if outcome.fuzz_method == "failed" or outcome.fuzzed_program is None:
            logger.warning(
                f"Failed to fuzz {file_name_only}: {outcome.error or 'unknown error'}"
            )
            failure_cnt += 1
            outcomes.append(outcome)
            report_records.append(
                _outcome_to_report_record(
                    outcome, input_dir=imp_data_dir, output_dir=output_dir
                )
            )
            continue

        fuzzed_file_name = (
            f"{file_name_only.rsplit('.', 1)[0]}_{outcome.fuzzing_type}.imp"
        )
        outcome.invalid_src_filename = fuzzed_file_name
        su.io.dump(
            output_dir / fuzzed_file_name, outcome.fuzzed_program, fmt=su.io.Fmt.txt
        )
        success_cnt += 1
        if outcome.fuzz_method == "visitor":
            visitor_cnt += 1
        elif outcome.fuzz_method == "fallback":
            fallback_cnt += 1
            logger.warning(
                f"Used fallback fuzzer for {file_name_only} "
                f"({outcome.fuzzing_type}); see {report_path}"
            )
        outcomes.append(outcome)
        report_records.append(
            _outcome_to_report_record(
                outcome, input_dir=imp_data_dir, output_dir=output_dir
            )
        )

    summary = {
        "input-dir": str(imp_data_dir),
        "output-dir": str(output_dir),
        "total-programs": len(program_files),
        "success-count": success_cnt,
        "failure-count": failure_cnt,
        "visitor-count": visitor_cnt,
        "fallback-count": fallback_cnt,
        "fallback-files": [
            o.valid_src_filename for o in outcomes if o.fuzz_method == "fallback"
        ],
        "failed-files": [
            o.valid_src_filename for o in outcomes if o.fuzz_method == "failed"
        ],
        "fuzzing-type-counts": dict(
            Counter(o.fuzzing_type for o in outcomes if o.fuzzing_type)
        ),
        "fuzz-method-counts": dict(Counter(o.fuzz_method for o in outcomes)),
    }
    su.io.dump(report_path, report_records)
    summary_path = report_path.with_suffix(".summary.json")
    su.io.dump(summary_path, summary, fmt=su.io.Fmt.jsonNoSort)

    batch = FuzzBatchResult(
        success_count=success_cnt,
        failure_count=failure_cnt,
        visitor_count=visitor_cnt,
        fallback_count=fallback_cnt,
        outcomes=outcomes,
        report_path=report_path,
    )
    _print_fuzz_batch_summary(batch, input_dir=imp_data_dir)
    print(f"  Summary: {summary_path}")
    return batch


def _parse_fuzzing_category(category: str) -> IMPFuzzerVisitor.FUZZING_TYPES:
    normalized = category.strip().lower()
    for fuzzing_type in IMPFuzzerVisitor.FUZZING_TYPES:
        if fuzzing_type.value == normalized:
            return fuzzing_type
    valid = ", ".join(ft.value for ft in IMPFuzzerVisitor.FUZZING_TYPES)
    raise ValueError(f"Unknown fuzzing category {category!r}. Expected one of: {valid}")


def _invalid_filename_for_category(valid_filename: str, category: str) -> str:
    stem = valid_filename.rsplit(".", 1)[0]
    return f"{stem}_{category}.imp"


def _collect_valid_files_already_fuzzed_for_category(
    invalid_dirs: List[Path],
    category: str,
) -> set[str]:
    """
    Return valid basenames (e.g. addition.imp) that already have an invalid
    variant for the given category in any of invalid_dirs.
    """
    used: set[str] = set()
    suffix = f"_{category}.imp"
    for invalid_dir in invalid_dirs:
        if not invalid_dir.exists():
            continue
        for file_path in invalid_dir.glob("*.imp"):
            if file_path.name.endswith(suffix):
                valid_name = file_path.name[: -len(suffix)] + ".imp"
                used.add(valid_name)
    return used


def _get_pcp_fuzz_k_runtime() -> tuple[KFramework, IMP]:
    global _PCP_FUZZ_K_FRAMEWORK, _PCP_FUZZ_IMP_LANG
    if _PCP_FUZZ_K_FRAMEWORK is None or _PCP_FUZZ_IMP_LANG is None:
        if not Macros.tmp_dir.exists():
            Macros.tmp_dir.mkdir(parents=True)
        imp_lang = IMP("IMP_PCP_FUZZ", Language.SEMANTICS_TYPE.K)
        imp_lang.build_visitors()
        k_spec_name = "IMP_PCP_FUZZ.k"
        write_to_tmp(imp_lang.get_semantics(), k_spec_name)
        k_framework = KFramework()
        k_framework.compile_k_specification(
            k_file=str(Macros.tmp_dir / k_spec_name),
            output_dir=str(Macros.tmp_dir / "IMP_PCP_FUZZ"),
        )
        _PCP_FUZZ_K_FRAMEWORK = k_framework
        _PCP_FUZZ_IMP_LANG = imp_lang
    return _PCP_FUZZ_K_FRAMEWORK, _PCP_FUZZ_IMP_LANG


def _attach_pcp_fuzz_k_runtime() -> tuple[KFramework, IMP]:
    """
    Attach to an already-kompiled IMP_PCP_FUZZ definition without recompiling.

    Intended for worker processes after the parent has called _get_pcp_fuzz_k_runtime().
    """
    output_dir = Macros.tmp_dir / "IMP_PCP_FUZZ"
    if not (output_dir / "interpreter").is_file():
        return _get_pcp_fuzz_k_runtime()
    k_framework = KFramework()
    k_framework.language_file = str(output_dir)
    imp_lang = IMP("IMP_PCP_FUZZ", Language.SEMANTICS_TYPE.K)
    return k_framework, imp_lang


def _k_output_triggered_rule(k_output: str, rule: str) -> bool:
    """
    Return True when rule appears in K's recorded execution trace.
    """
    return bool(re.search(rf'ListItem\s*\(\s*"{re.escape(rule)}"\s*\)', k_output))


def _k_output_triggered_semantic_error(
    k_output: str,
    *,
    expected_error_rule: Optional[str] = None,
) -> Optional[str]:
    """
    Return the semantic error rule observed in K output, if any.

    IMP's K semantics swallow `{ERROR}` and continue execution, so the final
    `<k>` cell can still be `.K` even after a semantic error. The reliable
    signal is whether an error rule such as `Rule 7` appears in `<rules>`.
    """
    rules_to_check = (
        [expected_error_rule]
        if expected_error_rule is not None
        else list(K_SEMANTIC_ERROR_RULES)
    )
    for rule in rules_to_check:
        if rule is not None and _k_output_triggered_rule(k_output, rule):
            return rule
    return None


def _list_flow_control_stmt_sites(imp_program_norm: str) -> List[FlowControlSite]:
    tree = _parse_imp_program_tree(imp_program_norm)
    if not tree.stmt_list():
        return []

    sites: List[FlowControlSite] = []

    def walk_stmt_list(stmt_list_ctx, *, depth: int = 0, in_loop: bool = False) -> None:
        for stmt in stmt_list_ctx.stmt():
            if isinstance(stmt, IMPBASEParser.ContinueStmtContext):
                sites.append(
                    FlowControlSite(
                        kind="continue",
                        depth=depth,
                        in_loop=in_loop,
                        line_number=stmt.start.line,
                    )
                )
            elif isinstance(stmt, IMPBASEParser.BreakStmtContext):
                sites.append(
                    FlowControlSite(
                        kind="break",
                        depth=depth,
                        in_loop=in_loop,
                        line_number=stmt.start.line,
                    )
                )
            elif isinstance(stmt, IMPBASEParser.WhileStmtContext):
                walk_stmt_list(stmt.stmt_list(), depth=depth + 1, in_loop=True)
            elif isinstance(stmt, IMPBASEParser.IfElseStmtContext):
                walk_stmt_list(stmt.stmt_list(0), depth=depth + 1, in_loop=in_loop)
                walk_stmt_list(stmt.stmt_list(1), depth=depth + 1, in_loop=in_loop)
            elif isinstance(stmt, IMPBASEParser.IfStmtContext):
                walk_stmt_list(stmt.stmt_list(), depth=depth + 1, in_loop=in_loop)

    walk_stmt_list(tree.stmt_list())
    return sites


def _parse_k_executed_lines(k_output: str) -> set[int]:
    match = re.search(r"<lines>\s*(.*?)\s*</lines>", k_output, re.S)
    if not match:
        return set()
    return {
        int(line_no)
        for line_no in re.findall(r"ListItem\s*\(\s*(\d+)\s*\)", match.group(1))
    }


def _is_flow_control_error_reachable_under_k(
    program: str,
    k_framework: KFramework,
    *,
    kind: str,
    program_file: Optional[Path] = None,
) -> tuple[bool, Optional[str]]:
    """
    Return True when K executes a break/continue outside any loop.
    """
    outside_lines = {
        site.line_number
        for site in _list_flow_control_stmt_sites(program)
        if site.kind == kind and not site.in_loop
    }
    if not outside_lines:
        return False, f"No outside-loop {kind} statement found in program."

    if program_file is None:
        program_file = Macros.tmp_dir / "_validate_flow_reachability.imp"
        write_to_tmp(program, program_file.name)

    try:
        output = k_framework.run_program(program_file=str(program_file))
    except KFrameworkError as err:
        return False, f"Unexpected validation error: {err}"
    except Exception as err:
        return False, f"Unexpected validation error: {err}"

    executed_lines = _parse_k_executed_lines(output)
    if outside_lines & executed_lines:
        return True, None
    return (
        False,
        f"K execution never reached an outside-loop {kind} statement "
        f"(candidate lines: {sorted(outside_lines)}).",
    )


def is_semantic_error_reachable_under_k(
    program: str,
    k_framework: KFramework,
    imp_lang: IMP,
    *,
    semantic_error_type: Optional[str] = None,
    expected_error_rule: Optional[str] = None,
    program_file: Optional[Path] = None,
) -> tuple[bool, Optional[str]]:
    if semantic_error_type == "break_outside_loop":
        return _is_flow_control_error_reachable_under_k(
            program, k_framework, kind="break", program_file=program_file
        )
    if semantic_error_type == "continue_outside_loop":
        return _is_flow_control_error_reachable_under_k(
            program, k_framework, kind="continue", program_file=program_file
        )
    return validate_fuzz_outcome(
        program,
        k_framework,
        imp_lang,
        expected_error_rule=expected_error_rule,
        program_file=program_file,
    )


def _validate_flow_control_fuzz_outcome(
    fuzzed_program: str,
    normalized_valid_program: str,
    category: IMPFuzzerVisitor.FUZZING_TYPES,
) -> tuple[bool, Optional[str]]:
    """
    Validate break/continue fuzzing without K execution.

    K Rule 31/34 only fire when break/continue is alone on the K cell, so
    corpus-style injections (declarations, then continue, then more code) never
    record those rules even though SOS treats them as semantic errors. Instead
    we require that the fuzzer added a new break/continue outside any loop.
    """
    if category == IMPFuzzerVisitor.FUZZING_TYPES.CONTINUE_OUTSIDE_LOOP:
        kind = "continue"
    elif category == IMPFuzzerVisitor.FUZZING_TYPES.BREAK_OUTSIDE_LOOP:
        kind = "break"
    else:
        raise ValueError(
            f"Flow-control validation is not supported for {category.value}."
        )

    if fuzzed_program.strip() == normalized_valid_program.strip():
        return False, "Fuzzed program is unchanged from the valid program."

    fuzzed_sites = _list_flow_control_stmt_sites(fuzzed_program)
    valid_sites = _list_flow_control_stmt_sites(normalized_valid_program)
    fuzzed_outside = [
        site for site in fuzzed_sites if site.kind == kind and not site.in_loop
    ]
    valid_outside = [
        site for site in valid_sites if site.kind == kind and not site.in_loop
    ]

    if not fuzzed_outside:
        return False, f"Fuzzer output has no {kind} statement outside a loop."
    if len(fuzzed_outside) <= len(valid_outside):
        return (
            False,
            f"No new outside-loop {kind} statement was added by the fuzzer.",
        )
    return True, None


def _validate_category_fuzz_outcome(
    fuzzed_program: str,
    category: IMPFuzzerVisitor.FUZZING_TYPES,
    k_framework: KFramework,
    imp_lang: IMP,
    *,
    normalized_valid_program: Optional[str] = None,
    expected_error_rule: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    if category in (
        IMPFuzzerVisitor.FUZZING_TYPES.CONTINUE_OUTSIDE_LOOP,
        IMPFuzzerVisitor.FUZZING_TYPES.BREAK_OUTSIDE_LOOP,
    ):
        if normalized_valid_program is None:
            return False, "Valid program is required for flow-control validation."
        return _validate_flow_control_fuzz_outcome(
            fuzzed_program,
            normalized_valid_program,
            category,
        )
    return validate_fuzz_outcome(
        fuzzed_program,
        k_framework,
        imp_lang,
        normalized_valid_program=normalized_valid_program,
        expected_error_rule=expected_error_rule,
    )


def validate_fuzz_outcome(
    fuzzed_program: str,
    k_framework: KFramework,
    imp_lang: IMP,
    *,
    normalized_valid_program: Optional[str] = None,
    program_file: Optional[Path] = None,
    expected_error_rule: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Return (True, None) when fuzzed_program triggers a semantic error under K.

    Because IMP's K definition consumes `{ERROR}` and keeps executing, validity
    is determined from the semantic error rules recorded in K output (e.g.
    `Rule 7` for divide-by-zero), not from whether the final configuration is
    `.K`.
    """
    if normalized_valid_program is not None:
        if fuzzed_program.strip() == normalized_valid_program.strip():
            return False, "Fuzzed program is unchanged from the valid program."

    if program_file is None:
        program_file = Macros.tmp_dir / "_validate_fuzz_outcome.imp"
        write_to_tmp(fuzzed_program, program_file.name)

    try:
        output = k_framework.run_program(program_file=str(program_file))
        triggered_rule = _k_output_triggered_semantic_error(
            output,
            expected_error_rule=expected_error_rule,
        )
        if triggered_rule is not None:
            return True, None
        if expected_error_rule is not None:
            return (
                False,
                f"K completed without triggering {expected_error_rule}.",
            )
        return False, "K execution completed without any semantic error rule."
    except KFrameworkError as err:
        return True, None
    except Exception as err:
        return False, f"Unexpected validation error: {err}"


def _parse_imp_program_tree(imp_program_norm: str):
    input_stream = InputStream(imp_program_norm)
    lexer = IMPBASELexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = IMPBASEParser(token_stream)
    return parser.program()


def _walk_assign_sites(
    stmt_list_ctx,
    *,
    depth: int = 0,
    in_loop: bool = False,
    in_if: bool = False,
    sites: Optional[List[AssignSite]] = None,
) -> List[AssignSite]:
    if sites is None:
        sites = []
    for stmt in stmt_list_ctx.stmt():
        if isinstance(stmt, IMPBASEParser.AssignStmtContext):
            sites.append(
                AssignSite(
                    site_id=len(sites),
                    depth=depth,
                    in_loop=in_loop,
                    in_if=in_if,
                )
            )
        elif isinstance(stmt, IMPBASEParser.WhileStmtContext):
            _walk_assign_sites(
                stmt.stmt_list(),
                depth=depth + 1,
                in_loop=True,
                in_if=in_if,
                sites=sites,
            )
        elif isinstance(stmt, IMPBASEParser.IfElseStmtContext):
            _walk_assign_sites(
                stmt.stmt_list(0),
                depth=depth + 1,
                in_loop=in_loop,
                in_if=True,
                sites=sites,
            )
            _walk_assign_sites(
                stmt.stmt_list(1),
                depth=depth + 1,
                in_loop=in_loop,
                in_if=True,
                sites=sites,
            )
        elif isinstance(stmt, IMPBASEParser.IfStmtContext):
            _walk_assign_sites(
                stmt.stmt_list(),
                depth=depth + 1,
                in_loop=in_loop,
                in_if=True,
                sites=sites,
            )
    return sites


def _pick_multi_site_combos(
    assign_sites: List[AssignSite],
) -> tuple[List[tuple[int, ...]], List[tuple[int, ...]]]:
    """
    Build 2-site and 3-site target combos mirroring the old bulk fuzzer:
    often one top-level assign plus one nested assign.
    """
    top_level = [site.site_id for site in assign_sites if site.depth == 0]
    nested = [site.site_id for site in assign_sites if site.depth > 0]
    all_ids = [site.site_id for site in assign_sites]
    pairs: List[tuple[int, ...]] = []
    seen_pairs: set[tuple[int, ...]] = set()

    def add_pair(a: int, b: int) -> None:
        if a == b:
            return
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        pairs.append((a, b))

    for idx in range(max(len(top_level), len(nested))):
        combo: List[int] = []
        if idx < len(nested):
            combo.append(nested[idx])
        if idx < len(top_level):
            combo.append(top_level[idx])
        if len(combo) >= 2:
            add_pair(combo[0], combo[1])

    for idx in range(len(all_ids) - 1):
        add_pair(all_ids[idx], all_ids[idx + 1])

    triples: List[tuple[int, ...]] = []
    seen_triples: set[tuple[int, ...]] = set()
    for idx in range(len(all_ids) - 2):
        key = tuple(all_ids[idx : idx + 3])
        if len(set(key)) < 3 or key in seen_triples:
            continue
        seen_triples.add(key)
        triples.append(key)
    return pairs, triples


def _visit_fuzz_with_category(
    imp_program_norm: str,
    category: IMPFuzzerVisitor.FUZZING_TYPES,
    *,
    seed: int,
    target_site_ids: List[int],
) -> str:
    random.seed(seed)
    tree = _parse_imp_program_tree(imp_program_norm)
    assign_categories = (
        IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO,
        IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO,
        IMPFuzzerVisitor.FUZZING_TYPES.VAR_USE_BEFORE_DECLARE,
    )
    if category in assign_categories:
        fuzzer_visitor = IMPFuzzerVisitor(
            num_stmts_to_fuzz=1,
            fuzzing_types=[category],
            target_assign_site_ids=target_site_ids,
            exact_target_stmt=True,
        )
    else:
        if len(target_site_ids) != 1:
            raise ValueError(
                f"Category {category.value} supports exactly one target statement index."
            )
        fuzzer_visitor = IMPFuzzerVisitor(
            num_stmts_to_fuzz=1,
            fuzzing_types=[category],
            target_stmt_index=target_site_ids[0],
            exact_target_stmt=True,
        )
    fuzzed_program = fuzzer_visitor.visit(tree)
    selected_type = fuzzer_visitor.get_selected_fuzzing_type()
    if selected_type != category:
        raise RuntimeError(
            f"Fuzzer selected {selected_type.value}, expected {category.value}."
        )
    return fuzzed_program


def _count_imp_program_stmts(imp_program_norm: str) -> int:
    tree = _parse_imp_program_tree(imp_program_norm)
    if tree.stmt_list():
        return len(tree.stmt_list().stmt())
    return 0


def _list_top_level_assign_stmt_indices(imp_program_norm: str) -> List[int]:
    tree = _parse_imp_program_tree(imp_program_norm)
    if not tree.stmt_list():
        return []
    assign_indices: List[int] = []
    for idx, stmt in enumerate(tree.stmt_list().stmt()):
        if isinstance(stmt, IMPBASEParser.AssignStmtContext):
            assign_indices.append(idx)
    return assign_indices


def _list_top_level_flow_control_injection_indices(
    imp_program_norm: str,
) -> List[int]:
    """
    Top-level statement indices where break/continue can be injected outside loops.

    Skips declaration statements so injections match the human-written corpus style:
    declarations first, then break/continue, then executable code.
    """
    tree = _parse_imp_program_tree(imp_program_norm)
    if not tree.stmt_list():
        return []
    return [
        idx
        for idx, stmt in enumerate(tree.stmt_list().stmt())
        if not isinstance(stmt, IMPBASEParser.DeclStmtContext)
    ]


FLOW_CONTROL_APPEND_TARGET = (-1,)


def _append_flow_control_to_program(
    imp_program_norm: str,
    category: IMPFuzzerVisitor.FUZZING_TYPES,
) -> str:
    if category == IMPFuzzerVisitor.FUZZING_TYPES.CONTINUE_OUTSIDE_LOOP:
        stmt = "continue;"
    elif category == IMPFuzzerVisitor.FUZZING_TYPES.BREAK_OUTSIDE_LOOP:
        stmt = "break;"
    else:
        raise ValueError(f"Flow-control append is not supported for {category.value}.")
    return imp_program_norm.rstrip() + f"\n{stmt}\n"


def _list_assign_stmt_sites(imp_program_norm: str) -> List[AssignSite]:
    tree = _parse_imp_program_tree(imp_program_norm)
    if not tree.stmt_list():
        return []
    return _walk_assign_sites(tree.stmt_list())


def _collect_declared_var_names(imp_program_norm: str) -> List[str]:
    tree = _parse_imp_program_tree(imp_program_norm)
    if not tree.stmt_list():
        return []

    declared: List[str] = []

    def walk_stmt_list(stmt_list_ctx) -> None:
        for stmt in stmt_list_ctx.stmt():
            if isinstance(stmt, IMPBASEParser.DeclStmtContext) and stmt.ids():
                declared.extend(stmt.ids().ID())
            elif isinstance(stmt, IMPBASEParser.WhileStmtContext):
                walk_stmt_list(stmt.stmt_list())
            elif isinstance(stmt, IMPBASEParser.IfElseStmtContext):
                walk_stmt_list(stmt.stmt_list(0))
                walk_stmt_list(stmt.stmt_list(1))
            elif isinstance(stmt, IMPBASEParser.IfStmtContext):
                walk_stmt_list(stmt.stmt_list())

    walk_stmt_list(tree.stmt_list())
    return [var.getText() for var in declared]


def _inject_divide_or_modulo_for_decl_only_program(
    imp_program_norm: str,
    category: IMPFuzzerVisitor.FUZZING_TYPES,
    *,
    seed: int,
) -> str:
    """
    Append a divide/modulo-by-zero assignment to declaration-only programs.
    """
    if category not in (
        IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO,
        IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO,
    ):
        raise ValueError(
            f"Declaration-only injection is not supported for {category.value}."
        )

    op = "/" if category == IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO else "%"
    rng = random.Random(seed)
    declared = _collect_declared_var_names(imp_program_norm)
    if not declared:
        declared = ["n", "ans"]

    assign_lhs = rng.choice(declared)
    zero_candidates = [var for var in declared if var != assign_lhs] or declared
    zero_var = rng.choice(zero_candidates)
    program = imp_program_norm.rstrip() + "\n"
    if rng.random() < 0.3:
        return program + f"{assign_lhs} = (1 {op} 0);\n"
    return program + f"{zero_var} = 0;\n{assign_lhs} = (1 {op} {zero_var});\n"


def _has_zero_division_pattern(program: str, *, op: str) -> bool:
    if op == "/":
        if re.search(r"/\s*0\b", program):
            return True
        var_pattern = r"/\s*([a-z][a-z0-9_]*)\b"
    else:
        if re.search(r"%\s*0\b", program):
            return True
        var_pattern = r"%\s*([a-z][a-z0-9_]*)\b"
    for match in re.finditer(var_pattern, program):
        var = match.group(1)
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(var)}\s*=\s*0\s*;", program):
            return True
    return False


def _choose_desired_min_div_zero_sites(
    base_seed: int,
    category: IMPFuzzerVisitor.FUZZING_TYPES,
) -> int:
    """
    Match the legacy invalid corpus: ~58% of divide/modulo programs have 2+ sites.
    """
    if category not in (
        IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO,
        IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO,
    ):
        return 1
    rng = random.Random(base_seed)
    return 2 if rng.random() < 0.58 else 1


def _interleave_single_and_multi_site_plans(
    single_targets: List[int],
    multi_pairs: List[tuple[int, ...]],
    multi_triples: List[tuple[int, ...]],
) -> List[tuple[int, ...]]:
    """
    Interleave single- and multi-site plans without giving multi-site priority.
    Pattern: single, multi, single, multi, ...
    """
    singles = [(target,) for target in single_targets]
    multis = list(multi_pairs) + list(multi_triples)
    interleaved: List[tuple[int, ...]] = []
    single_idx = 0
    multi_idx = 0
    while single_idx < len(singles) or multi_idx < len(multis):
        if single_idx < len(singles):
            interleaved.append(singles[single_idx])
            single_idx += 1
        if multi_idx < len(multis):
            interleaved.append(multis[multi_idx])
            multi_idx += 1
    return interleaved


def _build_category_fuzz_attempts(
    stmt_count: int,
    assign_sites: List[AssignSite],
    category: IMPFuzzerVisitor.FUZZING_TYPES,
    *,
    base_seed: int,
    max_attempts: int,
    imp_program_norm: Optional[str] = None,
) -> List[tuple[int, tuple[int, ...]]]:
    """
    Build (seed, target_site_ids) plans for category fuzzing.
    Single-site attempts keep the existing nested/top-level ordering; multi-site
    attempts are interleaved alongside singles (not deferred until after them).
    """
    if category in (
        IMPFuzzerVisitor.FUZZING_TYPES.CONTINUE_OUTSIDE_LOOP,
        IMPFuzzerVisitor.FUZZING_TYPES.BREAK_OUTSIDE_LOOP,
    ):
        if imp_program_norm is None:
            raise ValueError(
                "imp_program_norm is required for break/continue category fuzzing."
            )
        injection_indices = _list_top_level_flow_control_injection_indices(
            imp_program_norm
        )
        if injection_indices:
            rng = random.Random(base_seed)
            shuffled = list(injection_indices)
            rng.shuffle(shuffled)
            plan_cycle = [(idx,) for idx in shuffled]
        else:
            plan_cycle = [FLOW_CONTROL_APPEND_TARGET]
    elif category in (
        IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO,
        IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO,
        IMPFuzzerVisitor.FUZZING_TYPES.VAR_USE_BEFORE_DECLARE,
    ):
        top_level = [site.site_id for site in assign_sites if site.depth == 0]
        nested = [site.site_id for site in assign_sites if site.depth > 0]
        single_targets: List[int] = []
        for idx in range(max(len(top_level), len(nested))):
            if idx < len(nested):
                single_targets.append(nested[idx])
            if idx < len(top_level):
                single_targets.append(top_level[idx])
        if not single_targets and assign_sites:
            single_targets = [site.site_id for site in assign_sites]
        multi_pairs, multi_triples = _pick_multi_site_combos(assign_sites)
        plan_cycle = _interleave_single_and_multi_site_plans(
            single_targets,
            multi_pairs,
            multi_triples,
        )
    else:
        plan_cycle = [
            (target,)
            for target in (list(range(min(stmt_count, 5))) if stmt_count > 0 else [0])
        ]

    if not plan_cycle:
        plan_cycle = [(0,)]

    attempts: List[tuple[int, tuple[int, ...]]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    seed_cursor = 0

    def add(seed: int, site_ids: tuple[int, ...]) -> None:
        key = (seed, site_ids)
        if key in seen:
            return
        seen.add(key)
        attempts.append(key)

    seeds_per_plan = 2
    while len(attempts) < max_attempts:
        for site_ids in plan_cycle:
            for _ in range(seeds_per_plan):
                add(base_seed + seed_cursor, site_ids)
                seed_cursor += 1
                if len(attempts) >= max_attempts:
                    break
            if len(attempts) >= max_attempts:
                break

    return attempts[:max_attempts]


def _structural_fuzz_check(
    fuzzed_program: str,
    category: IMPFuzzerVisitor.FUZZING_TYPES,
) -> tuple[bool, Optional[str]]:
    """
    Fast syntactic check that the visitor injected the requested error pattern.
    """
    if category == IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO:
        if _has_zero_division_pattern(fuzzed_program, op="/"):
            return True, None
        return False, (
            "Fuzzer output has no divide-by-zero pattern (/ 0 or / V with V = 0)."
        )
    if category == IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO:
        if _has_zero_division_pattern(fuzzed_program, op="%"):
            return True, None
        return False, (
            "Fuzzer output has no modulo-by-zero pattern (% 0 or % V with V = 0)."
        )
    if category == IMPFuzzerVisitor.FUZZING_TYPES.BREAK_OUTSIDE_LOOP:
        if "break;" not in fuzzed_program:
            return False, "Fuzzer output has no break statement."
        return True, None
    if category == IMPFuzzerVisitor.FUZZING_TYPES.CONTINUE_OUTSIDE_LOOP:
        if "continue;" not in fuzzed_program:
            return False, "Fuzzer output has no continue statement."
        return True, None
    if category == IMPFuzzerVisitor.FUZZING_TYPES.VAR_USE_BEFORE_DECLARE:
        return True, None
    return True, None


def _fuzz_and_validate_program_with_category(
    imp_lang: IMP,
    imp_program: str,
    category: IMPFuzzerVisitor.FUZZING_TYPES,
    k_framework: KFramework,
    k_validate_lang: IMP,
    *,
    base_seed: int,
    max_attempts: int,
) -> tuple[
    Optional[str],
    Optional[int],
    Optional[List[int]],
    List[Dict[str, Any]],
    Optional[str],
]:
    """
    Try visitor-only fuzzing with multiple seeds and injection points, validating
    each candidate with K. Returns fuzzed source and metadata on success.
    """
    imp_program_norm = imp_lang.translate_unmutated_program(imp_program)
    expected_error_rule = DatasetProcessor.SEMANTIC_INVALID_RULES_IMP["K"][
        category.value
    ]
    stmt_count = _count_imp_program_stmts(imp_program_norm)
    assign_sites = _list_assign_stmt_sites(imp_program_norm)
    attempt_errors: List[Dict[str, Any]] = []
    last_reason: Optional[str] = "No fuzz attempts were made."
    fallback_success: Optional[tuple[str, int, List[int]]] = None

    if not assign_sites and category in (
        IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO,
        IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO,
    ):
        for attempt in range(max_attempts):
            attempt_seed = base_seed + attempt
            fuzzed_program = _inject_divide_or_modulo_for_decl_only_program(
                imp_program_norm,
                category,
                seed=attempt_seed,
            )
            struct_ok, struct_reason = _structural_fuzz_check(fuzzed_program, category)
            if not struct_ok:
                attempt_errors.append(
                    {
                        "attempt": attempt,
                        "seed": attempt_seed,
                        "target-site-ids": [],
                        "assign-sites": [],
                        "stage": "decl-only-structural",
                        "error": struct_reason or "Structural fuzz check failed.",
                    }
                )
                last_reason = struct_reason or "Structural fuzz check failed."
                continue

            is_invalid, validation_reason = _validate_category_fuzz_outcome(
                fuzzed_program,
                category,
                k_framework,
                k_validate_lang,
                normalized_valid_program=imp_program_norm,
                expected_error_rule=expected_error_rule,
            )
            if is_invalid:
                return (
                    fuzzed_program,
                    attempt_seed,
                    [],
                    attempt_errors,
                    None,
                )

            attempt_errors.append(
                {
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "target-site-ids": [],
                    "assign-sites": [],
                    "stage": "decl-only-k-validation",
                    "error": validation_reason or "K validation failed.",
                    "fuzzed-program-preview": fuzzed_program[:2000],
                }
            )
            last_reason = validation_reason or "K validation failed."

        return None, None, None, attempt_errors, last_reason

    attempt_plan = _build_category_fuzz_attempts(
        stmt_count,
        assign_sites,
        category,
        base_seed=base_seed,
        max_attempts=max_attempts,
        imp_program_norm=imp_program_norm,
    )
    desired_min_sites = _choose_desired_min_div_zero_sites(base_seed, category)

    for attempt, (attempt_seed, target_site_ids) in enumerate(attempt_plan):
        try:
            if target_site_ids == FLOW_CONTROL_APPEND_TARGET:
                fuzzed_program = _append_flow_control_to_program(
                    imp_program_norm,
                    category,
                )
            else:
                fuzzed_program = _visit_fuzz_with_category(
                    imp_program_norm,
                    category,
                    seed=attempt_seed,
                    target_site_ids=list(target_site_ids),
                )
        except Exception as err:
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "target-site-ids": list(target_site_ids),
                    "assign-sites": [asdict(site) for site in assign_sites],
                    "stage": "visitor",
                    "error": str(err),
                }
            )
            last_reason = f"Visitor failed: {err}"
            continue

        struct_ok, struct_reason = _structural_fuzz_check(fuzzed_program, category)
        if not struct_ok:
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "target-site-ids": list(target_site_ids),
                    "assign-sites": [asdict(site) for site in assign_sites],
                    "stage": "structural",
                    "error": struct_reason or "Structural fuzz check failed.",
                }
            )
            last_reason = struct_reason or "Structural fuzz check failed."
            continue

        is_invalid, validation_reason = _validate_category_fuzz_outcome(
            fuzzed_program,
            category,
            k_framework,
            k_validate_lang,
            normalized_valid_program=imp_program_norm,
            expected_error_rule=expected_error_rule,
        )
        if is_invalid:
            if len(target_site_ids) >= desired_min_sites:
                return (
                    fuzzed_program,
                    attempt_seed,
                    list(target_site_ids),
                    attempt_errors,
                    None,
                )
            if fallback_success is None:
                fallback_success = (
                    fuzzed_program,
                    attempt_seed,
                    list(target_site_ids),
                )
            continue

        attempt_errors.append(
            {
                "attempt": attempt,
                "seed": attempt_seed,
                "target-site-ids": list(target_site_ids),
                "assign-sites": [asdict(site) for site in assign_sites],
                "stage": "k-validation",
                "error": validation_reason or "K validation failed.",
                "fuzzed-program-preview": fuzzed_program[:2000],
            }
        )
        last_reason = validation_reason or "K validation failed."

    if fallback_success is not None:
        fuzzed_program, attempt_seed, target_site_ids = fallback_success
        return (
            fuzzed_program,
            attempt_seed,
            target_site_ids,
            attempt_errors,
            None,
        )

    return None, None, None, attempt_errors, last_reason


def _default_fuzz_by_category_report_path(
    category: str,
    valid_dir: Path,
) -> Path:
    PCP_FUZZ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return (
        PCP_FUZZ_REPORTS_DIR / f"pcp-fuzz-by-category-{category}-{valid_dir.name}.jsonl"
    )


def _default_fuzz_by_category_all_splits_report_path(category: str) -> Path:
    PCP_FUZZ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return (
        PCP_FUZZ_REPORTS_DIR / f"pcp-fuzz-by-category-{category}-all-pcp-splits.jsonl"
    )


def _print_fuzz_by_category_summary(result: FuzzByCategoryResult) -> None:
    print(f"\nFuzz-by-category summary ({result.category}):")
    print(f"  Requested: {result.requested_count}")
    print(f"  Success: {result.success_count}")
    print(f"  Failed: {result.failure_count}")
    print(
        f"  Skipped (already fuzzed for category): {result.skipped_already_fuzzed_count}"
    )
    if result.report_path is not None:
        print(f"  Report: {result.report_path}")
    if result.failures:
        print("  Failed programs:")
        for failure in result.failures:
            print(f"    - {failure.valid_src_filename}: {failure.reason}")

        for failure in result.failures[:10]:
            print(f"    - {failure.valid_src_filename}: {failure.reason}")


def _print_fuzz_by_category_all_splits_summary(
    result: FuzzByCategoryAllSplitsResult,
) -> None:
    print(f"\nFuzz-by-category summary ({result.category}, all PCP splits):")
    print(f"  Requested total: {result.requested_count_total}")
    print(f"  Total success: {result.success_count}")
    print(f"  Total failed: {result.failure_count}")
    print(
        f"  Total skipped (already fuzzed for category): {result.skipped_already_fuzzed_count}"
    )
    for split_result in result.split_results:
        split_label = split_result.valid_split or "unknown"
        print(
            f"  [{split_label}]: "
            f"success={split_result.success_count}, "
            f"failed={split_result.failure_count}, "
            f"skipped={split_result.skipped_already_fuzzed_count}"
        )


def _resolve_fuzz_by_category_output_dir(
    valid_dir: Path,
    output_dir: Optional[Path],
) -> Path:
    if output_dir is not None:
        return output_dir
    return Macros.invalid_imp_uk_dir / valid_dir.name


def _run_fuzz_by_category_for_valid_dir(
    *,
    category_value: str,
    fuzzing_category: IMPFuzzerVisitor.FUZZING_TYPES,
    valid_dir: Path,
    output_dir: Path,
    n: int,
    seed: int,
    skip_already_fuzzed: bool,
    already_fuzzed_invalid_dirs: List[Path],
    max_attempts_per_program: int,
    report_path: Path,
    imp_lang: IMP,
    k_framework: KFramework,
    k_validate_lang: IMP,
    program_names: Optional[List[str]] = None,
) -> FuzzByCategoryResult:
    su.io.mkdir(output_dir, parents=True)

    all_valid_files = sorted(valid_dir.glob("*.imp"))
    skipped_already_fuzzed: set[str] = set()
    if skip_already_fuzzed and not program_names:
        skipped_already_fuzzed = _collect_valid_files_already_fuzzed_for_category(
            already_fuzzed_invalid_dirs,
            category_value,
        )

    if program_names:
        selected_files = _resolve_program_files_in_valid_dir(program_names, valid_dir)
        if n != len(selected_files):
            logger.info(
                f"Ignoring n={n}; fuzzing {len(selected_files)} explicitly "
                f"specified program(s)."
            )
    else:
        candidate_files = [
            file_path
            for file_path in all_valid_files
            if file_path.name not in skipped_already_fuzzed
        ]
        rng = random.Random(seed)
        rng.shuffle(candidate_files)
        selected_files = candidate_files[:n]

        if len(selected_files) < n:
            logger.warning(
                f"Requested {n} programs but only {len(selected_files)} candidates are "
                f"available in {valid_dir} after filtering."
            )

    successes: List[FuzzByCategorySuccess] = []
    failures: List[FuzzByCategoryFailure] = []

    for idx, file_path in enumerate(
        tqdm(selected_files, desc=f"Fuzzing {category_value} ({valid_dir.name})")
    ):
        valid_filename = file_path.name
        imp_program = _clean_imp_program_lines(su.io.load(file_path, fmt=su.io.Fmt.txt))
        (
            fuzzed_program,
            winning_seed,
            target_site_ids,
            attempt_errors,
            failure_reason,
        ) = _fuzz_and_validate_program_with_category(
            imp_lang,
            imp_program,
            fuzzing_category,
            k_framework,
            k_validate_lang,
            base_seed=seed + idx * max_attempts_per_program,
            max_attempts=max_attempts_per_program,
        )

        if fuzzed_program is None or winning_seed is None or target_site_ids is None:
            failures.append(
                FuzzByCategoryFailure(
                    valid_src_filename=valid_filename,
                    category=category_value,
                    reason=failure_reason or "Fuzzing and K validation failed.",
                    attempts=max_attempts_per_program,
                    attempt_errors=attempt_errors,
                )
            )
            continue

        invalid_filename = _invalid_filename_for_category(
            valid_filename, category_value
        )
        output_path = output_dir / invalid_filename
        su.io.dump(output_path, fuzzed_program, fmt=su.io.Fmt.txt)
        successes.append(
            FuzzByCategorySuccess(
                valid_src_filename=valid_filename,
                invalid_src_filename=invalid_filename,
                category=category_value,
                seed=winning_seed,
                target_site_ids=target_site_ids,
                output_path=str(output_path),
            )
        )

    report_records = {
        "category": category_value,
        "valid-dir": str(valid_dir),
        "output-dir": str(output_dir),
        "requested-count": len(selected_files) if program_names else n,
        "selected-count": len(selected_files),
        "success-count": len(successes),
        "failure-count": len(failures),
        "skipped-already-fuzzed-count": len(skipped_already_fuzzed),
        "skip-already-fuzzed": skip_already_fuzzed,
        "explicit-programs": list(program_names) if program_names else [],
        "already-fuzzed-invalid-dirs": [
            str(path) for path in already_fuzzed_invalid_dirs
        ],
        "successes": [asdict(item) for item in successes],
        "failures": [asdict(item) for item in failures],
    }
    su.io.dump(report_path, [report_records])
    summary_path = report_path.with_suffix(".summary.json")
    su.io.dump(summary_path, report_records, fmt=su.io.Fmt.jsonNoSort)

    result = FuzzByCategoryResult(
        category=category_value,
        requested_count=n,
        success_count=len(successes),
        failure_count=len(failures),
        skipped_already_fuzzed_count=len(skipped_already_fuzzed),
        successes=successes,
        failures=failures,
        report_path=report_path,
        valid_split=valid_dir.name,
    )
    _print_fuzz_by_category_summary(result)
    print(f"  Summary: {summary_path}")
    return result


@subcommand(category=Category.DATA)
def fuzz_by_category(
    category: str,
    n: int,
    valid_dir: Path = Macros.valid_imp_uk_dir,
    output_dir: Optional[Path] = None,
    seed: int = 42,
    skip_already_fuzzed: bool = False,
    all_pcp_splits: bool = False,
    programs: Optional[Union[str, List[str]]] = None,
    already_fuzzed_invalid_dirs: Optional[List[Path]] = None,
    max_attempts_per_program: int = 60,
    report_path: Optional[Path] = None,
):
    """
    Fuzz up to n valid IMP programs with a specific semantic-error category.

    Uses only the ANTLR fuzzer visitor (multiple seeds and injection points).
    Each candidate is validated by checking that K execution fails. Programs
    that cannot be fuzzed and validated are recorded in the failure list and
    are not written to output_dir.

    When skip_already_fuzzed is True, valid programs that already have an
    invalid variant named {stem}_{category}.imp in already_fuzzed_invalid_dirs
    are excluded from the candidate pool.

    When programs is set, fuzz exactly those valid filenames (with or without
    the .imp suffix), or pass a path relative to the current working directory
    such as data/imp/valid_imp_programs/synthetic_cpp/pgm_357.imp. In that
    mode n is ignored, skip_already_fuzzed does not filter the selection, and
    each file is resolved across all three PCP valid corpora unless an explicit
    path is provided. valid_dir is ignored in that mode.

    When all_pcp_splits is True (and programs is not set), pools unfuzzed valid
    programs from all three PCP corpora and fuzzes up to n of them in total.
    valid_dir is ignored in that mode.
    """
    if n <= 0 and not programs:
        raise ValueError("n must be a positive integer.")

    fuzzing_category = _parse_fuzzing_category(category)
    category_value = fuzzing_category.value
    program_names = _normalize_fuzz_programs_arg(programs)

    if already_fuzzed_invalid_dirs is None:
        already_fuzzed_invalid_dirs = list(DEFAULT_ALREADY_FUZZED_INVALID_DIRS)

    imp_lang = IMP("IMP_UNMUTATED", Language.SEMANTICS_TYPE.K)
    k_framework, k_validate_lang = _get_pcp_fuzz_k_runtime()

    if all_pcp_splits or program_names:
        if report_path is None:
            combined_report_path = _default_fuzz_by_category_all_splits_report_path(
                category_value
            )
        else:
            combined_report_path = report_path

        return _run_fuzz_by_category_all_pcp_splits(
            category_value=category_value,
            fuzzing_category=fuzzing_category,
            n=n,
            seed=seed,
            skip_already_fuzzed=skip_already_fuzzed,
            already_fuzzed_invalid_dirs=already_fuzzed_invalid_dirs,
            max_attempts_per_program=max_attempts_per_program,
            output_dir_base=output_dir,
            combined_report_path=combined_report_path,
            imp_lang=imp_lang,
            k_framework=k_framework,
            k_validate_lang=k_validate_lang,
            program_names=program_names,
        )

    resolved_output_dir = _resolve_fuzz_by_category_output_dir(valid_dir, output_dir)
    if report_path is None:
        report_path = _default_fuzz_by_category_report_path(category_value, valid_dir)

    return _run_fuzz_by_category_for_valid_dir(
        category_value=category_value,
        fuzzing_category=fuzzing_category,
        valid_dir=valid_dir,
        output_dir=resolved_output_dir,
        n=n,
        seed=seed,
        skip_already_fuzzed=skip_already_fuzzed,
        already_fuzzed_invalid_dirs=already_fuzzed_invalid_dirs,
        max_attempts_per_program=max_attempts_per_program,
        report_path=report_path,
        imp_lang=imp_lang,
        k_framework=k_framework,
        k_validate_lang=k_validate_lang,
        program_names=program_names,
    )


@subcommand(category=Category.DATA)
def clean_imp_seed_programs(
    imp_data_dir: Path = Macros.data_dir / "imp/valid_imp_programs/human_written/",
):
    """
    Clean the IMP seed programs:
    1. Remove the inline comments
    2. Remove the potential empty line at the beginning of file
    """
    # imp_data_dir = Macros.data_dir / "imp/valid_imp_programs/human_written/"
    # imp_data_dir = Macros.data_dir / "imp/synthetic/cpp/"
    invalid_dir = imp_data_dir / "invalid"
    su.io.mkdir(invalid_dir, parents=True, fresh=True)
    imp_lang = IMP("IMP_UNMUTATED", Language.SEMANTICS_TYPE.K)
    valid_cnt = 0
    invalid_cnt = 0

    for file_name in imp_data_dir.glob("*.imp"):
        file_path = file_name
        base_name = file_name.stem
        imp_program = su.io.load(file_path, fmt=su.io.Fmt.txt)
        cleaned_lines = []
        for line in imp_program.splitlines():
            stripped_line = line.strip()
            line = line.replace("_", "")
            if not cleaned_lines and not stripped_line:
                continue  # Skip leading empty lines
            if stripped_line.startswith("//"):
                continue  # Skip full-line comments
            elif "//" in line:
                cleaned_lines.append(line.split("//")[0].rstrip())
            else:
                cleaned_lines.append(line)
        imp_program = "\n".join(cleaned_lines) + "\n"
        try:
            imp_program = imp_lang.translate_unmutated_program(imp_program)
            valid_cnt += 1
        except Exception:
            logger.error(f"Error translating {file_name}")
            invalid_cnt += 1
            # move this file to the invalid directory
            su.io.dump(f"{invalid_dir / base_name}.imp", imp_program, fmt=su.io.Fmt.txt)
            su.io.rm(file_path)
            continue
        su.io.dump(file_path, imp_program, fmt=su.io.Fmt.txt)

    print(f"Valid programs: {valid_cnt}, Invalid programs: {invalid_cnt}")


@subcommand(category=Category.DATA)
def fuzz_imp_seed_programs_to_semantically_invalid(
    imp_data_dir: Path = Macros.data_dir / "imp/valid_imp_programs/human_written/",
    output_dir: Path = Macros.data_dir / "imp/invalid_imp_programs",
    seed: int = 42,
    report_path: Optional[Path] = None,
):
    """
    Fuzz IMP seed programs to generate semantically invalid IMP programs.
    Writes a per-file JSONL report under data/imp/pcp_fuzz_reports/ by default.
    """
    _fuzz_imp_programs_in_dir(
        imp_data_dir=imp_data_dir,
        output_dir=output_dir,
        seed=seed,
        report_path=report_path,
    )


@subcommand(category=Category.DATA)
def merge_pcp_valid_and_invalid_programs(
    target_dir: Path = Macros.data_dir / "imp/valid_and_invalid_programs",
):
    """
    Rebuild the PCP program pool from human-written, fuzzer-generated, and
    LLM-translated (synthetic_cpp) valid/invalid corpora.
    """
    imp_root = Macros.data_dir / "imp"
    source_dirs = [
        imp_root / "valid_imp_programs/human_written",
        imp_root / "invalid_imp_programs",
        imp_root / "fuzzer_generated",
        imp_root / "invalid_imp_programs/fuzzer_generated",
        imp_root / "valid_imp_programs/synthetic_cpp",
        imp_root / "invalid_imp_programs/synthetic_cpp",
    ]
    for source_dir in source_dirs:
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing PCP source directory: {source_dir}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    su.io.mkdir(target_dir, parents=True)

    seen_names: set[str] = set()
    for source_dir in source_dirs:
        for file_path in sorted(source_dir.glob("*.imp")):
            if file_path.name in seen_names:
                raise ValueError(
                    f"Duplicate program filename {file_path.name} from {source_dir}"
                )
            seen_names.add(file_path.name)
            shutil.copy2(file_path, target_dir / file_path.name)

    valid_cnt = sum(
        1
        for name in seen_names
        if not any(
            name.endswith(f"_{suffix}.imp") for suffix in PCP_SEMANTIC_ERROR_SUFFIXES
        )
    )
    invalid_cnt = len(seen_names) - valid_cnt
    print(
        f"Merged {len(seen_names)} programs into {target_dir} "
        f"({valid_cnt} valid, {invalid_cnt} invalid)."
    )


@subcommand(category=Category.DATA)
def build_pcp_program_corpus(seed: int = 42):
    """
    Fuzz new PCP sources, then rebuild valid_and_invalid_programs for dataset generation.
    """
    imp_root = Macros.data_dir / "imp"
    fuzz_jobs = [
        (
            imp_root / "fuzzer_generated",
            imp_root / "invalid_imp_programs/fuzzer_generated",
        ),
        (
            imp_root / "valid_imp_programs/synthetic_cpp",
            imp_root / "invalid_imp_programs/synthetic_cpp",
        ),
    ]
    batch_results: List[FuzzBatchResult] = []
    for input_dir, output_dir in fuzz_jobs:
        batch = _fuzz_imp_programs_in_dir(
            imp_data_dir=input_dir,
            output_dir=output_dir,
            seed=seed,
        )
        batch_results.append(batch)
        expected = len(list(input_dir.glob("*.imp")))
        if batch.success_count != expected:
            raise RuntimeError(
                f"Expected {expected} fuzzed programs from {input_dir}, "
                f"got {batch.success_count}."
            )

    combined_summary = {
        "corpora": [
            {
                "input-dir": str(fuzz_jobs[i][0]),
                "output-dir": str(fuzz_jobs[i][1]),
                "report": str(batch.report_path),
                "summary": str(batch.report_path.with_suffix(".summary.json")),
                "success-count": batch.success_count,
                "failure-count": batch.failure_count,
                "visitor-count": batch.visitor_count,
                "fallback-count": batch.fallback_count,
                "fallback-files": [
                    o.valid_src_filename
                    for o in batch.outcomes
                    if o.fuzz_method == "fallback"
                ],
                "failed-files": [
                    o.valid_src_filename
                    for o in batch.outcomes
                    if o.fuzz_method == "failed"
                ],
            }
            for i, batch in enumerate(batch_results)
        ],
        "total-fallback-count": sum(b.fallback_count for b in batch_results),
        "total-visitor-count": sum(b.visitor_count for b in batch_results),
        "total-failure-count": sum(b.failure_count for b in batch_results),
    }

    combined_summary_path = (
        PCP_FUZZ_REPORTS_DIR / "pcp-fuzz-report-combined.summary.json"
    )
    PCP_FUZZ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    su.io.dump(combined_summary_path, combined_summary, fmt=su.io.Fmt.jsonNoSort)
    print(f"\nCombined fuzz summary: {combined_summary_path}")
    print(
        f"  Total fallback uses: {combined_summary['total-fallback-count']} "
        f"(inspect per-corpus JSONL reports in {PCP_FUZZ_REPORTS_DIR})"
    )

    merge_pcp_valid_and_invalid_programs()


# Data Stats
@subcommand(category=Category.DATA)
def collect_imp_data_stats():
    """
    Collect the statistics of the IMP dataset: # programs, loc, # tokens, len traces.
    """
    data_analyzer = DataAnalysis()
    data_analyzer.collect_imp_data_stats()


@subcommand(category=Category.DATA)
def collect_etp_data_stats():
    """
    Collect the statistics of the etp dataset: len trace
    """
    data_analyzer = DataAnalysis()
    data_analyzer.collect_etp_stats(
        data_dir=Macros.data_dir / "imp/valid_imp_programs/etp/"
    )


@subcommand(category=Category.DATA)
def collect_imp_mutation_rate():
    """
    Collect the mutation rate of the IMP dataset.
    """
    data_analyzer = DataAnalysis()
    data_analyzer.collect_imp_mutation_rate()


@subcommand(category=Category.DATA)
def collect_etp_dataset_tokens():
    """
    Collect the number of tokens in the ETP dataset.
    """
    data_analyzer = DataAnalysis()
    data_analyzer.collect_etp_prompt_tokens()


##
# Create datasets
##
@subcommand(category=Category.DATA)
def make_op_unmutated_imp_sos_dataset(
    imp_program_dir: Path = Macros.data_dir / "imp/valid_imp_programs/human_written/",
    dataset_name: str = "human_written",
):
    """
    Write the unmutated IMP dataset with SOS semantics to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_unmutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="op",
        setup_name="uk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
        dataset_name=dataset_name,
    )


@subcommand(category=Category.DATA)
def make_op_unmutated_imp_k_dataset(
    imp_program_dir: Path = Macros.data_dir / "imp/valid_imp_programs/human_written/",
    dataset_name: str = "human_written",
):
    """
    Write the unmutated IMP dataset with K semantics to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_unmutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="op",
        setup_name="uk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
        dataset_name=dataset_name,
    )


@subcommand(category=Category.DATA)
def make_op_mutated_imp_sos_dataset(
    imp_program_dir: Path = Macros.data_dir / "imp/valid_imp_programs/human_written/",
    dataset_name: str = "human_written",
):
    """
    Write the mutated IMP dataset with SOS semantics to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_mutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="op",
        setup_name="mk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
        dataset_name=dataset_name,
    )


@subcommand(category=Category.DATA)
def make_op_mutated_imp_k_dataset(
    imp_program_dir: Path = Macros.data_dir / "imp/valid_imp_programs/human_written/",
    dataset_name: str = "human_written",
):
    """
    Write the mutated IMP dataset with K semantics to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_mutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="op",
        setup_name="mk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
        dataset_name=dataset_name,
    )


@subcommand(category=Category.DATA)
def make_pcp_unmutated_imp_sos_dataset():
    """
    Write the pcp unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_unmutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="pcp",
        setup_name="uk",
        raw_programs_dir=PCP_PROGRAMS_DIR,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_pcp_unmutated_imp_k_dataset():
    """
    Write the pcp unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_unmutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="pcp",
        setup_name="uk",
        raw_programs_dir=PCP_PROGRAMS_DIR,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_pcp_mutated_imp_sos_dataset():
    """
    Write the pcp mutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_mutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="pcp",
        setup_name="mk",
        raw_programs_dir=PCP_PROGRAMS_DIR,
        target_dir=Macros.data_dir / "dataset",
        include_caucasian_albanian=True,
        use_k_framework=False,
    )


@subcommand(category=Category.DATA)
def make_pcp_mutated_imp_k_dataset():
    """
    Write the pcp mutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_mutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="pcp",
        setup_name="mk",
        raw_programs_dir=PCP_PROGRAMS_DIR,
        target_dir=Macros.data_dir / "dataset",
        include_caucasian_albanian=True,
    )


@subcommand(category=Category.DATA)
def make_srp_unmutated_imp_sos_dataset():
    """
    Write the SRP unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    imp_program_dir = Macros.data_dir / "imp/valid_imp_programs/human_written/"
    data_processor.create_dataset_unmutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="srp",
        setup_name="uk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_srp_unmutated_imp_k_dataset():
    """
    Write the SRP unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    imp_program_dir = Macros.data_dir / "imp/valid_imp_programs/human_written/"
    data_processor.create_dataset_unmutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="srp",
        setup_name="uk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_srp_mutated_imp_sos_dataset():
    """
    Write the SRP unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    imp_program_dir = Macros.data_dir / "imp/valid_imp_programs/human_written/"
    data_processor.create_dataset_mutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="srp",
        setup_name="mk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
        include_caucasian_albanian=True,
    )


@subcommand(category=Category.DATA)
def make_srp_mutated_imp_k_dataset():
    """
    Write the SRP unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    imp_program_dir = Macros.data_dir / "imp/valid_imp_programs/human_written/"
    data_processor.create_dataset_mutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="srp",
        setup_name="mk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
        include_caucasian_albanian=True,
    )


@subcommand(category=Category.DATA)
def make_etp_unmutated_imp_sos_dataset(
    imp_program_dir=Macros.data_dir / "imp/valid_imp_programs/human_written/",
    dataset_name: str = "human_written",
):
    """
    Write the ETP unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_unmutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="etp",
        setup_name="uk",
        dataset_name=dataset_name,
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_etp_unmutated_imp_k_dataset(
    imp_program_dir=Macros.data_dir / "imp/valid_imp_programs/human_written/",
    dataset_name: str = "human_written",
):
    """
    Write the ETP unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_unmutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="etp",
        setup_name="uk",
        dataset_name=dataset_name,
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_etp_mutated_imp_sos_dataset(
    imp_program_dir=Macros.data_dir / "imp/valid_imp_programs/human_written/",
):
    """
    Write the ETP mutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_mutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="etp",
        setup_name="mk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_etp_mutated_imp_k_dataset(
    imp_program_dir=Macros.data_dir / "imp/valid_imp_programs/human_written/",
):
    """
    Write the ETP mutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_mutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="etp",
        setup_name="mk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_ast_mutated_imp_sos_dataset(
    imp_program_dir=Macros.data_dir / "imp/valid_imp_programs/human_written/",
):
    """
    Write the SRP unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_mutated_semantics(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="ast",
        setup_name="mk",
        raw_programs_dir=imp_program_dir,
        target_dir=Macros.data_dir / "dataset",
    )


@subcommand(category=Category.DATA)
def make_nl2rule_unmutated_imp_sos_dataset(
    num_samples: int = 200,
    random_mode: bool = False,
    target_dir: Path = Macros.data_dir / "dataset",
    push_to_hub: bool = False,
    repo_id: str = "LambdaadbmaL/PLSemanticsBench",
):
    """
    Write the NL2Rule unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_notation_comprehension_unmutated(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="nl2rule",
        setup_name="uk",
        num_samples=num_samples,
        random_mode=random_mode,
        target_dir=target_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        split=f"Standard_NumRule5_RandomSample{random_mode}",
    )


@subcommand(category=Category.DATA)
def make_rule2nl_unmutated_imp_sos_dataset(
    num_samples: int = 200,
    random_mode: bool = False,
    target_dir: Path = Macros.data_dir / "dataset",
    push_to_hub: bool = False,
    repo_id: str = "LambdaadbmaL/PLSemanticsBench",
):
    """
    Write the Rule2NL unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_notation_comprehension_unmutated(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="rule2nl",
        setup_name="uk",
        num_samples=num_samples,
        random_mode=random_mode,
        target_dir=target_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        split=f"Standard_NumDescription5_RandomSample{random_mode}",
    )


@subcommand(category=Category.DATA)
def make_nl2rule_mutated_imp_sos_dataset(
    num_samples: int = 200,
    random_mode: bool = False,
    target_dir: Path = Macros.data_dir / "dataset",
    push_to_hub: bool = False,
    repo_id: str = "LambdaadbmaL/PLSemanticsBench",
):
    """
    Write the NL2Rule mutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_notation_comprehension_mutated(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="nl2rule",
        setup_name="mk",
        num_samples=num_samples,
        random_mode=random_mode,
        target_dir=target_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        split=f"NonStandard_NumRule5_RandomSample{random_mode}",
    )


@subcommand(category=Category.DATA)
def make_rule2nl_mutated_imp_sos_dataset(
    num_samples: int = 200,
    random_mode: bool = False,
    target_dir: Path = Macros.data_dir / "dataset",
    push_to_hub: bool = False,
    repo_id: str = "LambdaadbmaL/PLSemanticsBench",
):
    """
    Write the Rule2NL mutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_notation_comprehension_mutated(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.SOS,
        task="rule2nl",
        setup_name="mk",
        num_samples=num_samples,
        random_mode=random_mode,
        target_dir=target_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        split=f"NonStandard_NumDescription5_RandomSample{random_mode}",
    )


@subcommand(category=Category.DATA)
def make_nl2rule_unmutated_imp_k_dataset(
    num_samples: int = 200,
    random_mode: bool = False,
    target_dir: Path = Macros.data_dir / "dataset",
    push_to_hub: bool = False,
    repo_id: str = "LambdaadbmaL/PLSemanticsBench",
):
    """
    Write the NL2Rule unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_notation_comprehension_unmutated(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="nl2rule",
        setup_name="uk",
        num_samples=num_samples,
        random_mode=random_mode,
        target_dir=target_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        split=f"Standard_NumRule5_RandomSample{random_mode}",
    )


@subcommand(category=Category.DATA)
def make_rule2nl_unmutated_imp_k_dataset(
    num_samples: int = 200,
    random_mode: bool = False,
    target_dir: Path = Macros.data_dir / "dataset",
    push_to_hub: bool = False,
    repo_id: str = "LambdaadbmaL/PLSemanticsBench",
):
    """
    Write the Rule2NL unmutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_notation_comprehension_unmutated(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="rule2nl",
        setup_name="uk",
        num_samples=num_samples,
        random_mode=random_mode,
        target_dir=target_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        split=f"Standard_NumDescription5_RandomSample{random_mode}",
    )


@subcommand(category=Category.DATA)
def make_nl2rule_mutated_imp_k_dataset(
    num_samples: int = 200,
    random_mode: bool = False,
    target_dir: Path = Macros.data_dir / "dataset",
    push_to_hub: bool = False,
    repo_id: str = "LambdaadbmaL/PLSemanticsBench",
):
    """
    Write the NL2Rule mutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_notation_comprehension_mutated(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="nl2rule",
        setup_name="mk",
        num_samples=num_samples,
        random_mode=random_mode,
        target_dir=target_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        split=f"NonStandard_NumRule5_RandomSample{random_mode}",
    )


@subcommand(category=Category.DATA)
def make_rule2nl_mutated_imp_k_dataset(
    num_samples: int = 200,
    random_mode: bool = False,
    target_dir: Path = Macros.data_dir / "dataset",
    push_to_hub: bool = False,
    repo_id: str = "LambdaadbmaL/PLSemanticsBench",
):
    """
    Write the Rule2NL mutated IMP dataset to the jsonl file.
    """
    data_processor = DatasetProcessor(PL="IMP")
    data_processor.create_dataset_notation_comprehension_mutated(
        PL="IMP",
        semantics_type=Language.SEMANTICS_TYPE.K,
        task="rule2nl",
        setup_name="mk",
        num_samples=num_samples,
        random_mode=random_mode,
        target_dir=target_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        split=f"NonStandard_NumDescription5_RandomSample{random_mode}",
    )


@subcommand(category=Category.DATA)
def compute_imp_dataset_code_complexity_metrics(imp_program_dir: str, target_dir: str):
    """
    Write the computed code-complexity metrics to jsonl
    """
    imp_lang = IMP("IMP1", Language.SEMANTICS_TYPE.SOS)
    data_list: list = []
    with tqdm(
        total=len(list(Path(imp_program_dir).glob("*.imp"))),
        desc="Computing metrics",
    ) as pbar:
        all_program_files = sorted(Path(imp_program_dir).glob("*.imp"))
        for file_path in all_program_files:
            raw_program = su.io.load(file_path, fmt=su.io.Fmt.txt)
            dt: dict = {}
            dt["LOC"] = imp_lang.get_loc_metrics(raw_program)
            dt["Halstead"] = imp_lang.get_halstead_metrics(raw_program)
            dt["Cyclomatic"] = imp_lang.get_extended_cyclomatic_metrics(raw_program)
            dt["DepDegree"] = imp_lang.get_depdegree_metrics(raw_program)
            data_list.append(dt)
            pbar.update(1)
        # rof
    # htiw
    su.io.dump(
        Path(target_dir) / "code-complexity-metric.jsonl",
        data_list,
    )


# fed


@subcommand(category=Category.DATA)
def compute_program_code_complexity_metrics(
    dataset_file: Path = Macros.data_dir / "dataset" / "dataset-etp-uk-IMP-SOS.jsonl",
    output_file: Path = Macros.data_dir
    / "dataset"
    / "dataset-IMP-code-complexity-metric.jsonl",
):
    """
    Compute the code complexity metrics of the dataset.
    """
    dataset_metrics = collect_dataset_code_complexity_metrics(dataset_file)
    su.io.dump(
        output_file,
        dataset_metrics,
        fmt=su.io.Fmt.jsonNoSort,
    )


# fed


@subcommand(category=Category.DATA)
def filter_dataset_top_n_complex_programs(
    dataset_file: Path,
    dataset_complexity_metrics: Path,
    output_path: Path,
    top_n: int = 43,
    metrics: list[str] = [
        "loc",
        "trace-len",
        HalsteadMetric.Measures.VOCABULARY.value,
        HalsteadMetric.Measures.VOLUME.value,
        ExtendedCyclomaticMetric.Measures.MAX_NESTED_LOOP.value,
        ExtendedCyclomaticMetric.Measures.MAX_NESTED_IF.value,
        ExtendedCyclomaticMetric.Measures.CC.value,
        DepDegreeMetric.Measures.DEP_DEGREE.value,
        "num-assignments",
    ],
):
    """
    Filter existing dataset to select top n complex programs per metric
    """
    data_analyzer = DataAnalysis()
    [filtered_dataset, filtered_metrics] = (
        data_analyzer.filter_dataset_top_n_complex_programs(
            dataset_file, dataset_complexity_metrics, top_n, metrics
        )
    )
    su.io.dump(
        output_path / "filtered-dataset.jsonl",
        filtered_dataset,
    )
    su.io.dump(
        output_path / "filtered-metrics.json",
        filtered_metrics,
        fmt=su.io.Fmt.jsonNoSort,
    )


# fed


@subcommand(category=Category.DATA)
def remove_semantics_syntax_from_dataset(
    dataset_file: Path,
    output_path: Path,
):
    """
    Write programs from dataset to program files
    """
    dataset = su.io.load(dataset_file)
    new_dataset = []
    for data in dataset:
        del data["semantics-type"]
        del data["syntax"]
        del data["semantics"]
        data["language"] = "IMP"
        new_dataset.append(data)
    # rof
    su.io.dump(
        output_path,
        new_dataset,
    )


# fed


@subcommand(category=Category.DATA)
def write_dataset_to_programs(
    dataset_file: Path,
    output_dir: Path,
):
    """
    Write programs from dataset to program files
    """
    dataset = su.io.load(dataset_file)
    for data in dataset:
        program: str = data["program"]
        file_name: str = data["src-filename"]
        write_to_dir(program, file_name, str(output_dir))
    # rof


# fed


if platform.system() == "Linux":

    @subcommand(category=Category.DATA)
    def num_tokens_in_batch(batch_jsonl_path: Path, model_id: str):
        # Hugging Face tokenizer (same one vLLM uses under the hood)
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        prompts: list = su.io.load(batch_jsonl_path)
        print(f"Using tokenizer {tok}")
        num_tokens: list = []
        for prompt in prompts:
            num_tokens.append(len(tok.encode(prompt, add_special_tokens=True)))
        # rof
        print(f"Number of tokens in the prompts in order: {num_tokens}")

    # fed
# fi
