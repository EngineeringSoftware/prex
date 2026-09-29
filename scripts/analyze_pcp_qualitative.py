#!/usr/bin/env python3
"""
Qualitative analysis for PredExe (PCP) results.

Surfaces failure patterns (executability vs. rule identification) and selects
the smallest failing programs per dataset split for manual inspection.

Default focus: top-performing models
  - DeepSeek-R1-Distill-Qwen 32B (da)
  - Qwen2.5-Coder 32B (CoT)
  - Ministral 3 14B (CoT)

Outputs markdown reports and a JSON case file under results/examples/pcp-qualitative/.

Usage:
  python scripts/analyze_pcp_qualitative.py
  python scripts/analyze_pcp_qualitative.py --seed 42 --top-n 2
  python scripts/analyze_pcp_qualitative.py --setup uk --semantics SOS --split human_written
  python scripts/analyze_pcp_qualitative.py --output-dir scratch/pcp-qualitative
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from llm_interpreter.macros import Macros
from llm_interpreter.results.results_analysis import extract_pcp_prediction

from pcp_error_type_helpers import (  # noqa: E402
    canonical_error_rule_numbers,
    pcp_rule_to_label_map,
    rule_to_error_type,
)
from visualize_pcp_split_accuracy_tables import (  # noqa: E402
    TABLE_COLUMNS,
    column_exp_name,
    column_pattern,
    column_setup,
)

DATASET_SPLITS = (
    "human_written",
    "fuzzer_generated",
    "synthetic_cpp",
)
SPLIT_LABELS = {
    "human_written": "human-written",
    "fuzzer_generated": "fuzzer-generated",
    "synthetic_cpp": "llm-translated",
}
DEFAULT_OUTPUT = REPO_ROOT / "results" / "examples" / "pcp-qualitative"
DEFAULT_SEED = "42"
RULE_PATTERN = re.compile(r"(?:rule[\s-]*)?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ModelSpec:
    row_id: str
    model_name: str
    short_label: str


TOP_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        row_id="deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
        model_name="deepseek-ai-DeepSeek-R1-Distill-Qwen-32B",
        short_label="DeepSeek-Qwen 32B",
    ),
    ModelSpec(
        row_id="Qwen-Qwen2.5-Coder-32B-Instruct-cot",
        model_name="Qwen-Qwen2.5-Coder-32B-Instruct",
        short_label="Qwen2.5-Coder 32B-CoT",
    ),
    ModelSpec(
        row_id="mistralai-Ministral-3-14B-Instruct-2512-BF16-cot",
        model_name="mistralai-Ministral-3-14B-Instruct-2512",
        short_label="Ministral 14B-CoT",
    ),
)

CONFIG_LABELS = {
    ("uk", "IMP-K", None): "uk / IMP-K",
    ("mk", "IMP-K", "KeywordSwap"): "mk / IMP-K / KeywordSwap",
    ("mk", "IMP-K", "KeywordObf"): "mk / IMP-K / KeywordObf",
    ("uk", "IMP-SOS", None): "uk / IMP-SOS",
    ("mk", "IMP-SOS", "KeywordSwap"): "mk / IMP-SOS / KeywordSwap",
    ("mk", "IMP-SOS", "KeywordObf"): "mk / IMP-SOS / KeywordObf",
}


@dataclass
class ParsedPrediction:
    return_code: str
    rule_raw: str
    rule_normalized: str
    malformed: bool


@dataclass
class FailureAnalysis:
    mode: str
    exec_correct: bool
    rule_correct: bool
    true_return_code: str
    true_rule: str
    true_error_type: str
    pred_return_code: str
    pred_rule: str
    pred_error_type: str


@dataclass
class ProgramCase:
    config_label: str
    setup: str
    exp_name: str
    pattern: str | None
    split: str
    split_label: str
    seed: str
    program_id: str
    src_filename: str
    line_count: int
    program: str
    ground_truth: dict
    selected_for_model: str | None = None
    models: dict = field(default_factory=dict)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    opener = gzip.open if path.suffix == ".gz" else Path.open
    mode = "rt" if path.suffix == ".gz" else "r"
    records: list[dict] = []
    with opener(path, mode, encoding=None if path.suffix == ".gz" else "utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_result_file(result_dir: Path, seed: str) -> Path | None:
    for suffix in (".jsonl", ".jsonl.gz"):
        candidate = result_dir / f"results-{seed}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def normalize_rule(rule_text: str) -> str:
    match = RULE_PATTERN.search(rule_text or "")
    if match:
        return f"Rule {match.group(1)}"
    return (rule_text or "").strip()


def program_text(record: dict) -> str:
    for key in ("program", "mutated-program"):
        text = record.get(key) or ""
        if text.strip():
            return text.rstrip()
    return ""


def program_line_count(record: dict) -> int:
    text = program_text(record)
    if not text:
        return 0
    return len([line for line in text.splitlines() if line.strip()])


def semantics_from_col(col: tuple) -> str:
    return "K" if column_exp_name(col) == "IMP-K" else "SOS"


def config_label(col: tuple) -> str:
    return CONFIG_LABELS[(column_setup(col), column_exp_name(col), column_pattern(col))]


def result_dir_for(col: tuple, model: ModelSpec) -> Path:
    setup = column_setup(col)
    exp_name = column_exp_name(col)
    return Macros.results_dir / "pcp" / setup / exp_name / model.row_id


def filter_records(
    records: list[dict],
    split: str,
    setup: str,
    pattern: str | None,
) -> list[dict]:
    filtered = [record for record in records if record.get("dataset-source") == split]
    if setup == "mk" and pattern is not None:
        filtered = [
            record for record in filtered if record.get("mutation-pattern") == pattern
        ]
    return filtered


def parse_prediction(record: dict) -> ParsedPrediction:
    pred_return_code, pred_rule_raw = extract_pcp_prediction(record["model-prediction"])
    malformed = pred_return_code == ""
    return ParsedPrediction(
        return_code=pred_return_code,
        rule_raw=pred_rule_raw,
        rule_normalized=normalize_rule(pred_rule_raw),
        malformed=malformed,
    )


def analyze_failure(record: dict, semantics: str) -> FailureAnalysis:
    parsed = parse_prediction(record)
    rule_map = pcp_rule_to_label_map(semantics)
    canonical_rules = canonical_error_rule_numbers(semantics)

    true_return_code = record["ans"]
    true_rule = record.get("semantic-error-rule") or "None"
    if true_return_code == "##success##":
        true_rule = "None"

    pred_return_code = parsed.return_code
    pred_rule = parsed.rule_normalized if parsed.return_code == "##error##" else "None"
    if parsed.malformed:
        pred_return_code = ""
        pred_rule = ""

    exec_correct = (
        not parsed.malformed
        and (
            (true_return_code == "##success##" and pred_return_code == "##success##")
            or (true_return_code == "##error##" and pred_return_code == "##error##")
        )
    )
    rule_correct = exec_correct and (
        true_return_code == "##success##" or true_rule == pred_rule
    )

    if parsed.malformed:
        mode = "malformed"
    elif true_return_code == "##success##" and pred_return_code == "##error##":
        mode = "false_error"
    elif true_return_code == "##error##" and pred_return_code == "##success##":
        mode = "false_success"
    elif true_return_code == "##error##" and pred_return_code == "##error##":
        if true_rule == pred_rule:
            mode = "correct"
        elif not pred_rule or not pred_rule.startswith("Rule "):
            mode = "unparsed_rule"
        else:
            mode = "wrong_rule"
    elif true_return_code == "##success##" and pred_return_code == "##success##":
        mode = "correct"
    else:
        mode = "other"

    return FailureAnalysis(
        mode=mode,
        exec_correct=exec_correct,
        rule_correct=rule_correct,
        true_return_code=true_return_code,
        true_rule=true_rule,
        true_error_type=rule_to_error_type(true_rule, rule_map, canonical_rules),
        pred_return_code=pred_return_code,
        pred_rule=pred_rule,
        pred_error_type=rule_to_error_type(pred_rule, rule_map, canonical_rules),
    )


def is_incorrect(record: dict, semantics: str) -> bool:
    return not analyze_failure(record, semantics).rule_correct


def load_model_records(
    col: tuple,
    model: ModelSpec,
    seed: str,
) -> list[dict]:
    result_file = resolve_result_file(result_dir_for(col, model), seed)
    if result_file is None:
        return []
    return load_jsonl(result_file)


def build_records_by_id(records: list[dict]) -> dict[str, dict]:
    return {record["id"]: record for record in records}


def summarize_rule_failures(
    records: list[dict],
    semantics: str,
) -> dict[str, Counter]:
    by_rule: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        analysis = analyze_failure(record, semantics)
        if analysis.rule_correct:
            continue
        true_rule = analysis.true_rule
        by_rule[true_rule][analysis.mode] += 1
        by_rule[true_rule]["total_failures"] += 1
    return by_rule


def summarize_split(
    records: list[dict],
    semantics: str,
) -> dict[str, int | float]:
    total = len(records)
    if total == 0:
        return {"total": 0}

    rule_correct = 0
    exec_correct = 0
    mode_counts: Counter = Counter()
    for record in records:
        analysis = analyze_failure(record, semantics)
        mode_counts[analysis.mode] += 1
        if analysis.exec_correct:
            exec_correct += 1
        if analysis.rule_correct:
            rule_correct += 1

    return {
        "total": total,
        "exec_accuracy": exec_correct / total,
        "rule_accuracy": rule_correct / total,
        "mode_counts": dict(mode_counts),
    }


def select_smallest_failures(
    records: list[dict],
    semantics: str,
    top_n: int,
) -> list[dict]:
    failures = [
        record
        for record in records
        if is_incorrect(record, semantics) and program_line_count(record) > 0
    ]
    failures.sort(
        key=lambda record: (
            program_line_count(record),
            len(program_text(record)),
            record.get("src-filename") or record["id"],
        )
    )
    return failures[:top_n]


def truncate(text: str, limit: int = 1200) -> str:
    text = text.rstrip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_model_block(record: dict, semantics: str, model: ModelSpec) -> str:
    analysis = analyze_failure(record, semantics)
    parsed = parse_prediction(record)
    lines = [
        f"### {model.short_label}",
        f"- Failure mode: `{analysis.mode}`",
        f"- Processed prediction: return=`{analysis.pred_return_code or '(malformed)'}` "
        f"rule=`{analysis.pred_rule or '(none)'}` "
        f"error-type=`{analysis.pred_error_type}`",
    ]
    if analysis.mode == "wrong_rule":
        lines.append(
            f"- Confused `{analysis.true_error_type}` ({analysis.true_rule}) "
            f"with `{analysis.pred_error_type}` ({analysis.pred_rule})"
        )
    lines.extend(
        [
            "",
            "#### Model output",
            "```",
            truncate(record.get("model-prediction", "")),
            "```",
            "",
            f"Raw rule tag: `{parsed.rule_raw}`",
        ]
    )
    return "\n".join(lines)


def format_program_case(case: ProgramCase, semantics: str) -> str:
    gt = case.ground_truth
    lines = [
        f"## {case.program_id}",
        "",
        f"- Config: {case.config_label}",
        f"- Split: {case.split_label}",
        f"- Seed: {case.seed}",
        f"- Lines: {case.line_count}",
        f"- Source file: `{case.src_filename}`",
        f"- Selected because: failed for **{case.selected_for_model}**",
        f"- Ground truth: `{gt['ans']}` / `{gt['true_rule']}` "
        f"({gt['true_error_type']})",
        "",
        "### Program",
        "```",
        case.program,
        "```",
        "",
    ]
    for model_label, record in case.models.items():
        model = next(m for m in TOP_MODELS if m.short_label == model_label)
        lines.append(format_model_block(record, semantics, model))
        lines.append("")
    return "\n".join(lines)


def build_case(
    col: tuple,
    split: str,
    seed: str,
    anchor_record: dict,
    selected_model: ModelSpec,
    model_records: dict[str, dict[str, dict]],
) -> ProgramCase:
    setup = column_setup(col)
    exp_name = column_exp_name(col)
    pattern = column_pattern(col)
    semantics = semantics_from_col(col)
    program_id = anchor_record["id"]
    analysis = analyze_failure(anchor_record, semantics)

    models: dict[str, dict] = {}
    for model in TOP_MODELS:
        record = model_records[model.row_id].get(program_id)
        if record is not None:
            models[model.short_label] = record

    return ProgramCase(
        config_label=config_label(col),
        setup=setup,
        exp_name=exp_name,
        pattern=pattern,
        split=split,
        split_label=SPLIT_LABELS[split],
        seed=seed,
        program_id=program_id,
        src_filename=anchor_record.get("src-filename") or "",
        line_count=program_line_count(anchor_record),
        program=program_text(anchor_record),
        ground_truth={
            "ans": anchor_record["ans"],
            "true_rule": analysis.true_rule,
            "true_error_type": analysis.true_error_type,
            "semantic-error-type": anchor_record.get("semantic-error-type"),
        },
        selected_for_model=selected_model.short_label,
        models=models,
    )


def write_summary_markdown(
    output_dir: Path,
    seed: str,
    columns: list[tuple],
    summaries: dict[tuple, dict[str, dict]],
    rule_summaries: dict[tuple, dict[str, dict[str, Counter]]],
    splits: tuple[str, ...] = DATASET_SPLITS,
) -> Path:
    path = output_dir / f"summary-seed-{seed}.md"
    lines = [
        "# PredExe qualitative summary",
        "",
        f"Seed: `{seed}`",
        "",
        "Rule-level accuracy matches the paper metric: valid executability plus",
        "correct violated-rule identification on error programs.",
        "",
    ]

    for col in columns:
        label = config_label(col)
        semantics = semantics_from_col(col)
        lines.extend([f"## {label}", ""])
        col_summaries = summaries.get(col, {})
        if not col_summaries:
            lines.extend(["_(no results)_", ""])
            continue

        for model in TOP_MODELS:
            lines.append(f"### {model.short_label}")
            for split in splits:
                split_label = SPLIT_LABELS[split]
                stats = col_summaries.get(f"{model.short_label}:{split}")
                if not stats or stats.get("total", 0) == 0:
                    lines.append(f"- {split_label}: n/a")
                    continue
                mode_counts = stats.get("mode_counts", {})
                mode_text = ", ".join(
                    f"`{mode}`={count}"
                    for mode, count in sorted(mode_counts.items())
                )
                lines.append(
                    f"- {split_label}: rule accuracy {stats['rule_accuracy']:.1%}, "
                    f"exec accuracy {stats['exec_accuracy']:.1%} "
                    f"({stats['total']} programs; {mode_text})"
                )
            lines.append("")

            for split in splits:
                split_label = SPLIT_LABELS[split]
                rule_stats = rule_summaries.get(col, {}).get(model.short_label, {}).get(split, {})
                if not rule_stats:
                    continue
                lines.append(f"#### {model.short_label} / {split_label} failures by rule")
                lines.append("")
                lines.append("| True rule | Error type | Failures | Top failure modes |")
                lines.append("| --- | --- | ---: | --- |")
                for true_rule in sorted(rule_stats.keys(), key=lambda r: (r == "None", r)):
                    counter = rule_stats[true_rule]
                    total = counter.get("total_failures", 0)
                    if total == 0:
                        continue
                    error_type = rule_to_error_type(
                        true_rule,
                        pcp_rule_to_label_map(semantics),
                        canonical_error_rule_numbers(semantics),
                    )
                    top_modes = ", ".join(
                        f"{mode}={counter[mode]}"
                        for mode, _ in counter.most_common(4)
                        if mode != "total_failures"
                    )
                    lines.append(
                        f"| `{true_rule}` | {error_type} | {total} | {top_modes} |"
                    )
                lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_cases_markdown(output_dir: Path, seed: str, cases: list[ProgramCase]) -> Path:
    path = output_dir / f"smallest-failures-seed-{seed}.md"
    lines = [
        "# PredExe smallest failing programs",
        "",
        f"Seed: `{seed}`",
        "",
        "For each configuration, dataset split, and top model, this lists the",
        f"smallest non-empty programs (by line count) where that model failed.",
        "Each case includes all three top models for comparison.",
        "",
    ]

    current_header = None
    for case in cases:
        header = (case.config_label, case.split_label, case.selected_for_model)
        if header != current_header:
            current_header = header
            lines.extend(
                [
                    f"# {case.config_label} / {case.split_label} / {case.selected_for_model}",
                    "",
                ]
            )
        semantics = "K" if case.exp_name == "IMP-K" else "SOS"
        lines.append(format_program_case(case, semantics))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_cases_json(output_dir: Path, seed: str, cases: list[ProgramCase]) -> Path:
    path = output_dir / f"cases-seed-{seed}.json"
    serializable = []
    for case in cases:
        item = asdict(case)
        item["models"] = {
            model_label: {
                "failure": asdict(analyze_failure(record, "K" if case.exp_name == "IMP-K" else "SOS")),
                "model_prediction": record.get("model-prediction", ""),
            }
            for model_label, record in case.models.items()
        }
        serializable.append(item)
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_readme(output_dir: Path, seed: str, paths: list[Path], case_count: int) -> None:
    readme = output_dir / "README.md"
    lines = [
        "# PredExe qualitative analysis",
        "",
        "Generated by `scripts/analyze_pcp_qualitative.py`.",
        "",
        f"- Seed: `{seed}`",
        f"- Cases collected: {case_count}",
        "",
        "## Files",
        "",
    ]
    for path in paths:
        lines.append(f"- [{path.name}]({path.name})")
    lines.extend(
        [
            "",
            "## Models",
            "",
        ]
    )
    for model in TOP_MODELS:
        lines.append(f"- {model.short_label} (`{model.row_id}`)")
    lines.extend(
        [
            "",
            "## Suggested workflow",
            "",
            "1. Read `summary-seed-*.md` for per-rule failure counts and confusion patterns.",
            "2. Inspect `smallest-failures-seed-*.md` for concrete programs and model outputs.",
            "3. Use `cases-seed-*.json` if you want to filter or re-render cases programmatically.",
            "",
            "Re-run with `--top-n`, `--seed`, or `--setup`/`--semantics`/`--split` to narrow scope.",
            "",
        ]
    )
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_cases(
    columns: list[tuple],
    seed: str,
    top_n: int,
    splits: tuple[str, ...] = DATASET_SPLITS,
) -> tuple[list[ProgramCase], dict, dict]:
    cases: list[ProgramCase] = []
    summaries: dict[tuple, dict[str, dict]] = {}
    rule_summaries: dict[tuple, dict[str, dict[str, Counter]]] = {}

    for col in columns:
        setup = column_setup(col)
        pattern = column_pattern(col)
        semantics = semantics_from_col(col)
        col_summary: dict[str, dict] = {}
        col_rule_summary: dict[str, dict[str, Counter]] = {
            model.short_label: {} for model in TOP_MODELS
        }

        model_records_by_id: dict[str, dict[str, dict]] = {}
        for model in TOP_MODELS:
            records = load_model_records(col, model, seed)
            model_records_by_id[model.row_id] = build_records_by_id(records)

        for split in splits:
            for model in TOP_MODELS:
                filtered = filter_records(
                    load_model_records(col, model, seed), split, setup, pattern
                )
                col_summary[f"{model.short_label}:{split}"] = summarize_split(
                    filtered, semantics
                )
                col_rule_summary[model.short_label][split] = summarize_rule_failures(
                    filtered, semantics
                )

                selected = select_smallest_failures(filtered, semantics, top_n)
                for record in selected:
                    cases.append(
                        build_case(
                            col,
                            split,
                            seed,
                            record,
                            model,
                            model_records_by_id,
                        )
                    )

        summaries[col] = col_summary
        rule_summaries[col] = col_rule_summary

    return cases, summaries, rule_summaries


def filter_columns(
    setup: str | None,
    semantics: str | None,
    pattern: str | None,
) -> list[tuple]:
    columns = list(TABLE_COLUMNS)
    if setup is not None:
        columns = [col for col in columns if column_setup(col) == setup]
    if semantics is not None:
        exp_name = f"IMP-{semantics}"
        columns = [col for col in columns if column_exp_name(col) == exp_name]
    if pattern is not None:
        columns = [col for col in columns if column_pattern(col) == pattern]
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualitative PredExe analysis for top-performing models."
    )
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Result seed (default: 42)")
    parser.add_argument(
        "--top-n",
        type=int,
        default=2,
        help="Smallest failing programs per model/split/config (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--setup", choices=("uk", "mk"), help="Limit to one setup")
    parser.add_argument("--semantics", choices=("K", "SOS"), help="Limit to one semantics")
    parser.add_argument(
        "--pattern",
        choices=("KeywordSwap", "KeywordObf"),
        help="Limit to one mk mutation pattern",
    )
    parser.add_argument(
        "--split",
        choices=DATASET_SPLITS,
        help="Limit to one dataset split",
    )
    args = parser.parse_args()

    columns = filter_columns(args.setup, args.semantics, args.pattern)
    splits = (args.split,) if args.split else DATASET_SPLITS

    cases, summaries, rule_summaries = collect_cases(
        columns, args.seed, args.top_n, splits
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = write_summary_markdown(
        args.output_dir, args.seed, columns, summaries, rule_summaries, splits
    )
    cases_md_path = write_cases_markdown(args.output_dir, args.seed, cases)
    cases_json_path = write_cases_json(args.output_dir, args.seed, cases)
    write_readme(
        args.output_dir,
        args.seed,
        [summary_path, cases_md_path, cases_json_path],
        len(cases),
    )

    print(f"Wrote qualitative analysis to {args.output_dir}")
    print(f"  summary:   {summary_path.name}")
    print(f"  cases md:  {cases_md_path.name} ({len(cases)} cases)")
    print(f"  cases json:{cases_json_path.name}")
    print(f"  readme:    README.md")
    if args.split is not None:
        print(f"  split filter: {SPLIT_LABELS[args.split]}")


if __name__ == "__main__":
    main()
