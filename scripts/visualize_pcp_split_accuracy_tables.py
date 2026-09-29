"""
Build paper-style PredExe accuracy LaTeX tables per dataset split.

Produces three tables (human-written, llm-translated, fuzzer-generated) with
layout:
  UK K | MK K Swap | MK K Obf | UK SOS | MK SOS Swap | MK SOS Obf

Rows start with the random guesser baseline, then non-CoT models (Qwen2.5-Coder and
Ministral 3, sorted alphabetically by family and by size within each family), a
separating line, then CoT/reasoning models (Qwen2.5-Coder CoT, DeepSeek-R1-Distill,
and Ministral 3 CoT, sorted the same way).

Usage:
  python scripts/visualize_pcp_split_accuracy_tables.py
  python scripts/visualize_pcp_split_accuracy_tables.py --seed 42
  python scripts/visualize_pcp_split_accuracy_tables.py --output-dir scratch/pcp-split-tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "papers" / "lmpl26" / "tables"

DEFAULT_MODEL_PREFIX = "Qwen-Qwen2.5-Coder"
DATASET_SPLITS = (
    "human_written",
    "synthetic_cpp",
    "fuzzer_generated",
)
SPLIT_LABELS = {
    "human_written": "human-written",
    "fuzzer_generated": "fuzzer-generated",
    "synthetic_cpp": "llm-translated",
}
STRATEGIES = ("da", "cot")
DATASETS2MACROS = {
    "human_written": r"\humanwrit",
    "synthetic_cpp": r"\llmtrans",
    "fuzzer_generated": r"\fuzzgen",
}
METRICS2MACROS = {
    "pcp-uk-IMP-K-accuracy": r"\standardSemCap",
    "pcp-mk-IMP-K-KeywordSwap-accuracy": r"\KeywordSwap",
    "pcp-mk-IMP-K-KeywordObf-accuracy": r"\KeywordObf",
    "pcp-uk-IMP-SOS-accuracy": r"\standardSemCap",
    "pcp-mk-IMP-SOS-KeywordSwap-accuracy": r"\KeywordSwap",
    "pcp-mk-IMP-SOS-KeywordObf-accuracy": r"\KeywordObf",
}
LLMS2MACROS = {
    "Qwen-Qwen2.5-Coder-3B-Instruct-da": r"\qwenCoder{3}",
    "Qwen-Qwen2.5-Coder-3B-Instruct-cot": r"\qwenCoderCoT{3}",
    "Qwen-Qwen2.5-Coder-7B-Instruct-da": r"\qwenCoder{7}",
    "Qwen-Qwen2.5-Coder-7B-Instruct-cot": r"\qwenCoderCoT{7}",
    "Qwen-Qwen2.5-Coder-14B-Instruct-da": r"\qwenCoder{14}",
    "Qwen-Qwen2.5-Coder-14B-Instruct-cot": r"\qwenCoderCoT{14}",
    "Qwen-Qwen2.5-Coder-32B-Instruct-da": r"\qwenCoder{32}",
    "Qwen-Qwen2.5-Coder-32B-Instruct-cot": r"\qwenCoderCoT{32}",
    "random-guesser-dataset-distribution-da": "Random",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da": r"\dpskQwen{14}",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da": r"\dpskQwen{32}",
    "mistralai-Ministral-3-3B-Instruct-2512-BF16-da": r"\ministral{3}",
    "mistralai-Ministral-3-3B-Instruct-2512-BF16-cot": r"\ministralCoT{3}",
    "mistralai-Ministral-3-8B-Instruct-2512-BF16-da": r"\ministral{8}",
    "mistralai-Ministral-3-8B-Instruct-2512-BF16-cot": r"\ministralCoT{8}",
    "mistralai-Ministral-3-14B-Instruct-2512-BF16-da": r"\ministral{14}",
    "mistralai-Ministral-3-14B-Instruct-2512-BF16-cot": r"\ministralCoT{14}",
}
DEFAULT_SEED = "42"
MODEL_SIZES = ("3B", "7B", "14B", "32B")
MINISTRAL_MODEL_PREFIX = "mistralai-Ministral-3"
MINISTRAL_MODEL_SIZES = ("3B", "8B", "14B")
MINISTRAL_INSTRUCT_VARIANT = "2512-BF16"
RANDOM_GUESSER_MODEL_NAME = "random-guesser-dataset-distribution"
RANDOM_GUESSER_ROW_ID = f"{RANDOM_GUESSER_MODEL_NAME}-da"
RANDOM_GUESSER_RESULT_DIR = f"{RANDOM_GUESSER_MODEL_NAME}-da"
TABLE_SUFFIX = "qwen-coder-by-split"
SPLIT_SLUGS = {
    "human_written": "human-written",
    "fuzzer_generated": "fuzzer-generated",
    "synthetic_cpp": "llm-translated",
}

TABLE_COLUMNS = (
    ("uk", "IMP-K", None, "pcp-uk-IMP-K-accuracy"),
    ("mk", "IMP-K", "KeywordSwap", "pcp-mk-IMP-K-KeywordSwap-accuracy"),
    ("mk", "IMP-K", "KeywordObf", "pcp-mk-IMP-K-KeywordObf-accuracy"),
    ("uk", "IMP-SOS", None, "pcp-uk-IMP-SOS-accuracy"),
    ("mk", "IMP-SOS", "KeywordSwap", "pcp-mk-IMP-SOS-KeywordSwap-accuracy"),
    ("mk", "IMP-SOS", "KeywordObf", "pcp-mk-IMP-SOS-KeywordObf-accuracy"),
)
MK_BASELINE_COL_IDX = {1: 0, 2: 0, 4: 3, 5: 3}


def model_dir_name(
    model_prefix: str,
    model_size: str,
    strategy: str,
    instruct_variant: str = "",
) -> str:
    if instruct_variant:
        return f"{model_prefix}-{model_size}-Instruct-{instruct_variant}-{strategy}"
    return f"{model_prefix}-{model_size}-Instruct-{strategy}"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


@dataclass
class PCPResults:
    model_name: str = ""
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    rule_true: list[str] = field(default_factory=list)
    rule_pred: list[str] = field(default_factory=list)
    malformed_cnt: int = 0


@dataclass
class PCPAccuracyStats:
    total: int
    exec_correct: int
    success_total: int
    success_correct: int
    error_total: int
    error_correct: int
    error_rule_correct: int
    malformed: int

    @property
    def rule_accuracy(self) -> float:
        rule_correct = self.success_correct + self.error_rule_correct
        return rule_correct / self.total if self.total else 0.0


def extract_content_between_tags(text: str, tag: str) -> str:
    last_opening = text.rfind(f"<{tag}>")
    if last_opening == -1:
        return ""
    text = text[last_opening:]
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""


def extract_pcp_prediction(model_output: str) -> tuple[str, str]:
    pred_return_code = extract_content_between_tags(model_output, "ans").strip()
    pred_rule = extract_content_between_tags(model_output, "rule").strip()
    match = re.search(r"\[([^\]]+)\]", pred_rule)
    if match:
        pred_rule = match.group(1).strip()
    return pred_return_code, pred_rule


def analyze_pcp_records(results: list[dict], model_name: str) -> PCPResults:
    analysis = PCPResults(model_name=model_name)
    pattern = re.compile(r"(?:rule[\s-]*)?(\d+)", re.IGNORECASE)
    for result in results:
        if result["ans"] == "##success##":
            analysis.y_true.append(1)
            analysis.rule_true.append("None")
        else:
            analysis.y_true.append(0)
            analysis.rule_true.append(result["semantic-error-rule"])

        pred_return_code, pred_rule = extract_pcp_prediction(result["model-prediction"])
        if pred_return_code == "":
            analysis.malformed_cnt += 1
        if pred_return_code == "##success##":
            analysis.y_pred.append(1)
            analysis.rule_pred.append("None")
        else:
            analysis.y_pred.append(0)
            matches = pattern.search(pred_rule)
            if matches:
                analysis.rule_pred.append(f"Rule {matches.group(1)}")
            else:
                analysis.rule_pred.append(pred_rule)
    return analysis


def compute_pcp_stats(result: PCPResults) -> PCPAccuracyStats:
    success_total = sum(1 for y in result.y_true if y == 1)
    error_total = sum(1 for y in result.y_true if y == 0)
    success_correct = sum(
        1
        for y_true, y_pred in zip(result.y_true, result.y_pred)
        if y_true == 1 and y_pred == 1
    )
    error_correct = sum(
        1
        for y_true, y_pred in zip(result.y_true, result.y_pred)
        if y_true == 0 and y_pred == 0
    )
    error_rule_correct = sum(
        1
        for y_true, y_pred, rule_true, rule_pred in zip(
            result.y_true, result.y_pred, result.rule_true, result.rule_pred
        )
        if y_true == 0 and y_pred == 0 and rule_true == rule_pred
    )
    exec_correct = sum(
        1 for y_true, y_pred in zip(result.y_true, result.y_pred) if y_true == y_pred
    )
    return PCPAccuracyStats(
        total=len(result.y_true),
        exec_correct=exec_correct,
        success_total=success_total,
        success_correct=success_correct,
        error_total=error_total,
        error_correct=error_correct,
        error_rule_correct=error_rule_correct,
        malformed=result.malformed_cnt,
    )


@dataclass
class ModelConfig:
    size: str
    strategy: str
    model_prefix: str
    instruct_variant: str = ""

    @property
    def model_name(self) -> str:
        if self.instruct_variant:
            return f"{self.model_prefix}-{self.size}-Instruct-{self.instruct_variant}"
        return f"{self.model_prefix}-{self.size}-Instruct"

    @property
    def row_id(self) -> str:
        return model_dir_name(
            self.model_prefix,
            self.size,
            self.strategy,
            self.instruct_variant,
        )

    @property
    def row_label(self) -> str:
        strategy_label = "DA" if self.strategy == "da" else "CoT"
        return f"{self.size} ({strategy_label})"


@dataclass
class BaselineConfig:
    row_id: str
    row_label: str
    model_name: str


RANDOM_GUESSER_BASELINE = BaselineConfig(
    row_id=RANDOM_GUESSER_ROW_ID,
    row_label="Random",
    model_name=RANDOM_GUESSER_MODEL_NAME,
)

REASONING_MODELS = (
    BaselineConfig(
        row_id="deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
        row_label="DeepSeek-R1-Distill-Qwen-14B",
        model_name="deepseek-ai-DeepSeek-R1-Distill-Qwen-14B",
    ),
    BaselineConfig(
        row_id="deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
        row_label="DeepSeek-R1-Distill-Qwen-32B",
        model_name="deepseek-ai-DeepSeek-R1-Distill-Qwen-32B",
    ),
)


@dataclass
class SplitTable:
    split: str
    configs: list[ModelConfig | BaselineConfig]
    values: list[list[float | None]]


def result_dir_for_config(cfg: ModelConfig | BaselineConfig, col: tuple) -> Path:
    base = RESULTS_DIR / "pcp" / column_setup(col) / column_exp_name(col)
    if isinstance(cfg, ModelConfig):
        return base / cfg.row_id
    return base / cfg.row_id


def column_setup(col: tuple) -> str:
    return col[0]


def column_exp_name(col: tuple) -> str:
    return col[1]


def column_pattern(col: tuple) -> str | None:
    return col[2]


def column_key(col: tuple) -> str:
    return col[3]


def load_result_sets(result_dir: Path, seed: str | None = None) -> list[list[dict]]:
    if seed is not None:
        result_file = result_dir / f"results-{seed}.jsonl"
        if not result_file.is_file():
            return []
        return [load_jsonl(result_file)]

    result_files = sorted(result_dir.glob("results-*.jsonl"))
    if not result_files:
        return []
    return [load_jsonl(result_file) for result_file in result_files]


def filter_records(
    results: list[dict],
    split: str,
    setup: str,
    pattern: str | None,
) -> list[dict]:
    filtered = [record for record in results if record.get("dataset-source") == split]
    if setup == "mk" and pattern is not None:
        filtered = [
            record for record in filtered if record.get("mutation-pattern") == pattern
        ]
    return filtered


def cell_accuracy(
    results: list[dict],
    model_name: str,
    split: str,
    setup: str,
    pattern: str | None,
) -> float | None:
    filtered = filter_records(results, split, setup, pattern)
    if not filtered:
        return None
    trace = analyze_pcp_records(filtered, model_name)
    stats = compute_pcp_stats(trace)
    return stats.rule_accuracy


def cell_accuracy_averaged(
    result_sets: list[list[dict]],
    model_name: str,
    split: str,
    setup: str,
    pattern: str | None,
) -> float | None:
    accuracies: list[float] = []
    for results in result_sets:
        accuracy = cell_accuracy(
            results,
            model_name,
            split,
            setup,
            pattern,
        )
        if accuracy is not None:
            accuracies.append(accuracy)
    if not accuracies:
        return None
    return statistics.mean(accuracies)


def discover_model_configs(
    model_prefix: str,
    model_sizes: tuple[str, ...] = MODEL_SIZES,
    instruct_variant: str = "",
) -> list[ModelConfig]:
    configs: list[ModelConfig] = []
    for strategy in STRATEGIES:
        for size in model_sizes:
            found = False
            for col in TABLE_COLUMNS:
                result_dir = (
                    RESULTS_DIR
                    / "pcp"
                    / column_setup(col)
                    / column_exp_name(col)
                    / model_dir_name(model_prefix, size, strategy, instruct_variant)
                )
                if result_dir.is_dir() and list(result_dir.glob("results-*.jsonl")):
                    found = True
                    break
            if found:
                configs.append(
                    ModelConfig(
                        size=size,
                        strategy=strategy,
                        model_prefix=model_prefix,
                        instruct_variant=instruct_variant,
                    )
                )
    return configs


def discover_ministral_model_configs() -> list[ModelConfig]:
    return discover_model_configs(
        MINISTRAL_MODEL_PREFIX,
        MINISTRAL_MODEL_SIZES,
        MINISTRAL_INSTRUCT_VARIANT,
    )


def row_display_label(cfg: ModelConfig | BaselineConfig) -> str:
    return LLMS2MACROS.get(cfg.row_id, getattr(cfg, "row_label", cfg.row_id))


def parse_size_b(size: str) -> int:
    match = re.match(r"(\d+)", size)
    return int(match.group(1)) if match else 0


def model_family_key(cfg: ModelConfig | BaselineConfig) -> str:
    if isinstance(cfg, ModelConfig):
        if cfg.model_prefix == MINISTRAL_MODEL_PREFIX:
            return "ministral"
        if cfg.model_prefix == DEFAULT_MODEL_PREFIX:
            return "qwen"
        return cfg.model_prefix.casefold()
    if "deepseek" in cfg.row_id.casefold():
        return "deepseek"
    return cfg.row_id.casefold()


def model_size_key(cfg: ModelConfig | BaselineConfig) -> int:
    if isinstance(cfg, ModelConfig):
        return parse_size_b(cfg.size)
    match = re.search(r"-(\d+)B-", cfg.row_id)
    return int(match.group(1)) if match else 0


def row_sort_key(cfg: ModelConfig | BaselineConfig) -> tuple[str, int]:
    return (model_family_key(cfg), model_size_key(cfg))


def assemble_row_configs(
    qwen_configs: list[ModelConfig],
    ministral_configs: list[ModelConfig],
    reasoning_configs: list[BaselineConfig],
) -> list[ModelConfig | BaselineConfig]:
    qwen_da = [cfg for cfg in qwen_configs if cfg.strategy == "da"]
    qwen_cot = [cfg for cfg in qwen_configs if cfg.strategy == "cot"]
    ministral_da = [cfg for cfg in ministral_configs if cfg.strategy == "da"]
    ministral_cot = [cfg for cfg in ministral_configs if cfg.strategy == "cot"]
    non_cot = sorted([*qwen_da, *ministral_da], key=row_sort_key)
    cot_and_reasoning = sorted(
        [*qwen_cot, *reasoning_configs, *ministral_cot],
        key=row_sort_key,
    )
    return [*non_cot, *cot_and_reasoning]


def discover_reasoning_model_configs() -> list[BaselineConfig]:
    configs: list[BaselineConfig] = []
    for cfg in REASONING_MODELS:
        found = False
        for col in TABLE_COLUMNS:
            result_dir = result_dir_for_config(cfg, col)
            if result_dir.is_dir() and list(result_dir.glob("results-*.jsonl")):
                found = True
                break
        if found:
            configs.append(cfg)
    return configs


def separator_before_row_indices(
    configs: list[ModelConfig | BaselineConfig],
) -> list[int]:
    indices: list[int] = []
    has_random = (
        configs
        and isinstance(configs[0], BaselineConfig)
        and configs[0].row_id == RANDOM_GUESSER_ROW_ID
    )
    if has_random:
        indices.append(1)

    non_cot_count = sum(
        1 for cfg in configs if isinstance(cfg, ModelConfig) and cfg.strategy == "da"
    )
    cot_or_reasoning_count = sum(
        1
        for cfg in configs
        if (isinstance(cfg, ModelConfig) and cfg.strategy == "cot")
        or (isinstance(cfg, BaselineConfig) and cfg.row_id != RANDOM_GUESSER_ROW_ID)
    )
    if non_cot_count and cot_or_reasoning_count:
        offset = 1 if has_random else 0
        indices.append(offset + non_cot_count)

    return indices


def build_split_table(
    configs: list[ModelConfig | BaselineConfig],
    split: str,
    seed: str | None,
) -> SplitTable:
    values: list[list[float | None]] = []
    for cfg in configs:
        row_values: list[float | None] = []
        model_name = cfg.model_name
        for col in TABLE_COLUMNS:
            result_dir = result_dir_for_config(cfg, col)
            result_sets = load_result_sets(result_dir, seed)
            if not result_sets:
                row_values.append(None)
                continue
            row_values.append(
                cell_accuracy_averaged(
                    result_sets,
                    model_name,
                    split,
                    column_setup(col),
                    column_pattern(col),
                )
            )
        values.append(row_values)
    return SplitTable(split=split, configs=configs, values=values)


def build_random_guesser_row(
    split: str,
    seed: str | None,
) -> list[float | None]:
    row_values: list[float | None] = []
    for col in TABLE_COLUMNS:
        result_dir = (
            RESULTS_DIR
            / "pcp"
            / column_setup(col)
            / column_exp_name(col)
            / RANDOM_GUESSER_RESULT_DIR
        )
        result_sets = load_result_sets(result_dir, seed)
        if not result_sets:
            row_values.append(None)
            continue
        row_values.append(
            cell_accuracy_averaged(
                result_sets,
                RANDOM_GUESSER_MODEL_NAME,
                split,
                column_setup(col),
                column_pattern(col),
            )
        )
    return row_values


def prepend_random_guesser_row(
    split_table: SplitTable,
    seed: str | None,
) -> SplitTable:
    return SplitTable(
        split=split_table.split,
        configs=[RANDOM_GUESSER_BASELINE, *split_table.configs],
        values=[build_random_guesser_row(split_table.split, seed), *split_table.values],
    )


def warn_if_duplicate_seeds(model_prefix: str) -> None:
    sample_dir = (
        RESULTS_DIR
        / "pcp"
        / "uk"
        / "IMP-SOS"
        / model_dir_name(model_prefix, "32B", "cot")
    )
    seed_files = sorted(sample_dir.glob("results-*.jsonl"))
    if len(seed_files) < 2:
        return
    hashes = {hashlib.md5(path.read_bytes()).hexdigest() for path in seed_files}
    if len(hashes) == 1:
        print(
            "Warning: multiple seed files appear byte-identical; "
            "using a single seed file per config."
        )


def result_macro_name(row_id: str, col_key: str, split: str) -> str:
    return f"res-{row_id}-{col_key}-{SPLIT_SLUGS[split]}"


def format_accuracy(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "---"
    return str(round(value * 100))


def displayed_accuracy(value: float | None) -> int | None:
    if value is None or math.isnan(value):
        return None
    return round(value * 100)


def split_exp_id(split: str) -> str:
    return f"pcp-IMP-K-IMP-SOS-accuracy-{TABLE_SUFFIX}-{SPLIT_SLUGS[split]}"


def split_caption(split: str) -> str:
    dataset_macro = DATASETS2MACROS[split]
    return f"PredExe accuracy on {dataset_macro} programs."


def best_values_per_column(values: list[list[float | None]]) -> list[int | None]:
    bests: list[int | None] = []
    for col_idx in range(len(TABLE_COLUMNS)):
        col_values = [
            displayed
            for row in values
            if (displayed := displayed_accuracy(row[col_idx])) is not None
        ]
        bests.append(max(col_values) if col_values else None)
    return bests


def baseline_col_idx(col_idx: int) -> int | None:
    return MK_BASELINE_COL_IDX.get(col_idx)


def format_result_cell(
    macro_use: str,
    baseline_macro: str | None,
    bold: bool,
) -> str:
    if baseline_macro is None:
        if bold:
            return rf"\textbf{{{macro_use}}}"
        return macro_use
    if bold:
        return rf"\accdrobustbf{{{macro_use}}}{{{baseline_macro}}}"
    return rf"\accdrobust{{{macro_use}}}{{{baseline_macro}}}"


DELTA_HEADER_CELLS = [
    "",
    r"\scriptsize(pp)",
    r"\scriptsize(pp)",
    "",
    r"\scriptsize(pp)",
    r"\scriptsize(pp)",
]


def write_macros_file(output_dir: Path, split_tables: list[SplitTable]) -> Path:
    path = output_dir / f"macros-table-{TABLE_SUFFIX}.tex"
    lines = [
        "%% Automatically generated by: visualize_pcp_split_accuracy_tables.py",
        "",
    ]

    for split_table in split_tables:
        exp_id = split_exp_id(split_table.split)
        lines.append(
            f"\\DefMacro{{TCap-models-{exp_id}}}{{{split_caption(split_table.split)}}}"
        )
        lines.append(f"\\DefMacro{{THead-{exp_id}-models}}{{Models}}")
        for col in TABLE_COLUMNS:
            col_key = column_key(col)
            header = METRICS2MACROS.get(col_key, col_key)
            lines.append(f"\\DefMacro{{THead-{exp_id}-{col_key}}}{{{header}}}")
        for cfg in split_table.configs:
            row_macro = LLMS2MACROS.get(
                cfg.row_id, getattr(cfg, "row_label", cfg.row_id)
            )
            lines.append(f"\\DefMacro{{THead-{exp_id}-{cfg.row_id}}}{{{row_macro}}}")
        for row_idx, cfg in enumerate(split_table.configs):
            for col_idx, col in enumerate(TABLE_COLUMNS):
                col_key = column_key(col)
                macro_name = result_macro_name(cfg.row_id, col_key, split_table.split)
                lines.append(
                    f"\\DefMacro{{{macro_name}}}{{{format_accuracy(split_table.values[row_idx][col_idx])}}}"
                )
        lines.append("")

    path.write_text("\n".join(lines))
    return path


def write_table_tex(output_dir: Path, split_table: SplitTable) -> Path:
    exp_id = split_exp_id(split_table.split)
    path = output_dir / f"table-results-{exp_id}.tex"
    col_keys = [column_key(col) for col in TABLE_COLUMNS]
    bests = best_values_per_column(split_table.values)

    lines = [
        "%% Automatically generated by: visualize_pcp_split_accuracy_tables.py",
        "",
        r"\begin{table*}[t]",
        r"\begin{center}",
        r"\TableFont",
        rf"\caption{{\UseMacro{{TCap-models-{exp_id}}}\label{{tab:{exp_id}}}}}",
        r"\begin{tabular}{l  c  c  c  c  c  c  c }",
        r"\toprule",
        rf"\multirow{{3}}{{*}}{{\textbf{{\UseMacro{{THead-{exp_id}-models}}}}}}",
        r"& \multicolumn{3}{c}{\textbf{\Kos-semantics}}",
        r"& \multicolumn{3}{c}{\textbf{\Sos}}",
        r"\\",
        r"\cmidrule(lr){2-4}",
        r"\cmidrule(lr){5-7}",
    ]

    header_cells = [
        rf"\textbf{{\UseMacro{{THead-{exp_id}-{col_key}}}}}" for col_key in col_keys
    ]
    lines.append(" & " + " & ".join(header_cells) + r" \\")
    lines.append(" & " + " & ".join(DELTA_HEADER_CELLS) + r" \\")
    lines.append(r"\midrule")

    separator_before = set(separator_before_row_indices(split_table.configs))
    for row_idx, cfg in enumerate(split_table.configs):
        if row_idx in separator_before:
            lines.append(r"\midrule")
        cells = [rf"\UseMacro{{THead-{exp_id}-{cfg.row_id}}}"]
        for col_idx, col_key in enumerate(col_keys):
            value = split_table.values[row_idx][col_idx]
            displayed = displayed_accuracy(value)
            macro_name = result_macro_name(cfg.row_id, col_key, split_table.split)
            macro_use = rf"\UseMacro{{{macro_name}}}"
            baseline_col = baseline_col_idx(col_idx)
            baseline_macro = None
            if baseline_col is not None:
                baseline_macro_name = result_macro_name(
                    cfg.row_id,
                    col_keys[baseline_col],
                    split_table.split,
                )
                baseline_macro = rf"\UseMacro{{{baseline_macro_name}}}"
            cells.append(
                format_result_cell(
                    macro_use,
                    baseline_macro,
                    displayed is not None and displayed == bests[col_idx],
                )
            )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def write_combined_tex(output_dir: Path, split_tables: list[SplitTable]) -> Path:
    path = output_dir / f"table-results-{TABLE_SUFFIX}.tex"
    lines = [
        "%% Automatically generated by: visualize_pcp_split_accuracy_tables.py",
        "%% Combined view of all three dataset splits.",
        "%% Macros are loaded from macros.tex via tables/macros-table-qwen-coder-by-split.tex",
        "",
    ]
    for split_table in split_tables:
        exp_id = split_exp_id(split_table.split)
        lines.append(rf"\input{{tables/table-results-{exp_id}}}")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def print_table(split_table: SplitTable) -> None:
    split = split_table.split
    print(f"\n=== {SPLIT_LABELS[split]} ===")
    col_labels = [
        METRICS2MACROS.get(column_key(col), column_key(col)) for col in TABLE_COLUMNS
    ]
    header = f"{'Model':>12} | " + " | ".join(f"{label:>9}" for label in col_labels)
    print(header)
    print("-" * len(header))
    for cfg, row in zip(split_table.configs, split_table.values):
        display = [format_accuracy(value) for value in row]
        row_label = (
            cfg.row_label
            if isinstance(cfg, (ModelConfig, BaselineConfig))
            else cfg.row_id
        )
        print(f"{row_label:>12} | " + " | ".join(f"{cell:>9}" for cell in display))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PredExe split accuracy LaTeX tables."
    )
    parser.add_argument(
        "--model-prefix",
        default=DEFAULT_MODEL_PREFIX,
        help=f"Model prefix (default: {DEFAULT_MODEL_PREFIX})",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Use only this results seed (default: average all results-*.jsonl files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for LaTeX tables (default: papers/lmpl26/tables)",
    )
    args = parser.parse_args()

    qwen_configs = discover_model_configs(args.model_prefix)
    ministral_configs = discover_ministral_model_configs()
    reasoning_configs = discover_reasoning_model_configs()
    row_configs = assemble_row_configs(
        qwen_configs,
        ministral_configs,
        reasoning_configs,
    )
    if not row_configs:
        print("No result directories found.")
        raise SystemExit(1)

    warn_if_duplicate_seeds(args.model_prefix)
    output_dir = args.output_dir or TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    split_tables = [
        prepend_random_guesser_row(
            build_split_table(row_configs, split, args.seed),
            args.seed,
        )
        for split in DATASET_SPLITS
    ]

    for split_table in split_tables:
        print_table(split_table)

    macros_path = write_macros_file(output_dir, split_tables)
    table_paths = [
        write_table_tex(output_dir, split_table) for split_table in split_tables
    ]
    combined_path = write_combined_tex(output_dir, split_tables)

    print(f"\nSaved macros: {macros_path}")
    for path in table_paths:
        print(f"Saved table: {path}")
    print(f"Saved combined: {combined_path}")
    print(
        "\nTo include in the paper, add to macros.tex:\n"
        f"  \\input{{tables/macros-table-{TABLE_SUFFIX}.tex}}\n"
        "and to results.tex:\n"
        f"  \\input{{tables/table-results-{TABLE_SUFFIX}}}"
    )


if __name__ == "__main__":
    main()
