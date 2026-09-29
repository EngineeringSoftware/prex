"""
Generate numbers-pcp-split-drops.tex from PredExe split accuracy tables.

Computes performance drops from human-written to llm-translated / fuzzer-generated
splits (six columns per model: uk-K, mk-K swap/obf, uk-SOS, mk-SOS swap/obf).

Usage:
  python scripts/generate_pcp_split_drops.py
  python scripts/generate_pcp_split_drops.py --seed 42
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from visualize_pcp_split_accuracy_tables import (  # noqa: E402
    DEFAULT_MODEL_PREFIX,
    assemble_row_configs,
    build_split_table,
    discover_ministral_model_configs,
    discover_model_configs,
    discover_reasoning_model_configs,
)

OUTPUT = REPO_ROOT / "papers" / "lmpl26" / "tables" / "numbers-pcp-split-drops.tex"

CAPABLE_HW_THRESHOLD = 45

TOP3_ROW_IDS = (
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
    "mistralai-Ministral-3-14B-Instruct-2512-BF16-cot",
    "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
)

FEATURED_ROW_IDS = (
    *TOP3_ROW_IDS,
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
    "Qwen-Qwen2.5-Coder-14B-Instruct-da",
)

ROW_SLUGS = {
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da": "dpsk32",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da": "dpsk14",
    "mistralai-Ministral-3-14B-Instruct-2512-BF16-cot": "min14cot",
    "Qwen-Qwen2.5-Coder-32B-Instruct-cot": "qwen32cot",
    "Qwen-Qwen2.5-Coder-14B-Instruct-da": "qwen14da",
}

# Column index -> macro slug fragment (e.g. uk-k, mk-sos-obf)
COLUMN_SLUGS = (
    "uk-k",
    "mk-k-swap",
    "mk-k-obf",
    "uk-sos",
    "mk-sos-swap",
    "mk-sos-obf",
)

OTHER_SPLITS = (
    ("fuzzer_generated", "fuzzgen"),
    ("synthetic_cpp", "llmtrans"),
)


def pp_drop(baseline: float | None, other: float | None) -> float | None:
    if baseline is None or other is None:
        return None
    return (baseline - other) * 100


def rel_drop_pct(baseline: float | None, other: float | None) -> float | None:
    if baseline is None or other is None or baseline == 0:
        return None
    return (baseline - other) / baseline * 100


def fmt_num(value: float, decimals: int = 1) -> str:
    rounded = round(value, decimals)
    if decimals == 0 or rounded == int(rounded):
        return str(int(round(rounded)))
    return f"{rounded:.{decimals}f}"


def fmt_pct(value: float) -> str:
    return f"{fmt_num(value)}\\%"


def mean_hw(values: list[float | None]) -> float | None:
    accs = [v for v in values if v is not None]
    return statistics.mean(accs) * 100 if accs else None


def split_drops(
    hw_values: list[float | None],
    other_values: list[float | None],
) -> tuple[list[float | None], list[float | None]]:
    pp_values: list[float | None] = []
    pct_values: list[float | None] = []
    for hw, other in zip(hw_values, other_values):
        pp_values.append(pp_drop(hw, other))
        pct_values.append(rel_drop_pct(hw, other))
    return pp_values, pct_values


def mean_of(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return statistics.mean(nums) if nums else None


def max_of(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return max(nums) if nums else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numbers-pcp-split-drops.tex from split accuracy tables."
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Use only this results seed (default: average all results-*.jsonl files)",
    )
    args = parser.parse_args()

    qwen = discover_model_configs(DEFAULT_MODEL_PREFIX)
    ministral = discover_ministral_model_configs()
    reasoning = discover_reasoning_model_configs()
    configs = [
        cfg
        for cfg in assemble_row_configs(qwen, ministral, reasoning)
        if cfg.row_id != "random-guesser-dataset-distribution-da"
    ]

    hw = build_split_table(configs, "human_written", seed=args.seed)
    llmtrans = build_split_table(configs, "synthetic_cpp", seed=args.seed)
    fuzzgen = build_split_table(configs, "fuzzer_generated", seed=args.seed)

    per_row_hw: dict[str, list[float | None]] = {}
    per_row_llmtrans: dict[str, list[float | None]] = {}
    per_row_fuzzgen: dict[str, list[float | None]] = {}

    for idx, cfg in enumerate(hw.configs):
        per_row_hw[cfg.row_id] = hw.values[idx]
        per_row_llmtrans[cfg.row_id] = llmtrans.values[idx]
        per_row_fuzzgen[cfg.row_id] = fuzzgen.values[idx]

    capable_mean_fuzzgen: list[float] = []
    top3_max_fuzzgen_pp = -1.0
    top3_max_fuzzgen_pct = -1.0
    overall_max_fuzzgen_pp = -1.0
    overall_max_fuzzgen_pct = -1.0

    for cfg in hw.configs:
        row_id = cfg.row_id
        hw_mean = mean_hw(per_row_hw[row_id])
        fuzzgen_pp, fuzzgen_pct = split_drops(per_row_hw[row_id], per_row_fuzzgen[row_id])
        row_mean_fuzzgen = mean_of(fuzzgen_pp)
        if hw_mean is not None and hw_mean >= CAPABLE_HW_THRESHOLD and row_mean_fuzzgen is not None:
            capable_mean_fuzzgen.append(row_mean_fuzzgen)
        for pp, pct in zip(fuzzgen_pp, fuzzgen_pct):
            if pp is not None and pp > overall_max_fuzzgen_pp:
                overall_max_fuzzgen_pp = pp
            if pct is not None and pct > overall_max_fuzzgen_pct:
                overall_max_fuzzgen_pct = pct
        if row_id in TOP3_ROW_IDS:
            for pp, pct in zip(fuzzgen_pp, fuzzgen_pct):
                if pp is not None and pp > top3_max_fuzzgen_pp:
                    top3_max_fuzzgen_pp = pp
                if pct is not None and pct > top3_max_fuzzgen_pct:
                    top3_max_fuzzgen_pct = pct

    top3_mean_fuzzgen = [
        mean_of(split_drops(per_row_hw[row_id], per_row_fuzzgen[row_id])[0]) or 0.0
        for row_id in TOP3_ROW_IDS
    ]
    top3_mean_fuzzgen_avg = statistics.mean(top3_mean_fuzzgen)
    dpsk14_mean_fuzzgen = mean_of(
        split_drops(
            per_row_hw["deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da"],
            per_row_fuzzgen["deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da"],
        )[0]
    ) or 0.0

    lines = [
        "%% Automatically generated by: generate_pcp_split_drops.py",
        "%% Performance drops from human-written to other splits.",
        "",
        f"\\DefMacro{{pcp-split-drop-capable-hw-threshold}}{{{CAPABLE_HW_THRESHOLD}}}",
        f"\\DefMacro{{pcp-split-drop-capable-mean-fuzzgen-min-pp}}"
        f"{{{fmt_num(min(capable_mean_fuzzgen), 0)}}}",
        f"\\DefMacro{{pcp-split-drop-capable-mean-fuzzgen-max-pp}}"
        f"{{{fmt_num(max(capable_mean_fuzzgen), 0)}}}",
        f"\\DefMacro{{pcp-split-drop-capable-mean-fuzzgen-median-pp}}"
        f"{{{fmt_num(statistics.median(capable_mean_fuzzgen), 0)}}}",
        "",
    ]

    for row_id in FEATURED_ROW_IDS:
        slug = ROW_SLUGS[row_id]
        hw_values = per_row_hw[row_id]
        lines.append(
            f"\\DefMacro{{pcp-split-drop-{slug}-mean-hw}}{{{fmt_num(mean_hw(hw_values) or 0)}}}"
        )
        for other_split, other_slug in OTHER_SPLITS:
            other_values = (
                per_row_fuzzgen[row_id]
                if other_split == "fuzzer_generated"
                else per_row_llmtrans[row_id]
            )
            pp_values, pct_values = split_drops(hw_values, other_values)
            for col_slug, pp, pct in zip(COLUMN_SLUGS, pp_values, pct_values):
                lines.append(
                    f"\\DefMacro{{pcp-split-drop-{slug}-{col_slug}-{other_slug}-pp}}"
                    f"{{{fmt_num(pp or 0, 0)}}}"
                )
                lines.append(
                    f"\\DefMacro{{pcp-split-drop-{slug}-{col_slug}-{other_slug}-pct}}"
                    f"{{{fmt_pct(pct or 0)}}}"
                )
        lines.append("")

        llmtrans_pp, _ = split_drops(hw_values, per_row_llmtrans[row_id])
        llmtrans_pct = [
            rel_drop_pct(hw, other)
            for hw, other in zip(hw_values, per_row_llmtrans[row_id])
        ]
        fuzzgen_pp, _ = split_drops(hw_values, per_row_fuzzgen[row_id])
        fuzzgen_pct = [
            rel_drop_pct(hw, other)
            for hw, other in zip(hw_values, per_row_fuzzgen[row_id])
        ]

        lines.extend(
            [
                f"\\DefMacro{{pcp-split-drop-{slug}-mean-llmtrans-pp}}"
                f"{{{fmt_num(mean_of(llmtrans_pp) or 0)}}}",
                f"\\DefMacro{{pcp-split-drop-{slug}-mean-fuzzgen-pp}}"
                f"{{{fmt_num(mean_of(fuzzgen_pp) or 0)}}}",
                f"\\DefMacro{{pcp-split-drop-{slug}-mean-fuzzgen-pct}}"
                f"{{{fmt_pct(mean_of(fuzzgen_pct) or 0)}}}",
                f"\\DefMacro{{pcp-split-drop-{slug}-max-fuzzgen-pp}}"
                f"{{{fmt_num(max_of(fuzzgen_pp) or 0, 0)}}}",
                f"\\DefMacro{{pcp-split-drop-{slug}-max-fuzzgen-pct}}"
                f"{{{fmt_pct(max_of(fuzzgen_pct) or 0)}}}",
                f"\\DefMacro{{pcp-split-drop-{slug}-mean-llmtrans-pct}}"
                f"{{{fmt_pct(mean_of(llmtrans_pct) or 0)}}}",
                f"\\DefMacro{{pcp-split-drop-{slug}-max-llmtrans-pp}}"
                f"{{{fmt_num(max_of(llmtrans_pp) or 0, 0)}}}",
                f"\\DefMacro{{pcp-split-drop-{slug}-max-llmtrans-pct}}"
                f"{{{fmt_pct(max_of(llmtrans_pct) or 0)}}}",
                "",
            ]
        )

    lines.extend(
        [
            f"\\DefMacro{{pcp-split-drop-overall-max-fuzzgen-pp}}"
            f"{{{fmt_num(overall_max_fuzzgen_pp, 0)}}}",
            f"\\DefMacro{{pcp-split-drop-overall-max-fuzzgen-pct}}"
            f"{{{fmt_pct(overall_max_fuzzgen_pct)}}}",
            f"\\DefMacro{{pcp-split-drop-top3-max-fuzzgen-pp}}"
            f"{{{fmt_num(top3_max_fuzzgen_pp, 0)}}}",
            f"\\DefMacro{{pcp-split-drop-top3-max-fuzzgen-pct}}"
            f"{{{fmt_pct(top3_max_fuzzgen_pct)}}}",
            f"\\DefMacro{{pcp-split-drop-dpsk14-vs-top3-mean-fuzzgen-gap-pp}}"
            f"{{{fmt_num(top3_mean_fuzzgen_avg - dpsk14_mean_fuzzgen)}}}",
        ]
    )

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
