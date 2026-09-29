"""
Generate numbers-pcp-semantic-drops.tex from PredExe split accuracy tables.

Computes performance drops from standardSem (uk) to KeywordSwap / KeywordObf,
mirroring the split-drop statistics in numbers-pcp-split-drops.tex.

Usage:
  python scripts/generate_pcp_semantic_drops.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from visualize_pcp_split_accuracy_tables import (  # noqa: E402
    DEFAULT_MODEL_PREFIX,
    TABLE_COLUMNS,
    assemble_row_configs,
    build_split_table,
    discover_ministral_model_configs,
    discover_model_configs,
    discover_reasoning_model_configs,
)

OUTPUT = REPO_ROOT / "papers" / "lmpl26" / "tables" / "numbers-pcp-semantic-drops.tex"

CAPABLE_HW_THRESHOLD = 45

TOP3_ROW_IDS = (
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
    "mistralai-Ministral-3-14B-Instruct-2512-BF16-cot",
    "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
)

TOP3_SLUGS = {
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da": "dpsk32",
    "mistralai-Ministral-3-14B-Instruct-2512-BF16-cot": "min14cot",
    "Qwen-Qwen2.5-Coder-32B-Instruct-cot": "qwen32cot",
}

# uk-K, swap-K, obf-K, uk-SOS, swap-SOS, obf-SOS
SEMANTIC_PAIRS = (
    ("k", "swap", 0, 1),
    ("k", "obf", 0, 2),
    ("sos", "swap", 3, 4),
    ("sos", "obf", 3, 5),
)

COT_PAIRS = (
    ("Qwen-Qwen2.5-Coder-14B-Instruct-da", "Qwen-Qwen2.5-Coder-14B-Instruct-cot", "qwen14"),
    ("Qwen-Qwen2.5-Coder-32B-Instruct-da", "Qwen-Qwen2.5-Coder-32B-Instruct-cot", "qwen32"),
)


def pp_drop(uk: float | None, mutated: float | None) -> float | None:
    if uk is None or mutated is None:
        return None
    return (uk - mutated) * 100


def fmt_num(value: float, decimals: int = 1) -> str:
    rounded = round(value, decimals)
    if decimals == 0 or rounded == int(rounded):
        return str(int(round(rounded)))
    return f"{rounded:.{decimals}f}"


def row_drops(values: list[float | None]) -> dict[str, float | None]:
    drops: dict[str, float | None] = {}
    for sem, mut, uk_idx, mut_idx in SEMANTIC_PAIRS:
        drops[f"{sem}-{mut}"] = pp_drop(values[uk_idx], values[mut_idx])
    return drops


def mean_drop(values: list[float | None]) -> float | None:
    drops = [d for d in row_drops(values).values() if d is not None]
    return statistics.mean(drops) if drops else None


def mean_uk(values: list[float | None]) -> float | None:
    accs = [values[i] for i in (0, 3) if values[i] is not None]
    return statistics.mean(accs) if accs else None


def main() -> None:
    qwen = discover_model_configs(DEFAULT_MODEL_PREFIX)
    ministral = discover_ministral_model_configs()
    reasoning = discover_reasoning_model_configs()
    configs = [
        cfg
        for cfg in assemble_row_configs(qwen, ministral, reasoning)
        if cfg.row_id != "random-guesser-dataset-distribution-da"
    ]

    hw = build_split_table(configs, "human_written", seed=None)
    fg = build_split_table(configs, "fuzzer_generated", seed=None)

    all_swap: list[float] = []
    all_obf: list[float] = []
    capable_mean: list[float] = []
    capable_swap: list[float] = []
    capable_obf: list[float] = []
    capable_count = 0

    per_row_hw: dict[str, list[float | None]] = {}
    per_row_fg: dict[str, list[float | None]] = {}

    for idx, cfg in enumerate(hw.configs):
        per_row_hw[cfg.row_id] = hw.values[idx]
        per_row_fg[cfg.row_id] = fg.values[idx]
        drops = row_drops(hw.values[idx])
        for key, drop in drops.items():
            if drop is None:
                continue
            if "swap" in key:
                all_swap.append(drop)
            else:
                all_obf.append(drop)

        uk_mean = mean_uk(hw.values[idx])
        if uk_mean is not None and uk_mean * 100 >= CAPABLE_HW_THRESHOLD:
            capable_count += 1
            row_mean = mean_drop(hw.values[idx])
            if row_mean is not None:
                capable_mean.append(row_mean)
            for key, drop in drops.items():
                if drop is None:
                    continue
                if "swap" in key:
                    capable_swap.append(drop)
                else:
                    capable_obf.append(drop)

    fg_all_swap: list[float] = []
    fg_all_obf: list[float] = []
    for values in per_row_fg.values():
        drops = row_drops(values)
        for key, drop in drops.items():
            if drop is None:
                continue
            if "swap" in key:
                fg_all_swap.append(drop)
            else:
                fg_all_obf.append(drop)

    max_hw = (-1.0, "", "", 0.0, 0.0)
    for cfg in hw.configs:
        drops = row_drops(per_row_hw[cfg.row_id])
        for key, drop in drops.items():
            if drop is not None and drop > max_hw[0]:
                uk_idx = 0 if key.startswith("k") else 3
                mut_idx = {"k-swap": 1, "k-obf": 2, "sos-swap": 4, "sos-obf": 5}[key]
                max_hw = (
                    drop,
                    cfg.row_id,
                    key,
                    (per_row_hw[cfg.row_id][uk_idx] or 0) * 100,
                    (per_row_hw[cfg.row_id][mut_idx] or 0) * 100,
                )

    lines = [
        "%% Automatically generated by: generate_pcp_semantic_drops.py",
        "%% Performance drops from standardSem (uk) to KeywordSwap / KeywordObf.",
        "",
        f"\\DefMacro{{pcp-sem-drop-capable-count}}{{{capable_count}}}",
        f"\\DefMacro{{pcp-sem-drop-all-swap-median-pp}}{{{fmt_num(statistics.median(all_swap), 0)}}}",
        f"\\DefMacro{{pcp-sem-drop-all-swap-mean-pp}}{{{fmt_num(statistics.mean(all_swap))}}}",
        f"\\DefMacro{{pcp-sem-drop-all-obf-median-pp}}{{{fmt_num(statistics.median(all_obf), 0)}}}",
        f"\\DefMacro{{pcp-sem-drop-all-obf-mean-pp}}{{{fmt_num(statistics.mean(all_obf))}}}",
        f"\\DefMacro{{pcp-sem-drop-hw-obf-minus-swap-mean-pp}}{{{fmt_num(statistics.mean(all_obf) - statistics.mean(all_swap))}}}",
        "",
        f"\\DefMacro{{pcp-sem-drop-capable-mean-min-pp}}{{{fmt_num(min(capable_mean), 0)}}}",
        f"\\DefMacro{{pcp-sem-drop-capable-mean-max-pp}}{{{fmt_num(max(capable_mean), 0)}}}",
        f"\\DefMacro{{pcp-sem-drop-capable-mean-median-pp}}{{{fmt_num(statistics.median(capable_mean), 0)}}}",
        f"\\DefMacro{{pcp-sem-drop-capable-mean-pp}}{{{fmt_num(statistics.mean(capable_mean))}}}",
        f"\\DefMacro{{pcp-sem-drop-capable-swap-median-pp}}{{{fmt_num(statistics.median(capable_swap), 0)}}}",
        f"\\DefMacro{{pcp-sem-drop-capable-swap-mean-pp}}{{{fmt_num(statistics.mean(capable_swap))}}}",
        f"\\DefMacro{{pcp-sem-drop-capable-obf-median-pp}}{{{fmt_num(statistics.median(capable_obf), 0)}}}",
        f"\\DefMacro{{pcp-sem-drop-capable-obf-mean-pp}}{{{fmt_num(statistics.mean(capable_obf))}}}",
        "",
    ]

    top3_hw_means: list[float] = []
    for row_id in TOP3_ROW_IDS:
        slug = TOP3_SLUGS[row_id]
        values = per_row_hw[row_id]
        drops = row_drops(values)
        row_mean = mean_drop(values)
        swap_mean = statistics.mean([drops["k-swap"], drops["sos-swap"]])
        obf_mean = statistics.mean([drops["k-obf"], drops["sos-obf"]])
        top3_hw_means.append(row_mean or 0.0)
        lines.extend(
            [
                f"\\DefMacro{{pcp-sem-drop-{slug}-mean-hw-pp}}{{{fmt_num(row_mean or 0)}}}",
                f"\\DefMacro{{pcp-sem-drop-{slug}-mean-swap-hw-pp}}{{{fmt_num(swap_mean)}}}",
                f"\\DefMacro{{pcp-sem-drop-{slug}-mean-obf-hw-pp}}{{{fmt_num(obf_mean)}}}",
                f"\\DefMacro{{pcp-sem-drop-{slug}-k-obf-hw-pp}}{{{fmt_num(drops['k-obf'] or 0, 0)}}}",
                f"\\DefMacro{{pcp-sem-drop-{slug}-sos-obf-hw-pp}}{{{fmt_num(drops['sos-obf'] or 0, 0)}}}",
            ]
        )

    lines.extend(
        [
            f"\\DefMacro{{pcp-sem-drop-top3-mean-hw-min-pp}}{{{fmt_num(min(top3_hw_means))}}}",
            f"\\DefMacro{{pcp-sem-drop-top3-mean-hw-max-pp}}{{{fmt_num(max(top3_hw_means))}}}",
            "",
            f"\\DefMacro{{pcp-sem-drop-max-hw-pp}}{{{fmt_num(max_hw[0], 0)}}}",
            f"%% max-hw case: {max_hw[1]} {max_hw[2]} "
            f"({fmt_num(max_hw[3], 0)} -> {fmt_num(max_hw[4], 0)})",
            "",
            f"\\DefMacro{{pcp-sem-drop-qwen32cot-mean-fuzzgen-pp}}{{{fmt_num(mean_drop(per_row_fg[TOP3_ROW_IDS[2]]) or 0)}}}",
            f"\\DefMacro{{pcp-sem-drop-fuzzgen-swap-mean-pp}}{{{fmt_num(statistics.mean(fg_all_swap))}}}",
            f"\\DefMacro{{pcp-sem-drop-fuzzgen-obf-mean-pp}}{{{fmt_num(statistics.mean(fg_all_obf))}}}",
            f"\\DefMacro{{pcp-sem-drop-fuzzgen-obf-minus-swap-mean-pp}}{{{fmt_num(statistics.mean(fg_all_obf) - statistics.mean(fg_all_swap))}}}",
            "",
        ]
    )

    for da_id, cot_id, slug in COT_PAIRS:
        da_mean = mean_drop(per_row_hw[da_id])
        cot_mean = mean_drop(per_row_hw[cot_id])
        lines.append(
            f"\\DefMacro{{pcp-sem-drop-{slug}da-mean-hw-pp}}{{{fmt_num(da_mean or 0)}}}"
        )
        if cot_id not in TOP3_ROW_IDS:
            lines.append(
                f"\\DefMacro{{pcp-sem-drop-{slug}cot-mean-hw-pp}}{{{fmt_num(cot_mean or 0)}}}"
            )
        lines.append(
            f"\\DefMacro{{pcp-sem-drop-{slug}cot-vs-da-mean-hw-gap-pp}}"
            f"{{{fmt_num((da_mean or 0) - (cot_mean or 0))}}}"
        )

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
