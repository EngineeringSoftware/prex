"""
Shared PredExe (PCP) semantic-error-type helpers.

Extracted so paper-facing scripts (radar, qualitative) do not depend on the
full confusion-matrix / recall-heatmap CLIs.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from visualize_pcp_split_accuracy_tables import (  # noqa: E402
    analyze_pcp_records,
    column_exp_name,
    column_pattern,
    column_setup,
    filter_records,
)

ERROR_TYPE_LABELS = [
    "No error",
    "Use before declare",
    "Divide by 0",
    "Modulo by 0",
    "Continue outside loop",
    "Break outside loop",
]

ERROR_TYPE_SHORT = {
    "No error": "No error",
    "Use before declare": "Use bef. decl.",
    "Divide by 0": "Div by 0",
    "Modulo by 0": "Mod by 0",
    "Continue outside loop": "Cont. o.loop",
    "Break outside loop": "Break o.loop",
}

INVALID_ERROR_TYPES = ERROR_TYPE_LABELS[1:]


def pcp_rule_to_label_map(semantics: str) -> dict[str, str]:
    if semantics == "SOS":
        return {
            "None": "No error",
            "Rule 2": "Use before declare",
            "Rule 6": "Use before declare",
            "Rule 19": "Divide by 0",
            "Rule 23": "Modulo by 0",
            "Rule 76": "Continue outside loop",
            "Rule 73": "Break outside loop",
        }
    if semantics == "K":
        return {
            "None": "No error",
            "Rule 2": "Use before declare",
            "Rule 7": "Divide by 0",
            "Rule 9": "Modulo by 0",
            "Rule 31": "Continue outside loop",
            "Rule 34": "Break outside loop",
        }
    raise ValueError(f"Unknown semantics: {semantics!r}")


def canonical_error_rule_numbers(semantics: str) -> list[tuple[int, str]]:
    rule_map = pcp_rule_to_label_map(semantics)
    numbers: dict[str, list[int]] = {}
    for rule, label in rule_map.items():
        if rule == "None":
            continue
        numbers.setdefault(label, []).append(int(rule.split()[1]))
    return [(n, label) for label in INVALID_ERROR_TYPES for n in numbers[label]]


def snap_rule_number(rule_number: int, canonical_rules: list[tuple[int, str]]) -> str:
    return min(canonical_rules, key=lambda item: abs(item[0] - rule_number))[1]


def rule_to_error_type(
    rule: str,
    rule_map: dict[str, str],
    canonical_rules: list[tuple[int, str]],
) -> str:
    if rule in rule_map:
        return rule_map[rule]
    match = re.search(r"Rule\s+(\d+)", rule or "")
    if not match:
        return "No error" if not rule or rule == "None" else snap_rule_number(-1, canonical_rules)
    rule_number = int(match.group(1))
    key = f"Rule {rule_number}"
    if key in rule_map:
        return rule_map[key]
    return snap_rule_number(rule_number, canonical_rules)


def semantics_from_col(col: tuple) -> str:
    exp_name = column_exp_name(col)
    return "K" if exp_name == "IMP-K" else "SOS"


def per_seed_error_type_recall(
    results: list[dict],
    model_name: str,
    split: str,
    col: tuple,
) -> list[float]:
    setup = column_setup(col)
    pattern = column_pattern(col)
    semantics = semantics_from_col(col)
    filtered = filter_records(results, split, setup, pattern)
    if not filtered:
        return [float("nan")] * len(INVALID_ERROR_TYPES)

    trace = analyze_pcp_records(filtered, model_name)
    rule_map = pcp_rule_to_label_map(semantics)
    canonical_rules = canonical_error_rule_numbers(semantics)

    recalls: list[float] = []
    for error_type in INVALID_ERROR_TYPES:
        total = 0
        correct = 0
        for y_true, y_pred, rule_true, rule_pred in zip(
            trace.y_true,
            trace.y_pred,
            trace.rule_true,
            trace.rule_pred,
        ):
            if y_true != 0:
                continue
            true_label = rule_to_error_type(rule_true, rule_map, canonical_rules)
            if true_label != error_type:
                continue
            total += 1
            if y_pred != 0:
                continue
            pred_label = rule_to_error_type(rule_pred, rule_map, canonical_rules)
            if pred_label == error_type:
                correct += 1
        recalls.append(100.0 * correct / total if total else float("nan"))
    return recalls


def averaged_error_type_recall(
    result_sets: list[list[dict]],
    model_name: str,
    split: str,
    col: tuple,
) -> np.ndarray:
    per_seed: list[list[float]] = []
    for results in result_sets:
        recalls = per_seed_error_type_recall(results, model_name, split, col)
        if any(not np.isnan(value) for value in recalls):
            per_seed.append(recalls)
    if not per_seed:
        return np.full(len(INVALID_ERROR_TYPES), np.nan)

    averaged: list[float] = []
    for idx in range(len(INVALID_ERROR_TYPES)):
        values = [recalls[idx] for recalls in per_seed if not np.isnan(recalls[idx])]
        averaged.append(statistics.mean(values) if values else float("nan"))
    return np.array(averaged)


# Alias used by visualize_pcp_error_type_radar.py
averaged_error_type_accuracy = averaged_error_type_recall
