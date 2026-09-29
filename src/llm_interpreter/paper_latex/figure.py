import json
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import copy
from llm_interpreter.macros import Macros
from llm_interpreter.results.results_analysis import PCPResults, OPResults
from llm_interpreter.language import IMP, Language
from llm_interpreter.language.metrics import HalsteadMetric, ExtendedCyclomaticMetric
from llm_interpreter.utils import (
    read_from_json_file,
)


from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import OrderedDict, defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Sequence, Tuple, Optional, Dict
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import iqr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
import statsmodels.api as sm
from scipy.spatial.distance import squareform
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from llm_interpreter.results.results_analysis import (
    gen_op_regression_coefficients,
    prepare_dataset_and_op_output_dfs,
    dataset_code_complexity_metrics_to_df
)

from datasets import load_dataset

llms2macros = {
    "meta-llama-Llama-3.3-70B-Instruct-da": "Llama-3.3 70B",
    "meta-llama-Llama-3.3-70B-Instruct-cot": "Llama-3.3 70B\nCoT",
    "Qwen-Qwen2.5-Coder-3B-Instruct-da": "Qwen2.5-Inst\n3B",
    "Qwen-Qwen2.5-Coder-3B-Instruct-cot": "Qwen2.5-Inst\n3B CoT",
    "Qwen-Qwen2.5-Coder-14B-Instruct-da": "Qwen2.5-Inst\n14B",
    "Qwen-Qwen2.5-Coder-14B-Instruct-cot": "Qwen2.5-Inst\n14B CoT",
    "Qwen-Qwen2.5-Coder-32B-Instruct-da": "Qwen2.5-Inst\n32B",
    "Qwen-Qwen2.5-Coder-32B-Instruct-cot": "Qwen2.5-Inst\n32B CoT",
    "gpt-4o-mini-da": "GPT-4o-mini",
    "gpt-4o-mini-cot": "GPT-4o-mini\nCoT",
    "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da": "DeepSeek\nLlama 70B",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da": "DeepSeek\nQwen 14B",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da": "DeepSeek\nQwen 32B",
    "Qwen-QwQ-32B-da": "QwQ 32B",
    "o3-mini-da": "o3-mini",
    "gpt-5-mini-da": "GPT-5-mini",
    "gemini-2.5-pro-da": "Gemini-2.5\npro",
}

llms2macrosfull = {
    "meta-llama-Llama-3.3-70B-Instruct-da": "Llama-3.3 70B",
    "meta-llama-Llama-3.3-70B-Instruct-cot": "Llama-3.3 70B-CoT",
    "Qwen-Qwen2.5-Coder-3B-Instruct-da": "Qwen2.5-Instruct 3B",
    "Qwen-Qwen2.5-Coder-3B-Instruct-cot": "Qwen2.5-Instruct 3B-CoT",
    "Qwen-Qwen2.5-Coder-14B-Instruct-da": "Qwen2.5-Instruct 14B",
    "Qwen-Qwen2.5-Coder-14B-Instruct-cot": "Qwen2.5-Instruct 14B-CoT",
    "Qwen-Qwen2.5-Coder-32B-Instruct-da": "Qwen2.5-Instruct 32B",
    "Qwen-Qwen2.5-Coder-32B-Instruct-cot": "Qwen2.5-Instruct 32B-CoT",
    "gpt-4o-mini-da": "GPT-4o-mini",
    "gpt-4o-mini-cot": "GPT-4o-mini-CoT",
    "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da": "DeepSeek-Llama 70B",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da": "DeepSeek-Qwen 14B",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da": "DeepSeek-Qwen 32B",
    "Qwen-QwQ-32B-da": "QwQ 32B",
    "o3-mini-da": "o3-mini",
    "gpt-5-mini-da": "GPT-5-mini",
    "gemini-2.5-pro-da": "Gemini-2.5-pro",
}

# --- Configure Matplotlib to use LaTeX ---
plt.rcParams['text.usetex'] = True
# Set default font to serif, which is often Computer Modern by default in LaTeX
plt.rcParams['font.family'] = 'times'
# Add amsmath (and other packages if needed) to the LaTeX preamble
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' 
# ----------------------------------------

metrics2macros: dict = {
    'CC': r"$\Omega_{\mathsf{CC}}$",
    'LOC': r"$\Omega_\mathsf{Loc}$",
    'MaxNestedIf': r"$\Omega_\mathsf{If}$",
    'MaxNestedLoop': r"$\Omega_\mathsf{Loop}$",
    'MaxTakenIf': r"$\hat\Omega_\mathsf{If}$",
    'MaxTakenLoop': r"$\hat\Omega_\mathsf{Loop}$",
    'NumAssignments': r"$\hat\Omega_\mathsf{Assign}$",
    'TraceLength': r"$\hat\Omega_\mathsf{Trace}$",
    'Volume': r"$\Omega_\mathsf{Vol}$",
    'Vocabulary': r"$\Omega_\mathsf{Voc}$",
    'DepDegree': r"$\Omega_\mathsf{DD}$"
}

def radar_factory(num_vars, frame="polygon"):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError(f"Unknown frame: {frame}")

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError(f"Unknown frame: {frame}")

    register_projection(RadarAxes)
    return theta


def load_json_data(file_name: str) -> dict:
    content: dict = {}
    with open(file_name, "r") as file:
        content = json.load(file)
    # htiw
    return content


# fed


def gen_pcp_radar_chart(
    mutation_type: str, semantic_type: str, rules: dict, name: str, legend: bool = False
):
    data = load_json_data(
        f"{Macros.results_dir}/metrics-pcp-{mutation_type}-IMP-{semantic_type}.json"
    )
    model_names = list(data.keys())
    model_axis_names = [llms2macros[model] for model in model_names]
    num_vars = len(model_names)
    theta = radar_factory(num_vars, frame="polygon")

    plot_data = []
    for rule in rules.keys():
        rule_values = [data[model].get(rule, 0.0) for model in model_names]
        plot_data.append(rule_values)
    # rof

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="radar"))
    colors = ["b", "r", "g", "steelblue", "orange", "brown"]

    for values, rule, color in zip(plot_data, rules.keys(), colors):
        ax.plot(theta, values, label=rules[rule], color=color)
        ax.fill(theta, values, facecolor=color, alpha=0.2)

    ax.set_varlabels(model_axis_names)
    for label, angle in zip(ax.get_xticklabels(), theta):
        angle_deg = np.degrees(angle)
        if angle_deg == 0.0:
            label.set_horizontalalignment("center")
        elif angle_deg == 180.0:
            label.set_verticalalignment("top")
            label.set_horizontalalignment("center")
        elif 0 < angle_deg < 160:
            label.set_horizontalalignment("right")
        elif 160 < angle_deg < 180:
            label.set_verticalalignment("top")
            label.set_horizontalalignment("right")
        elif 180 < angle_deg < 200:
            label.set_verticalalignment("top")
            label.set_horizontalalignment("left")
        elif 200 < angle_deg < 360:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("center")
    # ax.set_title("Rule Scores Across Mo", weight='bold', size=16, position=(0.5, 1.1))
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(0, 1)
    # Resize theta (axis) labels
    for label in ax.get_xticklabels():
        label.set_fontsize(26)

    # Resize radial grid labels
    for label in ax.get_yticklabels():
        label.set_fontsize(25)
    if legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.6),
            ncols=6,
            frameon=False,
            fontsize=16,
        )

    plt.tight_layout()
    plt.show()
    #plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")


# fed


def get_rule_groups_for_srp_radar_plot(expr_name: str):
    lang: str = expr_name.split('-')[0]
    semantics_type: str = expr_name.split('-')[1]

    match lang:
        case "IMP":
            match semantics_type:
                case "K":
                    return IMP.get_semantics_rule_groups(Language.SEMANTICS_TYPE.K)         
                case "SOS":
                    return  IMP.get_semantics_rule_groups(Language.SEMANTICS_TYPE.SOS)
                case _:
                    raise NotImplementedError("Semantics not implemented for IMP")
        case _:
            raise NotImplementedError("Language not implemented!")
    #hctam
#fed


def gen_srp_radar_chart(
    prefix: str,
    expr_name: str,
    data_type: str,
    legend: bool = False
):
    rule_groups: dict = get_rule_groups_for_srp_radar_plot(expr_name)
    semantic: str = "uk" if prefix == "" else "mk"
    mutate: str = "standard" if prefix == "" else ("swap" if prefix == "addSub_mulDiv_negateRelation-" else "unseen")
    data: dict = load_json_data(f"{Macros.results_dir}/metrics-srp-{semantic}-{expr_name}.json")
    model_names: list = list(data.keys())
    model_axis_names: list = [llms2macros[model] for model in model_names]
    num_vars: int = len(model_names)
    theta = radar_factory(num_vars, frame="polygon")
    labels: list = [
        "Assignment",
        "Arithmetic",
        "Relational",
        "Logical",
        "Declaration",
        "Loop",
        "Halt",
        "Id",
        "Conditional",
        "Break & Continue",
    ]
    plot_dict_data: dict = {
        name: {label: -1 for label in labels} for name in model_names
    }
    for n, d in data.items():
        rule_data: dict = {label: [] for label in labels}
        for r, v in d[f"{prefix}{data_type}"].items():
            rule_data[_get_srp_rule_category(r, rule_groups)].append(v)
        # rof
        for k in rule_data.keys():
            if len(rule_data[k]) == 0:
                del plot_dict_data[n][k]
            else:
                plot_dict_data[n][k] = max(rule_data[k])
            # fi
        # rof
    # rof
    
    plot_data = []
    for label in labels:
        rule_values = [plot_dict_data[model].get(label, 0.0) for model in model_names]
        plot_data.append(rule_values)
    # rof

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="radar"))
    colors = ["b", "r", "g", "m", "orange", "c", "y", "peru", "steelblue", "gray"]

    for values, rule, color in zip(plot_data, labels, colors):
        ax.plot(theta, values, label=rule, color=color)
        ax.fill(theta, values, facecolor=color, alpha=0.2)

    ax.set_varlabels(model_axis_names)
    for label, angle in zip(ax.get_xticklabels(), theta):
        angle_deg = np.degrees(angle)
        if angle_deg == 0.0:
            label.set_horizontalalignment("center")
        elif angle_deg == 180.0:
            label.set_verticalalignment("top")
            label.set_horizontalalignment("center")
        elif 0 < angle_deg < 160:
            label.set_horizontalalignment("right")
        elif 160 < angle_deg < 180:
            label.set_verticalalignment("top")
            label.set_horizontalalignment("right")
        elif 180 < angle_deg < 200:
            label.set_verticalalignment("top")
            label.set_horizontalalignment("left")
        elif 200 < angle_deg < 360:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("center")
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(0, 1)
    # Resize theta (axis) labels
    for label in ax.get_xticklabels():
        label.set_fontsize(26)

    # Resize radial grid labels
    for label in ax.get_yticklabels():
        label.set_fontsize(25)
    if legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncols=5,
            frameon=False,
            fontsize=14,
        )

    plt.tight_layout()
    #plt.show()
    plt.savefig(
        f"srp-{semantic}-{mutate}-{data_type}-{expr_name}-radar-chart.pdf",
        format="pdf",
        bbox_inches="tight",
    )


# fed


# fed


def gen_etp_radar_chart(prefix: str, semantic: str, mutate: str, legend: bool = False):
    data = load_json_data(f"{Macros.results_dir}/metrics-etp-{semantic}-IMP-SOS.json")
    model_names = list(data.keys())
    model_axis_names = [llms2macros[model] for model in model_names]
    num_vars = len(model_names)
    theta = radar_factory(num_vars, frame="polygon")
    labels: list = [
        "extra execution error",
        "condition evaluation error",
        "control flow error",
        "computation error",
    ]
    error_types: list = [
        "long-execution-errors",
        "condition-errors",
        "control-flow-errors",
        "computation-errors",
    ]

    plot_data = []
    for error_type in error_types:
        rule_values = [
            data[model].get(f"{prefix}{error_type}", 0.0) for model in model_names
        ]
        plot_data.append(rule_values)
    # rof

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="radar"))
    colors = ["b", "r", "g", "m"]

    for values, rule, color in zip(plot_data, labels, colors):
        ax.plot(theta, values, label=rule, color=color)
        ax.fill(theta, values, facecolor=color, alpha=0.2)

    ax.set_varlabels(model_axis_names)
    for label, angle in zip(ax.get_xticklabels(), theta):
        angle_deg = np.degrees(angle)
        if angle_deg == 0.0:
            label.set_horizontalalignment("center")
        elif angle_deg == 180.0:
            label.set_verticalalignment("top")
            label.set_horizontalalignment("center")
        elif 0 < angle_deg < 180:
            label.set_horizontalalignment("right")
        elif 180 < angle_deg < 360:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("center")
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(0, 1)
    # Resize theta (axis) labels
    for label in ax.get_xticklabels():
        label.set_fontsize(26)

    # Resize radial grid labels
    for label in ax.get_yticklabels():
        label.set_fontsize(25)
    if legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncols=8,
            frameon=False,
            fontsize=14,
        )

    plt.tight_layout()
    # plt.show()
    plt.savefig(
        f"etp-{semantic}-{mutate}-radar-chart.pdf", format="pdf", bbox_inches="tight"
    )


# fed


def _get_srp_rule_category(rule: str, rule_groups: dict) -> str:
    for k, v in rule_groups.items():
        for ranges in v:
            low: int = ranges[0]
            high: int = ranges[-1]
            if low <= int(rule) <= high:
                return k
            # fi
        # rof
    # rof
# fed


def make_bars(data, metric="accuracy", color=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    models = [
        "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
        "meta-llama-Llama-3.3-70B-Instruct-cot",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
        "Qwen-QwQ-32B-da",
        "o3-mini-da",
        "gemini-2.5-pro-preview-05-06-da",
    ]
    model_labels = [llms2macros[model] for model in models]
    x = np.arange(len(model_labels))

    # Collect average performance for each model
    model_values = []
    for model in models:
        # Get all values for this model and calculate average
        value = data[model].get(f"addSub_mulDiv_negateRelation-{metric}", 0.0)
        model_values.append(value)

    # Create bars with different colors for each model
    bars = ax.bar(
        x,
        model_values,
        width=0.6,
        alpha=0.8,
        color=color if color else "steelblue",
        edgecolor="black",  # Add a black border to each bar
        linewidth=1.5,  # Optional: set the thickness of the border
    )
    # Set up the plot
    for bar, value in zip(bars, model_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Models", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=12)
    ax.set_ylim(0, 1.0)
    # ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    return ax

def plot_code_complexity_metric(data_lists: list, metric_name: str, data_names: list):
    metric_values_list: list = []
    for data_list in data_lists:
        metric_values: list = []
        for dt in data_list:
            match metric_name:
                case "Vocabulary" | "Volume" | "Difficulty" | "Effort":
                    metric_values.append(dt["Halstead"][metric_name])
                case "CC":
                    metric_values.append(dt["Cyclomatic"][metric_name])
                case _:
                    metric_values.append(dt[metric_name])
            #hctam
        #rof
        metric_values_list.append(metric_values)
    #rof
    fig, ax = plt.subplots()
    ax.violinplot(metric_values_list, showmeans=True)
    ax.set_xticks(np.arange(1, len(data_names) + 1))
    ax.set_xticklabels(data_names)
    ax.set_ylabel(metric_name)
    plt.show()
#fed

def compute_pcp_normalized_error_confusion_matrix(models: list, semantics_types: list, mutation_types: list, strategies: list, pcp_traces: list, rules: list):
    cm_matrices: list = []

    for model_pcp_traces, model_rules in zip(pcp_traces, rules):
        y_true = []
        y_pred = []

        for pcp_trace in model_pcp_traces:
            for (true_pcp, pred_pcp) in zip(pcp_trace.rule_true, pcp_trace.rule_pred):
                y_true.append(model_rules[true_pcp])
                if pred_pcp in model_rules:
                    y_pred.append(model_rules[pred_pcp])
                else:
                    # When the model prediction falls outside of the error categories for the PCP task
                    y_pred.append("Unknown")
                #fi
            #rof
        #rof
        rule_types = list(OrderedDict.fromkeys(model_rules.values()))
        y_true.append("Unknown")
        y_pred.append("Unknown")
        cm = confusion_matrix(y_true, y_pred, labels=rule_types)
        # Normalize the rows of the matrices
        cm_matrices.append(np.round(cm.astype(float) / cm.sum(axis=1)[:, None], 2))
    #rof

    return cm_matrices, rule_types
#fed


def plot_pcp_error_confusion_matrix(models: list, semantics_types: list, mutation_types: list, strategies: list, pcp_traces: list, rules: dict):
    [cm_matrices, rule_types] = compute_pcp_normalized_error_confusion_matrix(models, semantics_types, mutation_types, strategies, pcp_traces, rules)
    fig, axes = plt.subplots(math.ceil(len(cm_matrices) / 3), 3, figsize=(16, 8), sharey=True)
    axes = axes.ravel()
    sns.set_theme(font='Times New Roman')
    colors = ['orange', 'orange', 'green', 'red', 'red', 'green']

    for idx, cm_normalized in enumerate(cm_matrices):
        cm = cm_normalized.astype(float).copy()
        np.fill_diagonal(cm, 0.0)
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        norm = cm / rs
        np.around(norm, 2, out=norm)
        disp = sns.heatmap(norm, annot=True,cmap="Blues",cbar=False, ax=axes[idx], linewidth=1, square=True,edgecolor=None,xticklabels=rule_types, yticklabels=rule_types, annot_kws={"fontsize": 16},)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=rule_types)
        #disp.plot(ax=axes[idx] if len(cm_matrices) > 1 else axes, cmap="Blues", xticks_rotation=90, colorbar=False, text_kw={"fontsize": 14})
        if idx >= 4:
            axes[idx].set_xticks([])
            plt.setp(axes[idx].get_yticklabels(), rotation=0, fontsize=15)
        else:
            axes[idx].xaxis.tick_top()
            plt.setp(axes[idx].get_xticklabels(), rotation=90, fontsize=15)
            plt.setp(axes[idx].get_yticklabels(), rotation=0, fontsize=15)
        #fi
        
        for side in ["top", "bottom", "left", "right"]:
            axes[idx].spines[side].set_visible(True)
            axes[idx].spines[side].set_linewidth(2)
            #axes[idx].spines[side].set_color(colors[idx])
        #rof
    #rof

    for ax, model, strategy, semantics_type in zip(axes, models, strategies, semantics_types):
        ax.set_ylabel("")
        ax.set_xlabel(f"{llms2macrosfull[f'{model}-{strategy}']} - {semantics_type}",  fontsize=22)
    #rof

    #plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.tight_layout()
    plt.show()
#fed
    
def plot_pcp_dendrogram(models: list, semantics_types: list, mutation_types: list, strategies: list, pcp_traces: list, rules: list):
    [cm_matrices,rule_types] = compute_pcp_normalized_error_confusion_matrix(models, semantics_types, mutation_types, strategies, pcp_traces, rules)
    full_model_names: list = []
    for model, strategy, semantics_type in zip(models, strategies, semantics_types):
        full_model_names.append(f"{llms2macrosfull[f'{model}-{strategy}']} - {semantics_type}")
    #rof

    # out = cluster_models_with_dendrogram(
    #     cm_matrices,
    #     rule_types=rule_types,
    #     model_names=full_model_names,
    #     metric="cosine",            # or "js"
    #     include_asymmetry=False,
    #     keep_diagonal=False,
    #     pick_k_by_silhouette=True,
    # )

    results = analyze_confusion_matrices_auto(
        cm_matrices,
        sample_labels=full_model_names,
        class_names=rule_types,
        max_clusters=8,
        top_k=2
    )
    # print("Chosen k:", out["k"], "Silhouette:", round(out["silhouette"], 3))

    # # Characterize clusters
    # summaries = characterize_clusters(cm_matrices, out["labels"], rule_types=rule_types, top=5)
    # for s in summaries:
    #     print(f"\nCluster {s['cluster']} (n={s['size']}) — top mislabels:")
    #     for (ti, pj, delta, tname, pname) in s["top_mislabels"]:
    #         print(f"  {tname:>6} → {pname:<6}  Δ={delta:+.3f}")
    #     #rof
    # #rof
    # ax = out["ax_dendro"]
    # ax.tick_params(axis="x", labelrotation=90)
    # out["fig_dendro"].subplots_adjust(bottom=0.28)
    # plt.show()
#fed

def _cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp_den = (n1 + n2 - 2)
    sp = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / sp_den) if sp_den > 0 else 0.0
    if sp == 0.0:
        return 0.0 if m1 == m2 else np.sign(m1 - m2) * np.inf
    return (m1 - m2) / sp

def analyze_confusion_matrices_auto(
    cms,
    sample_labels=None,
    class_names=None,
    max_clusters=10,
    dendro_method="average",   # use 'average' (cosine is not compatible with 'ward')
    top_k=2,
    figsize=(11, 6),
):
    """
    End-to-end:
      1) Remove diagonals, row-normalize, vectorize off-diagonals
      2) Cosine distance + hierarchical clustering
      3) Auto-select #clusters via silhouette (max over k = 2..max_clusters)
      4) One-vs-rest Cohen's d -> top-K distinguishing features per cluster
      5) Plot dendrogram with *branch lines* colored by cluster (NOT tick labels)
         and legend listing top-K features per cluster with ↑/↓ arrows.

    Returns dict with X, features, distance_matrix, linkage, clusters, n_clusters, silhouette, top_features.
    """
    if len(cms) == 0:
        raise ValueError("No confusion matrices provided.")
    K = cms[0].shape[0]
    if any(cm.shape != (K, K) for cm in cms):
        raise ValueError("All confusion matrices must have the same KxK size.")

    # ---- (1) Preprocess + vectorize off-diagonals
    offmask = ~np.eye(K, dtype=bool)
    if class_names is None:
        features = [f"r{r}→c{c}" for r in range(K) for c in range(K) if r != c]
    else:
        if len(class_names) != K:
            raise ValueError("class_names must have length K.")
        features = [f"{class_names[r]}→{class_names[c]}" for r in range(K) for c in range(K) if r != c]


    X_rows = []
    for cm in cms:
        cm = cm.astype(float).copy()
        np.fill_diagonal(cm, 0.0)               # emphasize mispredictions
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        norm = cm / rs                           # row-normalize
        X_rows.append(norm[offmask])             # vectorize off-diagonal
        #X_rows.append(norm.ravel())             # vectorize off-diagonal
    X = np.vstack(X_rows)                        # (N x K*(K-1))
    N = X.shape[0]

    # ---- (2) Cosine distance + hierarchical clustering
    dists = pdist(X, metric="cosine")
    D = squareform(dists)
    Z = linkage(dists, method=dendro_method)

    # ---- (3) Auto choose #clusters via silhouette
    best_k, best_labels, best_score = None, None, -1
    for k in range(2, min(max_clusters, N) + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        counts = np.bincount(labels)[1:]  # labels start at 1
        if len(np.unique(labels)) < 2 or (counts <= 1).any():
            continue
        score = silhouette_score(X, labels, metric="cosine")
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score
    if best_labels is None:
        best_k = 2
        best_labels = fcluster(Z, best_k, criterion="maxclust")
        best_score = float("nan")

    print(best_score)

    # ---- (4) One-vs-rest Cohen's d top-K features per cluster
    top_features = {}
    for cl in np.unique(best_labels):
        in_idx = best_labels == cl
        out_idx = ~in_idx
        X_in, X_out = X[in_idx], X[out_idx]
        if X_out.shape[0] == 0 or X_in.shape[0] == 0:
            ds = np.zeros(X.shape[1])
        else:
            ds = np.array([_cohens_d(X_in[:, j], X_out[:, j]) for j in range(X.shape[1])])
        order = np.argsort(-np.abs(ds))[:top_k]
        top_features[int(cl)] = [(features[j], float(ds[j])) for j in order]

    # ---- (5) Plot dendrogram with *branch colors* per cluster
    if sample_labels is None:
        sample_labels = [f"cm_{i}" for i in range(N)]

    # Compute the height (distance) at which k clusters are formed.
    # Cutting at this threshold lets dendrogram color branches by cluster.
    color_threshold = Z[-(best_k - 1), 2] if best_k > 1 else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    dn = dendrogram(
        Z,
        labels=sample_labels,
        leaf_rotation=0,
        orientation="right",
        color_threshold=color_threshold,
        ax=ax,
        leaf_font_size=13,
    )
    #ax.set_title(f"Hierarchical Clustering (auto k={best_k}, silhouette={best_score:.3f})")
    ax.set_xlabel("Cosine Distance", fontsize=16)

    # Build legend colors to match the dendrogram's actual branch colors:
    # Map cluster -> color by reading leaf colors from the plotted dendrogram.
    leaves = dn["leaves"]                      # sample indices in dendrogram order
    leaf_colors = dn["leaves_color_list"]      # color used for each leaf's branch
    cluster_to_color = {}
    for pos, leaf_idx in enumerate(leaves):
        cl = int(best_labels[leaf_idx])
        if cl not in cluster_to_color:
            cluster_to_color[cl] = leaf_colors[pos]

    # Legend with top-K features (↑/↓)
    handles, labels_txt = [], []
    for cl in sorted(cluster_to_color.keys()):
        feats = top_features.get(int(cl), [])
        parts = []
        for name, d in feats:
            arrow = "↑" if d > 0 else "↓" if d < 0 else "→"
            parts.append(f"{name} {arrow} (d = {round(d,2)})")
        txt = f"Cluster {cl}: " + ", ".join(parts) if parts else f"C{cl}"
       # handles.append(Line2D([0], [0], color=cluster_to_color[cl], lw=2))
        handles.append(Patch(facecolor=cluster_to_color[cl], edgecolor="black", label=txt))
#        labels_txt.append(txt)

    ax.legend(
        handles=handles,
        #labels_txt,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
        fontsize=11,
    )

    ax.set_xlim(0, 0.8)
    ax.tick_params(axis="x", labelsize=14)


    plt.tight_layout()
    plt.show()

    return {
        "X": X,
        "features": features,
        "distance_matrix": D,
        "linkage": Z,
        "clusters": best_labels,
        "n_clusters": int(best_k),
        "silhouette": float(best_score),
        "top_features": top_features,
        "color_threshold": float(color_threshold),
        "cluster_colors": {int(k): v for k, v in cluster_to_color.items()},
    }

def failure_fingerprint(
    cm: np.ndarray,
    eps: float = 1e-9,
    include_asymmetry: bool = True,
    keep_diagonal: bool = False,
) -> np.ndarray:
    
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    P = cm / (row_sums + eps)

    # Remove diagonals (we want to emphasize the mispredictions)
    if not keep_diagonal:
        P = P.copy()
        np.fill_diagonal(P, 0.0)
    #fi
    feat = P.ravel()
    if include_asymmetry:
        A = P - P.T
        np.fill_diagonal(A, 0.0)
        feat = np.concatenate([feat, A.ravel()])
    #fi

    # Normalize feature vector to unit L1 (helps JS/cosine comparability)
    s = feat.sum()
    if s > 0:
        feat = feat / s
    return feat
#fed


def build_feature_matrix(
    cm_matrices: Sequence[np.ndarray],
    eps: float = 1e-9,
    include_asymmetry: bool = True,
    keep_diagonal: bool = False,
) -> np.ndarray:
    X = np.vstack([
        failure_fingerprint(cm, eps=eps,
                            include_asymmetry=include_asymmetry,
                            keep_diagonal=keep_diagonal)
        for cm in cm_matrices
    ])
    return X
#fed


def pairwise_distance(
    X: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    # Compute pairwise distance matrix
    if metric == "cosine":
        D = squareform(pdist(X, metric="cosine"))
    elif metric in {"js", "jensen-shannon"}:
        # pdist with custom JS (returns distance)
        D = squareform(pdist(X, metric=lambda a, b: jensenshannon(a, b)))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return D
#fed


def _agglo_precomputed(D: np.ndarray, n_clusters: int, linkage_kind: str = "average") -> np.ndarray:
    # Agglomerative clustering for hierarchical clustering from the pairwise distance matrix
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage_kind)
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage_kind)
    labels = model.fit_predict(D)
    return labels
#fed


def choose_k_by_silhouette(D: np.ndarray, k_min: int = 2, k_max: Optional[int] = None) -> Tuple[int, np.ndarray, float]:
    # Pick cluster count k by  maximizing silhouette score (precomputed distances)
    n = D.shape[0]
    if k_max is None:
        k_max = max(2, min(8, n))  # heuristic cap

    best_k, best_labels, best_s = None, None, -1.0
    for k in range(k_min, min(k_max, n - 1) + 1):
        labels = _agglo_precomputed(D, n_clusters=k, linkage_kind="average")
        # silhouette_score expects 'precomputed' distances
        s = silhouette_score(D, labels, metric="precomputed")
        if s > best_s:
            best_k, best_labels, best_s = k, labels, s
    if best_labels is None:
        # fallback to k=2
        best_k = 2
        best_labels = _agglo_precomputed(D, n_clusters=2, linkage_kind="average")
        best_s = silhouette_score(D, best_labels, metric="precomputed")
    return best_k, best_labels, best_s
#fed


def characterize_clusters(
    cm_matrices: Sequence[np.ndarray],
    labels: Sequence[int],
    rule_types: Optional[Sequence[str]] = None,
    top: int = 5,
    eps: float = 1e-9,
) -> List[Dict]:
    # Compare each cluster's average row-normalized confusion to the global average.
    # Report top 'lift' misdiagnoses (i->j where cluster overrates vs global).
    labels = np.asarray(labels)
    clusters = sorted(set(labels))
    cm_arr = np.stack(cm_matrices, axis=0)

    # Global mean (row-normalized)
    gm = cm_arr.mean(axis=0)
    gm_row = gm / (gm.sum(axis=1, keepdims=True) + eps)
    np.fill_diagonal(gm_row, 0.0)

    summaries = []
    for c in clusters:
        idxs = np.where(labels == c)[0]
        sub = cm_arr[idxs].mean(axis=0)
        sub_row = sub / (sub.sum(axis=1, keepdims=True) + eps)
        np.fill_diagonal(sub_row, 0.0)

        lift = sub_row - gm_row
        order = np.argsort(lift.ravel())[::-1]
        ti, pj = np.unravel_index(order, lift.shape)

        entries = []
        for k in range(min(top, len(order))):
            t, p = int(ti[k]), int(pj[k])
            entries.append((
                t, p, float(lift[t, p]),
                (rule_types[t] if rule_types is not None else str(t)),
                (rule_types[p] if rule_types is not None else str(p)),
            ))
        #rof
        summaries.append({"cluster": int(c), "size": int(len(idxs)), "top_mislabels": entries})
    #rof
    return summaries
#fed


def plot_dendrogram_from_distance(
    D: np.ndarray,
    model_names: Optional[Sequence[str]] = None,
    method: str = "average",
    title: str = "Model failure dendrogram",
    figsize: Tuple[float, float] = (8, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    # SciPy linkage expects a condensed distance vector
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        Z,
        labels=model_names,
        leaf_rotation=0,
        leaf_font_size=13,
        orientation="right",  
        color_threshold=None,            # we control colors ourselves
        above_threshold_color="black",   # fallback (also used for mixed nodes)
        ax=ax
    )

    #dendrogram(Z, labels=model_names, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig, ax
#fed


def plot_distance_heatmap(
    D: np.ndarray,
    order: Optional[Sequence[int]] = None,
    model_names: Optional[Sequence[str]] = None,
    title: str = "Pairwise model distance",
    figsize: Tuple[float, float] = (5, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    # Seaborn cluster map
    if order is None:
         order = np.arange(D.shape[0])
    # Do = D[np.ix_(order, order)]

    #fig, ax = plt.subplots(figsize=figsize)
    #im = ax.imshow(Do, interpolation="nearest", aspect="auto")
    # ax.set_title(title)
    # ax.set_xticks(range(len(order)))
    # ax.set_yticks(range(len(order)))
    # if model_names is not None:
    #      ordered_names = [model_names[i] for i in order]
    #      ax.set_xticklabels(ordered_names, rotation=90)
    #      ax.set_yticklabels(ordered_names)
    # fig.colorbar(im, ax=ax)
    # fig.tight_layout()
    sns.clustermap(D)
    #return fig, ax
#fed


def cluster_models_with_dendrogram(
    cm_matrices: Sequence[np.ndarray],
    rule_types: Optional[Sequence[str]] = None,
    model_names: Optional[Sequence[str]] = None,
    metric: str = "cosine",
    include_asymmetry: bool = True,
    keep_diagonal: bool = False,
    pick_k_by_silhouette: bool = True,
    k: Optional[int] = None,
) -> Dict:
    # build features -> distances -> clusters -> dendrogram.
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(cm_matrices))]

    # Features and distances
    X = build_feature_matrix(
        cm_matrices,
        include_asymmetry=include_asymmetry,
        keep_diagonal=keep_diagonal,
    )
    D = pairwise_distance(X, metric=metric)

    # Pick k or use provided
    if pick_k_by_silhouette or (k is None):
        k, labels, sil = choose_k_by_silhouette(D)
    else:
        labels = _agglo_precomputed(D, n_clusters=k, linkage_kind="average")
        sil = silhouette_score(D, labels, metric="precomputed")

    # Dendrogram
    fig, ax = plot_dendrogram_from_distance(D, model_names=model_names)
    #plot_distance_heatmap(D, model_names=model_names)

    return {
        "X": X,
        "D": D,
        "labels": labels,
        "k": k,
        "silhouette": sil,
        "fig_dendro": fig,
        "ax_dendro": ax,
    }
#fed


def plot_metric_against_metric(dataset_file: str, metric1: str, metric2: str):
    X: list = dataset_code_complexity_metrics_to_df(read_from_json_file(dataset_file))
    #X = X[X[metric2] > 150]
    grouped_data = X.groupby([metric1, metric2]).size().unstack(fill_value=0)
    #grouped_data = X.groupby(metric1)[metric2].sum()
    # Plot the stacked bar chart
    grouped_data.plot(kind='bar', stacked=False, figsize=(8, 6))
    plt.show()
#fed


def plot_dataset_violin_plots(dataset_files: list, metrics: list, dataset_names: list,  num_row: int, num_col: int):
    X: list = [dataset_code_complexity_metrics_to_df(read_from_json_file(dataset_file)) for dataset_file in dataset_files]
    colors: list = ['salmon', 'steelblue', 'forestgreen', 'orange']
    plt.rcParams["text.usetex"] = True
    
    def color_violin(parts, face_color, edge_color, alpha):
        for idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[idx])
            pc.set_alpha(alpha)
            pc.set_edgecolor(colors[idx])
        #rof
        parts['cmedians'].set_color('grey')
        parts['cmaxes'].set_color('grey')
        parts['cmins'].set_color('grey')
        parts['cbars'].set_color('grey')
        parts['cmedians'].set_linewidth(1)
        parts['cmaxes'].set_linewidth(1)
        parts['cmins'].set_linewidth(1)
        parts['cbars'].set_linewidth(1)
    #fed

    def make_legend(alpha: float):
        legend_elems = []
        for i, dataset_name in enumerate(dataset_names):
            legend_elems.append(Patch(facecolor=colors[i], edgecolor="black", label=dataset_name, alpha=alpha))
        #rof
        return legend_elems
    #fed


    def bare_violins(X2, metrics):
        plt.figure(0)
        fig, axs = plt.subplots(num_row, num_col, figsize=(32,16), num=0)
        if num_row > 1:
            axs2 = axs.ravel()
        #fi
        for idx, metric in enumerate(metrics):
            X = X2.copy()
            # threshold = X[metric].quantile(0.75)
            # X = X[X[metric] >= threshold]
            plot_datas: list = []
            axse = axs2 if num_row == 1 else axs2[idx]
            max_glob_metric = -100
            min_glob_metric = 10 ** 9
            for x in X:
                plot_data: list = x[metric]
                min_glob_metric = min(min(plot_data), min_glob_metric)
                #plot_data = [np.log10(y + 0.1) for y in list(x[metric])]
                q1 = np.percentile(plot_data, 25)
                q2 = np.percentile(plot_data, 50)
                q3 = np.percentile(plot_data, 75)
                max_metric = max(plot_data)
                if max_metric < 10:
                    max_metric = round(max_metric + 1)
                elif max_metric < 100:
                    max_metric += 100
                    max_metric = round(max_metric, -1)
                    max_metric = 10 if max_metric == 0 else max_metric
                else:
                    max_metric += 100
                    max_metric = round(max_metric, -2)
                #axse.axhspan(q1, q3, alpha=0.15, lw=0)
                #axse.axhline(y=q1, xmin=0, xmax=1, alpha=0.7, lw=0.2, ls='--', color='black')
                #axse.axhline(y=q2, xmin=0, xmax=1, alpha=0.7, lw=1, ls='--', color='black')
                #axse.axhline(y=q3, xmin=0, xmax=1, alpha=0.7, lw=0.2, ls='--', color='black')
                max_glob_metric = max(max_glob_metric, max_metric)
                plot_datas.append(plot_data)
            #rof
            part = axse.violinplot(plot_datas, showmedians=True, showextrema=True)
            color_violin(part, 'steelblue', 'steelblue', 0.4)
            axse.set_xticks([])
            axse.tick_params(axis='y', labelsize=19)
            axse.set_ylabel(metrics2macros[metric], fontsize=20)
            axse.set_xticklabels([])
            axse.set_yscale('symlog')
            axse.autoscale(enable=True, axis='y')
            axse.set_ylim(0 if min_glob_metric < 1 else 10 ** round(math.floor(np.log10(min_glob_metric))), 10 ** round(math.ceil(np.log10(max_glob_metric))))
            #axse.yaxis.set_major_locator(MaxNLocator(integer=True))
            #axse.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
        #rof
        plt.legend(
            handles=make_legend(0.4),
            loc="upper center",
            bbox_to_anchor=(-0.75, 5),
            ncol=len(dataset_names),
            labelspacing=3,
            frameon=False,
            fontsize=19,
        )
        plt.subplots_adjust(wspace=0.24) 
        # plt.savefig(
        #     "dataset-complexity-metrics.pdf",
        #     format="pdf",
        #     bbox_inches="tight",
        #     dpi=300,
        # )
        plt.show()
    #fed

    bare_violins(X, metrics)

#fed


def plot_op_violin_plots(models: list, semantics_types: list, strategies: list, op_traces: list, dataset_file: str, metrics: list[str], filter_while_count: int = -1, filter_if_count: int = -1):
    [X, Y] = prepare_dataset_and_op_output_dfs(models, semantics_types, strategies, op_traces, dataset_file, metrics, ['uk'] * len(models), filter_while_count, filter_if_count)
    metrics: list = X.columns
    models: list = Y.columns

    def color_violin(parts, face_color, edge_color, alpha):
        for pc in parts['bodies']:
            pc.set_facecolor(face_color)
            pc.set_edgecolor(edge_color)
            pc.set_alpha(alpha)
        #rof
        parts['cmedians'].set_color(edge_color)
    #fed

    def bare_violins(X, Y, metrics, model):
        X = pd.concat([X,Y], axis=1)
        plt.figure(0)
        fig, axs = plt.subplots(2, 4, num=0)
        axs = axs.ravel()
        for idx, metric in enumerate(metrics):
            metric_pass: list = list(X[X[model] == 1][metric])
            metric_fail: list = list(X[X[model] == 0][metric])
            q1 = np.percentile(list(X[metric]), 25)
            q3 = np.percentile(list(X[metric]), 75)
            axs[idx].axhspan(q1, q3, alpha=0.15, lw=0)  # IQR band
            correct_part = axs[idx].violinplot(metric_pass, side='low', showmedians=True, showextrema=False)
            wrong_part = axs[idx].violinplot(metric_fail,  side='high', showmedians=True, showextrema=False)
            color_violin(correct_part, 'blue', 'blue', 0.3)
            color_violin(wrong_part, 'red', 'red', 0.3)
            axs[idx].set_ylabel(metric)
            axs[idx].set_xticklabels([])
        #rof
        plt.show()
    #fed

    bare_violins(X,Y, metrics, models[0])
#fed

def plot_op_regression_heatmap(models: list, semantics_types: list, strategies: list, op_traces: list, dataset_file: str, metrics: list[str], filter_while_count: int = -1, filter_if_count: int = -1):
    [X, regression_details] = gen_op_regression_coefficients(models, semantics_types, strategies, op_traces, dataset_file, metrics, ["nk"] * len(models), filter_while_count, filter_if_count)

    y_labels: list[str] = []
    x_labels: list[str] = metrics
    data: list = []

    for model, model_regression_detail in regression_details.items():
        y_labels.append(model)
        coef_table = model_regression_detail['coef_table']
        beta_per_sd: list = []
        #beta_per_sd: list = coef_table['beta_orig']
        for metric in metrics:
            #beta_per_sd.append(100 * (coef_table['OR_per_IQR'][metric] - 1))
            beta_per_sd.append(coef_table['beta_per_SD'][metric])
        #rof
        data.append(beta_per_sd)
    #rof
    coef_matrix: pd.DataFrame = pd.DataFrame(data, columns=metrics, index=y_labels)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(coef_matrix.values, aspect="auto")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

#fed


def plot_op_beta_sd_dendrogram(models: list, semantics_types: list, strategies: list, op_traces: list, dataset_file: str, metrics: list[str], mutations: list = None, filter_while_count: int = -1, filter_if_count: int = -1):
    [X, regression_details] = gen_op_regression_coefficients(models, semantics_types, strategies, op_traces, dataset_file, metrics, mutations, filter_while_count, filter_if_count)

    y_labels: list[str] = []
    x_labels: list[str] = metrics
    data: list = []

    for model, model_regression_detail in regression_details.items():
        y_labels.append(model)
        coef_table = model_regression_detail['coef_table']
        beta_per_sd: list = []
        for metric in metrics:
            beta_per_sd.append(coef_table['beta_per_SD'][metric])
        #rof
        data.append(beta_per_sd)
    #rof
    coef_matrix: pd.DataFrame = pd.DataFrame(data, columns=metrics, index=y_labels)
    def cluster_and_signature_features(
        coef_matrix: pd.DataFrame,          
        distance: str = "cosine",          
        linkage_method: str = "average",   
        k: int | None = None,              
        top_n: int = 2,
        use_hedges_g: bool = True    
    ):
        X = coef_matrix.copy()
        n = len(X)
        if n < 3:
            raise ValueError("Need at least 3 rows to cluster and compute effects.")

        # Optional: z-score standardize each feature (column)
        X = (X - X.mean(axis=0)) / X.std(axis=0).replace(0, np.nan)
        X = X.fillna(0.0)  # if any std=0 columns


        # --- HCLUST ---
        D = pdist(X.values, metric=distance)
        Z = linkage(D, method=linkage_method)

        # choose k if needed
        if k is None:
            best_k, best_sil = None, -1
            for kk in range(2, min(8, n)):  # try small ks; adjust as you like
                labels_try = fcluster(Z, t=kk, criterion="maxclust")
                # silhouette needs feature space; cosine ok; ward->euclidean recommended
                sil = silhouette_score(
                    X.values, labels_try,
                    metric=distance if distance != "correlation" else "euclidean"
                )
                if sil > best_sil:
                    best_k, best_sil = kk, sil
            print(best_sil)
            k = best_k

        labels = fcluster(Z, t=k, criterion="maxclust")

        # --- One-vs-rest effect sizes (Cohen's d / Hedges' g) ---
        df = X.copy()
        df["__c__"] = labels
        clusters = np.unique(labels)
        feats = X.columns

        d_mat = pd.DataFrame(index=clusters, columns=feats, dtype=float)

        for c in clusters:
            in_c  = df["__c__"] == c
            out_c = ~in_c
            n_in, n_out = int(in_c.sum()), int(out_c.sum())
            # means & variances
            mu_in  = df.loc[in_c, feats].mean()
            mu_out = df.loc[out_c, feats].mean()
            s2_in  = df.loc[in_c, feats].var(ddof=1)
            s2_out = df.loc[out_c, feats].var(ddof=1)
            # pooled SD
            s_pooled = np.sqrt(((n_in - 1) * s2_in + (n_out - 1) * s2_out) / (n_in + n_out - 2)).replace(0, np.nan)
            d = (mu_in - mu_out) / s_pooled

            if use_hedges_g:
                # small-sample correction: g = J * d, where J ≈ 1 - 3/(4*(n_in+n_out) - 9)
                J = 1.0 - 3.0 / (4.0 * (n_in + n_out) - 9.0) if (n_in + n_out) > 3 else 1.0
                d = d * J

            d_mat.loc[c] = d.values

        # importance = |effect size|
        score_mat = d_mat.abs()
        signed    = d_mat

        # --- Pick top feature(s) per cluster ---
        signatures = {}
        for c in score_mat.index:
            ranked = score_mat.loc[c].sort_values(ascending=False)
            top_feats = ranked.index[:top_n].tolist()
            signatures[c] = [
                (f, float(signed.loc[c, f]))  # keep sign for ↑/↓
                for f in top_feats
            ]

        # cluster sizes & centroids for context
        sizes = pd.Series(labels).value_counts().sort_index()
        centroids = df.groupby("__c__").mean(numeric_only=True).drop(columns="__c__", errors="ignore")

        return {
            "Z": Z,
            "labels": labels,            # array of cluster labels per row
            "k": k,
            "sizes": sizes,              # models per cluster
            "centroids": centroids,      # mean beta_SD per feature in each cluster
            "effect_sizes": signed,      # Cohen's d (or Hedges' g) per cluster-feature (signed)
            "signatures": signatures     # dict: cluster -> list of (feature, signed_effect)
        }

    #plot_model_dendrogram(coef_matrix)
    res = cluster_and_signature_features(coef_matrix)
    print(res["signatures"])
    fig, ax = plt.subplots(figsize=(8,5))
    coef_with_clusters = coef_matrix.copy()
    labels = res['labels']
    coef_with_clusters["cluster"] = res['labels']
    Z = res['Z']

    color_map = {
        1: "#FA8072",
        2: "#9370DB",
        3: "#228B22",
        4: "#FFA500"
    }

    def cut_threshold_for_k(Z, k: int) -> float:
        """Pick a distance threshold that yields exactly k clusters."""
        n = Z.shape[0] + 1
        d = Z[:, 2]
        m = n - k
        if m <= 0:
            return float(d[0]) - 1e-9 if len(d) else 0.0
        if m >= len(d):
            return float(d[-1]) + 1e-9
        return float((d[m-1] + d[m]) / 2.0)

    def plot_dendrogram_color_by_clusters(Z, model_names, k, cluster_colors, signatures):
        """
        Z: linkage matrix
        model_names: list/Index of leaf labels (same row order used to build Z)
        k: number of clusters to cut
        cluster_colors: dict {cluster_id:int -> color:str}, e.g. {1:"#1f77b4", 2:"#ff7f0e", 3:"#2ca02c"}
        """
        n = Z.shape[0] + 1
        # 1) cluster labels for leaves (0..n-1)
        labels = fcluster(Z, t=k, criterion="maxclust")

        # 2) cut height
        h = cut_threshold_for_k(Z, k)

        # 3) precompute, for every node id, which cluster it belongs to (if homogeneous under the cut)
        # node ids: 0..n-1 leaves, n..2n-2 internal nodes (row i -> node id n+i)
        node_leaves = {i: {i} for i in range(n)}           # leaf sets per node
        node_cluster = {i: labels[i] for i in range(n)}    # cluster id per node if homogeneous; else None
        node_height  = {i: 0.0 for i in range(n)}          # height per node

        for i, (a, b, dist, _) in enumerate(Z):
            a, b = int(a), int(b)
            node_id = n + i
            # union of leaves
            leaves = node_leaves[a] | node_leaves[b]
            node_leaves[node_id] = leaves
            node_height[node_id] = float(dist)
            # homogeneous cluster id if all leaves share the same leaf label AND the node is under cut
            cl_set = set(labels[list(leaves)])
            if len(cl_set) == 1 and dist <= h:
                node_cluster[node_id] = next(iter(cl_set))
            else:
                node_cluster[node_id] = None

        # 4) link_color_func that uses our precomputed mapping
        def link_color_func(node_id: int) -> str:
            cid = node_cluster.get(node_id, None)
            if cid is not None:
                return cluster_colors.get(int(cid), "gray")
            # above threshold or mixed: use black
            return "steelblue"

        # 5) plot
        fig, ax = plt.subplots(figsize=(9, 5))
        dendrogram(
            Z,
            labels=list(map(str, model_names)),
            leaf_rotation=0,
            leaf_font_size=13,
            orientation="right",  
            color_threshold=None,            # we control colors ourselves
            link_color_func=link_color_func, # color branches by cluster
            above_threshold_color="black",   # fallback (also used for mixed nodes)
            ax=ax
        )
        ax.set_xlabel("Cosine Distance", fontsize=16)

        # Legend entries: one per cluster, showing signature feature + effect
        legend_elems = []
        for cid, color in cluster_colors.items():
            if cid in signatures:
                feat, val = signatures[cid][0]  # first/top feature
                feat2, val2 = signatures[cid][1]
                arrow = "↑" if val > 0 else "↓"
                arrow2 =  "↑" if val2 > 0 else "↓"
                label = f"Cluster {cid}: {feat} {arrow} (d={val:.2f})\n                {feat2} {arrow2} (d={val2:.2f})"
            else:
                continue
                #label = f"Cluster {cid}"
            legend_elems.append(Patch(facecolor=color, edgecolor="black", label=label))
        ax.legend(
            handles=legend_elems,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,                    
            frameon=False,
            fontsize=11,
        )
        ax.set_xlim(0, 1.6)
        ax.tick_params(axis="x", labelsize=14)

        plt.tight_layout()
        plt.show()

    model_names: list = coef_matrix.index.to_list()
    for idx, model in enumerate(model_names):
        model_details: list = model.split("-")
        model_name: str = "-".join(model_details[0: (len(model_details) - 2)])
        semantics_type: str = model_details[-2]
        mutation_type: str = model_details[-1]
        if mutation_type != 'nk':
            model_names[idx] = f"{llms2macrosfull[model_name]} - {semantics_type}"
        else:
            model_names[idx] = llms2macrosfull[model_name]
        #fi
    #rof
    plot_dendrogram_color_by_clusters(Z, model_names, res['k'], color_map, res['signatures'])
#fed

def plot_notation_comprehension_rule_distribution(
    hf_repo_id: str,
    hf_config: str,
    hf_split: str,
):
    dataset = load_dataset(hf_repo_id, name=hf_config, split=hf_split)
    rule_family_occurrence = defaultdict(int)
    for item in dataset:
        rule_family_occurrence[IMP._SOS_RULE_META_MAP[int(item['answer_rule_id'])][0]] += 1
    #normalize the occurrence by the total number of items
    total_items = len(dataset)
    for rule_family, occurrence in rule_family_occurrence.items():
        rule_family_occurrence[rule_family] = occurrence / total_items * 100
    #rof
    rule_family_occurrence = dict(sorted(rule_family_occurrence.items(), key=lambda item: item[1], reverse=True))
    #print(rule_family_occurrence)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rule_family_occurrence.keys(), rule_family_occurrence.values())
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_xlabel("Rule Family", fontsize=16)
    ax.set_ylabel("Occurrence (\%)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
#fed

def plot_notation_comprehension_rule_distribution_all(
    hf_repo_id: str,
):
    hf_config: dict = {
        "nl2rule": ["Standard_NumRule5_RandomSampleFalse", "NonStandard_NumRule5_RandomSampleFalse"],
        "rule2nl": ["Standard_NumDescription5_RandomSampleFalse", "NonStandard_NumDescription5_RandomSampleFalse"],
    }
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))
    for idx, (hf_config1, hf_splits) in enumerate(hf_config.items()):
        for jdx, hf_split in enumerate(hf_splits):
            dataset = load_dataset(hf_repo_id, name=hf_config1, split=hf_split)
            rule_family_occurrence = defaultdict(int)
            for item in dataset:
                rule_family_occurrence[IMP._SOS_RULE_META_MAP[int(item['answer_rule_id'])][0]] += 1
            #normalize the occurrence by the total number of items
            total_items = len(dataset)
            for rule_family, occurrence in rule_family_occurrence.items():
                rule_family_occurrence[rule_family] = occurrence / total_items * 100
            #rof
            rule_family_occurrence = dict(sorted(rule_family_occurrence.items(), key=lambda item: item[1], reverse=True))
            #print(rule_family_occurrence)
            ax[idx, jdx].bar(rule_family_occurrence.keys(), rule_family_occurrence.values())
            ax[idx, jdx].tick_params(axis='both', which='major', labelsize=12)
            ax[idx, jdx].set_xlabel("Rule Family", fontsize=16)
            ax[idx, jdx].set_ylabel("Occurrence (\%)", fontsize=16)
    plt.tight_layout()
    plt.show()
#fed

def plot_notation_comprehension_confusion_matrix(
    model_results: list,
    setup_name: str,
    mutation_type: str,
):

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    def pastelize_cmap(cmap, amount=0.45):
        colors = cmap(np.linspace(0, 1, 256))
        white = np.ones_like(colors)
        colors[:, :3] = colors[:, :3] * (1 - amount) + white[:, :3] * amount
        return mcolors.LinearSegmentedColormap.from_list(
            f"pastel_{cmap.name}", colors
        )

    pastel_coolwarm = pastelize_cmap(plt.cm.coolwarm, amount=0.45)

    y_true = []
    y_pred = []
    for model_result in model_results:
        if setup_name == 'mk':
            if mutation_type == 'keywordswap':
                y_true.extend(model_result['addSub_mulDiv_negateRelation'].true_mapped_ans)
                y_pred.extend(model_result['addSub_mulDiv_negateRelation'].pred_mapped_ans)
            elif mutation_type == 'keywordobf':
                y_true.extend(model_result['unseen'].true_mapped_ans)
                y_pred.extend(model_result['unseen'].pred_mapped_ans)
            #fi
        else:
            y_true.extend(model_result.true_mapped_ans)
            y_pred.extend(model_result.pred_mapped_ans)
    #rof
    
    # Get all unique labels to ensure consistent ordering
    all_labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    # Find the top 5 mismatches (off-diagonal cells with highest values)
    # Create a copy of the confusion matrix and zero out the diagonal
    cm_off_diagonal = cm.copy()
    np.fill_diagonal(cm_off_diagonal, 0)
    
    # Find the top 5 mismatches
    # Flatten the matrix and get indices of top values
    flat_indices = np.argsort(cm_off_diagonal.flatten())[::-1]
    top_5_indices = flat_indices[:3]
    
    # Convert flat indices back to row, col indices
    n_rows, n_cols = cm.shape
    top_5_rows = []
    top_5_cols = []
    for flat_idx in top_5_indices:
        row_idx = flat_idx // n_cols
        col_idx = flat_idx % n_cols
        if cm_off_diagonal[row_idx, col_idx] > 0:  # Only include non-zero mismatches
            top_5_rows.append(row_idx)
            top_5_cols.append(col_idx)
        #fi
    #rof
    
    if len(top_5_rows) == 0:
        print("Warning: No mismatches found. Showing full confusion matrix.")
        keep_row_indices = np.arange(len(cm))
        keep_col_indices = np.arange(len(cm))
    else:
        # Keep rows that contain the top 5 mismatches
        keep_row_indices = np.unique(top_5_rows)
        # For these rows, keep ALL columns (to show all predictions for these rows)
        # Find all columns that have non-zero values in the kept rows
        keep_col_indices = np.where(cm[keep_row_indices, :].sum(axis=0) > 0)[0]
    #fi
    
    # Filter the confusion matrix: keep only specified rows and columns
    # This may result in a non-square matrix
    cm_filtered = cm[np.ix_(keep_row_indices, keep_col_indices)]
    
    #np.fill_diagonal(cm_filtered, 0.0)
    cm_filtered = np.multiply(np.round(cm_filtered.astype(float) / cm_filtered.sum(axis=1)[:, None], 2), 100).astype(int)
    
    # Get labels for the filtered rows and columns
    filtered_row_labels = [all_labels[i] for i in keep_row_indices]
    filtered_col_labels = [all_labels[i] for i in keep_col_indices]
    
    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="white")
    sns.heatmap(
        cm_filtered,
        cmap=pastel_coolwarm,       # blue ↔ red like your figure
        vmin=0,
        vmax=100,
        square=True,
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt='d',
        annot_kws={"fontsize": 10},
        cbar_kws=dict(shrink=0.85),
        xticklabels=filtered_col_labels,
        yticklabels=filtered_row_labels,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.show()
#fed




