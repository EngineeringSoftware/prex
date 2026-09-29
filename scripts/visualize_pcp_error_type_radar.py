"""
Per-error-type accuracy radar charts for all PredExe models.

Two-by-three grid of IMP-SOS configurations only: human-written on top,
fuzzer-generated on bottom (uk, mk Swap, mk Obf columns). Each panel uses
one axis per model (hexadecagon) and one colored polygon per invalid semantic
error type (5 total, shared legend). The random-guess baseline is omitted.

Metric: among invalid programs whose true semantic error is type T, the fraction
where the model predicts invalid and identifies the violated rule correctly
(same rule-level criterion as the paper accuracy tables).

Usage:
  python scripts/visualize_pcp_error_type_radar.py
  python scripts/visualize_pcp_error_type_radar.py --show
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.path import Path as MplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.patches import Circle, RegularPolygon

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pcp_error_type_helpers import (  # noqa: E402
    ERROR_TYPE_LABELS,
    INVALID_ERROR_TYPES,
    averaged_error_type_accuracy,
)
from visualize_pcp_split_accuracy_tables import (  # noqa: E402
    DEFAULT_MODEL_PREFIX,
    BaselineConfig,
    ModelConfig,
    TABLE_COLUMNS,
    assemble_row_configs,
    column_exp_name,
    column_pattern,
    column_setup,
    discover_ministral_model_configs,
    discover_model_configs,
    discover_reasoning_model_configs,
    load_result_sets,
    result_dir_for_config,
)

DATASET_SPLITS = (
    "human_written",
    "fuzzer_generated",
    "synthetic_cpp",
)
SPLIT_LABELS = {
    "human_written": "Human-written",
    "fuzzer_generated": "Fuzzer-generated",
    "synthetic_cpp": "LLM-translated",
}

SPLIT_SLUGS = {
    "human_written": "human-written",
    "fuzzer_generated": "fuzzer-generated",
    "synthetic_cpp": "llm-translated",
}

FIGURES_DIR = REPO_ROOT / "papers" / "lmpl26" / "figures"
DEFAULT_RADAR_OUTPUT = FIGURES_DIR / "pcp-error-type-radar.pdf"
DEFAULT_LEGEND_OUTPUT = FIGURES_DIR / "pcp-error-type-radar-legend.pdf"

SOS_CONFIGS: tuple[tuple, ...] = (
    ("uk", "IMP-SOS", None),
    ("mk", "IMP-SOS", "KeywordSwap"),
    ("mk", "IMP-SOS", "KeywordObf"),
)

FIGURE_SPLITS: tuple[str, ...] = (
    "human_written",
    "fuzzer_generated",
)

PANEL_ORDER: tuple[tuple[str, tuple], ...] = tuple(
    (split, config)
    for split in FIGURE_SPLITS
    for config in SOS_CONFIGS
)

ERROR_TYPE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

PANEL_LINEWIDTH_FRAC = 0.20
PANEL_COLUMN_GAP = r"\hspace{0.03\linewidth}"
LEGEND_TO_PANELS_GAP = r"\\[-4mm]"
ROW_BREAK_BETWEEN_ROWS = r"\\[1mm]"
PANEL_FIGSIZE = (3.6, 3.6)
MODEL_AXIS_LABEL_FONT_MIN = 13.0
MODEL_AXIS_LABEL_FONT_SCALE = 13.0

SUBCAPTION_MACROS = {
    ("uk", "IMP-SOS", None): r"\FigPCPSOSUKRadarCaption",
    ("mk", "IMP-SOS", "KeywordSwap"): r"\FigPCPSOSSwapRadarCaption",
    ("mk", "IMP-SOS", "KeywordObf"): r"\FigPCPSOSObfRadarCaption",
}


def model_axis_label_fontsize(panel_scale: float) -> float:
    return max(MODEL_AXIS_LABEL_FONT_MIN, MODEL_AXIS_LABEL_FONT_SCALE * panel_scale)


def radar_factory(num_vars: int, frame: str = "polygon") -> np.ndarray:
    """Copied from llm_interpreter.paper_latex.figure (avoids usetex side effects)."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MplPath(self.transform(path.vertices), path.codes)

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
            if frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            raise ValueError(f"Unknown frame: {frame}")

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            if frame == "polygon":
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=MplPath.unit_regular_polygon(num_vars),
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            raise ValueError(f"Unknown frame: {frame}")

    register_projection(RadarAxes)
    return theta


@dataclass(frozen=True)
class ModelRadarSeries:
    row_id: str
    label: str
    axis_label: str
    accuracies: np.ndarray


def discover_all_row_configs() -> list[BaselineConfig | ModelConfig]:
    qwen_configs = discover_model_configs(DEFAULT_MODEL_PREFIX)
    ministral_configs = discover_ministral_model_configs()
    reasoning_configs = discover_reasoning_model_configs()
    return assemble_row_configs(
        qwen_configs,
        ministral_configs,
        reasoning_configs,
    )


def radar_model_label(cfg: BaselineConfig | ModelConfig, *, axis: bool = False) -> str:
    if isinstance(cfg, BaselineConfig):
        short = cfg.row_label.replace("DeepSeek-R1-Distill-Qwen-", "DS-")
        return short.replace("DeepSeek-R1-Distill-", "DS-")
    strategy = "CoT" if cfg.strategy == "cot" else "DA"
    if cfg.instruct_variant:
        prefix = "M" if axis else "Ministral"
        return f"{prefix}{cfg.size}\n{strategy}" if axis else f"Ministral {cfg.size} {strategy}"
    prefix = "Q" if axis else "Qwen2.5-Coder"
    return f"{prefix}{cfg.size}\n{strategy}" if axis else f"Qwen2.5-Coder {cfg.size} {strategy}"


def collect_radar_series(
    configs: list[BaselineConfig | ModelConfig],
    seed: str | None,
) -> dict[tuple[str, tuple], list[ModelRadarSeries]]:
    col_by_key = {
        (column_setup(col), column_exp_name(col), column_pattern(col)): col
        for col in TABLE_COLUMNS
    }
    panel_data: dict[tuple[str, tuple], list[ModelRadarSeries]] = {
        panel_key: [] for panel_key in PANEL_ORDER
    }

    for split in FIGURE_SPLITS:
        for config_key in SOS_CONFIGS:
            col = col_by_key[config_key]
            panel_key = (split, config_key)
            for cfg in configs:
                result_dir = result_dir_for_config(cfg, col)
                result_sets = load_result_sets(result_dir, seed)
                if not result_sets:
                    accuracies = np.full(len(INVALID_ERROR_TYPES), np.nan)
                else:
                    accuracies = averaged_error_type_accuracy(
                        result_sets,
                        cfg.model_name,
                        split,
                        col,
                    )
                panel_data[panel_key].append(
                    ModelRadarSeries(
                        row_id=cfg.row_id,
                        label=radar_model_label(cfg),
                        axis_label=radar_model_label(cfg, axis=True),
                        accuracies=accuracies,
                    )
                )
    return panel_data


def accuracies_by_error_type(series: list[ModelRadarSeries]) -> np.ndarray:
    """Shape: (n_error_types, n_models)."""
    matrix = np.vstack([model.accuracies for model in series]).T
    return np.nan_to_num(matrix, nan=0.0)


def align_radar_ticklabels(ax: Axes, theta: np.ndarray) -> None:
    for label, angle in zip(ax.get_xticklabels(), theta):
        angle_deg = np.degrees(angle) % 360
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


def plot_radar_panel(
    ax: Axes,
    theta: np.ndarray,
    series: list[ModelRadarSeries],
    *,
    show_axis_labels: bool,
    tick_fontsize: float,
    radial_fontsize: float,
    line_width: float,
    fill_alpha: float,
) -> None:
    accuracy_matrix = accuracies_by_error_type(series) / 100.0
    model_axis_labels = [model.axis_label for model in series]

    for error_idx, error_type in enumerate(INVALID_ERROR_TYPES):
        color = ERROR_TYPE_COLORS[error_idx]
        values = accuracy_matrix[error_idx]
        ax.plot(theta, values, color=color, linewidth=line_width, alpha=0.95)
        ax.fill(theta, values, color=color, alpha=fill_alpha)

    if show_axis_labels:
        ax.set_varlabels(model_axis_labels)
        for label in ax.get_xticklabels():
            label.set_fontsize(tick_fontsize)
    else:
        ax.set_varlabels([""] * len(series))

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], labels=["20", "40", "60", "80"], fontsize=radial_fontsize)
    ax.set_ylim(0.0, 1.0)
    if show_axis_labels:
        align_radar_ticklabels(ax, theta)


def build_radar_figure(
    panel_data: dict[tuple[str, tuple], list[ModelRadarSeries]],
    *,
    panel_scale: float,
) -> plt.Figure:
    series = panel_data[PANEL_ORDER[0]]
    theta = radar_factory(len(series), frame="polygon")

    n_rows, n_cols = len(FIGURE_SPLITS), len(SOS_CONFIGS)
    panel_size = 3.4 * panel_scale
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size * n_cols + 0.55, panel_size * n_rows + 0.2),
        subplot_kw={"projection": "radar"},
        squeeze=False,
    )

    tick_fontsize = model_axis_label_fontsize(panel_scale)
    radial_fontsize = max(5.0, 6.0 * panel_scale)
    row_title_fontsize = max(8.0, 9.0 * panel_scale)
    line_width = max(0.7, 0.9 * panel_scale)
    fill_alpha = 0.04

    for row, split in enumerate(FIGURE_SPLITS):
        for col_idx, config_key in enumerate(SOS_CONFIGS):
            ax = axes[row, col_idx]
            panel_key = (split, config_key)
            plot_radar_panel(
                ax,
                theta,
                panel_data[panel_key],
                show_axis_labels=col_idx == 0,
                tick_fontsize=tick_fontsize,
                radial_fontsize=radial_fontsize,
                line_width=line_width,
                fill_alpha=fill_alpha,
            )
            if col_idx == 0:
                ax.set_ylabel(
                    SPLIT_LABELS[split],
                    fontsize=row_title_fontsize,
                    labelpad=18,
                )

    fig.suptitle(
        "PredExe per-error-type accuracy on invalid programs (IMP-SOS)",
        fontsize=max(9.0, 10.5 * panel_scale),
        y=1.02,
    )
    fig.subplots_adjust(wspace=0.35, hspace=0.35, top=0.9, left=0.08)
    return fig


def build_legend_figure(*, ncol: int = 5) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 0.65))
    ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=ERROR_TYPE_COLORS[idx],
            linewidth=2.0,
            label=ERROR_TYPE_LABELS[idx + 1],
        )
        for idx in range(len(INVALID_ERROR_TYPES))
    ]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12)
    return fig


def write_figure_tex(output_path: Path) -> Path:
    tex_path = output_path.with_suffix(".tex")
    lines = [
        r"\begin{figure*}[t]",
        r"  \centering",
        r"  \includegraphics[width=\textwidth]{figures/pcp-error-type-radar-legend.pdf}",
        f"  {LEGEND_TO_PANELS_GAP}",
    ]

    split_macros = {
        "human_written": (r"\humanwrit", "figure:pcp-error-type-radar-hw"),
        "fuzzer_generated": (r"\fuzzgen", "figure:pcp-error-type-radar-fuzzgen"),
    }
    config_labels = {
        ("uk", "IMP-SOS", None): "uk-sos",
        ("mk", "IMP-SOS", "KeywordSwap"): "mk-sos-swap",
        ("mk", "IMP-SOS", "KeywordObf"): "mk-sos-obf",
    }

    for row_idx, split in enumerate(FIGURE_SPLITS):
        if row_idx > 0:
            lines.append(f"  {ROW_BREAK_BETWEEN_ROWS}")
        split_macro, _ = split_macros[split]
        for col_idx, config_key in enumerate(SOS_CONFIGS):
            if col_idx > 0:
                lines.append(f"  {PANEL_COLUMN_GAP}")
            pdf_name = panel_pdf_name(split, config_key)
            label_suffix = config_labels[config_key]
            lines.extend(
                [
                    r"  \subcaptionbox{%",
                    rf"    {split_macro}, {SUBCAPTION_MACROS[config_key]}%",
                    rf"    \label{{figure:pcp-error-type-radar-{SPLIT_SLUGS[split]}-{label_suffix}}}}}[{PANEL_LINEWIDTH_FRAC:.2f}\linewidth]{{%",
                    rf"    \includegraphics[width=\linewidth]{{figures/{pdf_name}}}",
                    r"  }%",
                ]
            )

    lines.extend(
        [
            r"  \caption{\FigPCPErrorTypeRadarCaption}",
            r"\end{figure*}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n")
    return tex_path


def panel_pdf_name(split: str, config_key: tuple) -> str:
    setup, exp_name, pattern = config_key
    parts = ["pcp-error-type-radar", SPLIT_SLUGS[split], setup, exp_name]
    if pattern:
        parts.append(pattern)
    return "-".join(parts) + ".pdf"


def save_panel_pdfs(
    panel_data: dict[tuple[str, tuple], list[ModelRadarSeries]],
    *,
    panel_scale: float,
    output_dir: Path,
) -> None:
    series = panel_data[PANEL_ORDER[0]]
    theta = radar_factory(len(series), frame="polygon")

    for split, config_key in PANEL_ORDER:
        fig, ax = plt.subplots(
            figsize=(
                PANEL_FIGSIZE[0] * panel_scale,
                PANEL_FIGSIZE[1] * panel_scale,
            ),
            subplot_kw={"projection": "radar"},
        )
        plot_radar_panel(
            ax,
            theta,
            panel_data[(split, config_key)],
            show_axis_labels=True,
            tick_fontsize=model_axis_label_fontsize(panel_scale),
            radial_fontsize=max(5.0, 6.0 * panel_scale),
            line_width=max(0.7, 0.9 * panel_scale),
            fill_alpha=0.04,
        )
        out = output_dir / panel_pdf_name(split, config_key)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-error-type accuracy radar charts for all PredExe models."
    )
    parser.add_argument(
        "--seed",
        default="all",
        help="Results seed to load (default: all — average across every results-*.jsonl file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RADAR_OUTPUT,
        help=f"Combined radar grid output (default: {DEFAULT_RADAR_OUTPUT})",
    )
    parser.add_argument(
        "--legend-output",
        type=Path,
        default=DEFAULT_LEGEND_OUTPUT,
        help=f"Legend output (default: {DEFAULT_LEGEND_OUTPUT})",
    )
    parser.add_argument(
        "--panel-scale",
        type=float,
        default=1.0,
        help="Scale factor for panel size (default: 1.0)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the combined figure interactively",
    )
    args = parser.parse_args()

    seed = None if args.seed == "all" else args.seed
    row_configs = discover_all_row_configs()
    if not row_configs:
        print("No model configurations found.")
        raise SystemExit(1)

    panel_data = collect_radar_series(row_configs, seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = build_radar_figure(panel_data, panel_scale=args.panel_scale)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"Saved combined figure to {args.output}")

    legend_fig = build_legend_figure()
    legend_fig.savefig(args.legend_output, bbox_inches="tight", pad_inches=0.02)
    legend_fig.savefig(args.legend_output.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved legend to {args.legend_output}")

    save_panel_pdfs(
        panel_data,
        panel_scale=args.panel_scale,
        output_dir=args.output.parent,
    )
    tex_path = write_figure_tex(args.output)
    print(f"Saved LaTeX wrapper to {tex_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(legend_fig)


if __name__ == "__main__":
    main()
