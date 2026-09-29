"""Generate LaTeX appendix tables for IMP semantics rules."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

IMP_LANGUAGE_PATH = (
    Path(__file__).resolve().parents[1] / "language" / "imp_language.py"
)
K_TXT_PATH = Path(__file__).resolve().parents[1] / "language" / "imp" / "k.txt"

_K_TAG_PATTERN = re.compile(r"</?(\w+)>")


@dataclass(frozen=True)
class ParsedKRule:
    rule_id: int
    full_rule_body: str
    nl_description: str


def escape_latex(text: str) -> str:
    """Escape characters that are special in LaTeX text mode."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _latexify_double_quotes(text: str) -> str:
    """Turn ASCII double-quoted strings into LaTeX ``...'' quotes."""
    return re.sub(r'"([^"]*)"', r"``\1''", text)


def _eval_imp_class_dict(attr: str) -> dict:
    """Load a dict literal from ``IMP`` in ``imp_language.py`` without importing it."""
    source = IMP_LANGUAGE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "IMP":
            continue
        for item in node.body:
            if isinstance(item, ast.Assign):
                targets = item.targets
                value = item.value
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                targets = [item.target]
                value = item.value
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id == attr:
                    return ast.literal_eval(value)
    raise ValueError(f"Could not find IMP.{attr} in {IMP_LANGUAGE_PATH}")


def _default_syntax_map() -> dict[str, str]:
    syntax_map = _eval_imp_class_dict("_DEFAULT_SYNTAX_MAP")
    syntax_map = dict(syntax_map)
    syntax_map["name"] = "IMP"
    syntax_map["semantic_type"] = "K"
    return syntax_map


def _apply_syntax_map(text: str, syntax_map: dict[str, str]) -> str:
    return text.format(**syntax_map)


def load_k_txt_rule_bodies() -> dict[int, str]:
    """Load numbered rule bodies from ``imp/k.txt`` (lines after ``rule [N]:``)."""
    text = K_TXT_PATH.read_text(encoding="utf-8")
    bodies: dict[int, str] = {}
    pattern = re.compile(
        r"rule \[(\d+)\]:\s*\n(.*?)(?=\n\s*rule \[|\nendmodule)",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        bodies[int(match.group(1))] = match.group(2).rstrip("\n")
    return bodies


def load_parsed_k_rules() -> list[ParsedKRule]:
    """Load K rules from ``k.txt`` and descriptions from ``imp_language.py``."""
    syntax_map = _default_syntax_map()
    descriptions = _eval_imp_class_dict("RULE_DESCRIPTIONS_NOTATION_COMPREHENSION_K")
    bodies = load_k_txt_rule_bodies()
    rule_ids = sorted(set(descriptions) | set(bodies))

    parsed: list[ParsedKRule] = []
    for rule_id in rule_ids:
        nl_description = _apply_syntax_map(
            descriptions.get(rule_id, ""), syntax_map
        )
        parsed.append(
            ParsedKRule(
                rule_id=rule_id,
                full_rule_body=bodies.get(rule_id, ""),
                nl_description=nl_description,
            )
        )
    return parsed


# K notation tokens rendered in math mode; other text stays in roman.
_K_TOKEN_PATTERN = re.compile(
    r"\.\.\.|"
    r"~>|"
    r"\|->|"
    r"<=Int|"
    r">=Int|"
    r"=/=Int|"
    r"==Int|"
    r"<Int|"
    r">Int|"
    r"\+Int|"
    r"-Int|"
    r"\*Int|"
    r"/Int|"
    r"%Int|"
    r"=>"
)

_MATH_TOKEN_LATEX: dict[str, str] = {
    "...": r"$\ldots$",
    "~>": r"$\leadsto$",
    "|->": r"$\mapsto$",
    "=>": r"$\Rightarrow$",
    "==Int": r"$= \mathrm{Int}$",
    "=/=Int": r"$\neq \mathrm{Int}$",
    "+Int": r"$+\ \mathrm{Int}$",
    "-Int": r"$-_{\mathrm{Int}}$",
    "*Int": r"$\times_{\mathrm{Int}}$",
    "/Int": r"$/\ \mathrm{Int}$",
    "%Int": r"$\%_{\mathrm{Int}}$",
    "<Int": r"$<_{\mathrm{Int}}$",
    ">Int": r"$>_{\mathrm{Int}}$",
    "<=Int": r"$\leq_{\mathrm{Int}}$",
    ">=Int": r"$\geq_{\mathrm{Int}}$",
}


def _math_for_k_token(token: str) -> str:
    if token in _MATH_TOKEN_LATEX:
        return _MATH_TOKEN_LATEX[token]
    return escape_latex(token)


def _format_k_map_update(text: str) -> str:
    """K map/update patterns ``(_ => v)`` as inline math."""
    return re.sub(
        r"\(_\s*=>\s*([^)]+)\)",
        lambda m: rf"$(\_ \Rightarrow {m.group(1).strip()})$",
        text,
    )


def _format_k_text_segment(text: str) -> str:
    """Escape roman text but leave ``$...$`` math literals untouched."""
    text = _latexify_double_quotes(_format_k_map_update(text.strip()))
    if not text:
        return ""
    parts: list[str] = []
    for bit in re.split(r"(\$[^$]+\$)", text):
        if not bit.strip():
            continue
        if bit.startswith("$") and bit.endswith("$"):
            parts.append(bit)
        else:
            parts.append(_format_k_plain_text(bit))
    return " ".join(parts)


def _format_k_notation_content(line: str) -> str:
    """Format K rule content (no configuration-cell wrappers)."""
    line = re.sub(r"\s+", " ", line.strip())
    if not line:
        return ""
    line = _format_k_map_update(line)
    parts: list[str] = []
    pos = 0
    for match in _K_TOKEN_PATTERN.finditer(line):
        if match.start() > pos:
            text = line[pos : match.start()].strip()
            if text:
                parts.append(_format_k_text_segment(text))
        parts.append(_math_for_k_token(match.group(0)))
        pos = match.end()
    if pos < len(line):
        tail = line[pos:].strip()
        if tail:
            parts.append(_format_k_text_segment(tail))
    return " ".join(parts)


def _format_k_plain_text(text: str) -> str:
    """Roman identifiers; bare K wildcard ``_`` in math."""
    text = text.strip()
    if text == "_":
        return r"$\_$"
    return escape_latex(text)


def _format_k_tag(tag: str) -> str:
    """Render K configuration tags literally, e.g. ``<k>`` / ``</rules>``."""
    match = _K_TAG_PATTERN.fullmatch(tag)
    if not match:
        return escape_latex(tag)
    name = escape_latex(match.group(1))
    if tag.startswith("</"):
        return rf"\textless{{}}/{name}\textgreater{{}}"
    return rf"\textless{{}}{name}\textgreater{{}}"


def _format_k_source_line(line: str) -> str:
    """Format one ``k.txt`` line: literal tags plus roman/math rule content."""
    parts: list[str] = []
    for segment in re.split(r"(</?\w+>)", line):
        if not segment:
            continue
        if _K_TAG_PATTERN.fullmatch(segment):
            parts.append(_format_k_tag(segment))
        else:
            parts.append(_format_k_notation_content(segment))
    return " ".join(part for part in parts if part)


# Visible indent after ``rule [N]:`` (\\quad is dropped at line starts in p-cells).
_K_RULE_BODY_INDENT = r"\hspace*{2em}"


def format_full_k_rule_cell(rule_id: int, body: str) -> str:
    """Format a full rule as in ``k.txt`` (``rule [N]:``, cells, ``requires``)."""
    if not body.strip():
        return ""
    lines: list[str] = [rf"rule [{rule_id}]:"]
    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("requires "):
            condition = _format_k_notation_content(stripped[len("requires ") :])
            lines.append(_K_RULE_BODY_INDENT + r"requires " + condition)
        else:
            lines.append(_K_RULE_BODY_INDENT + _format_k_source_line(stripped))
    # Use \newline inside table cells; \\ would end the longtable row.
    return r" \newline ".join(lines)


def format_description_cell(description: str) -> str:
    """Format the natural-language description column."""
    description = re.sub(r"\s+", " ", description.strip())
    return escape_latex(description)


def build_imp_k_rules_table_rows() -> list[str]:
    """Build longtable row lines for numbered K semantics rules."""
    rows: list[str] = []
    for rule in load_parsed_k_rules():
        rule_cell = format_full_k_rule_cell(rule.rule_id, rule.full_rule_body)
        desc_cell = format_description_cell(rule.nl_description)
        rows.append(
            f"{rule.rule_id} & {rule_cell} & {desc_cell}\\\\\\hline"
        )
    return rows


def generate_imp_k_rules_table_tex(output_path: Path) -> None:
    """Write ``imp_k_rules_table.tex`` for the K-framework semantics appendix."""
    rows = build_imp_k_rules_table_rows()

    # Match SOS ``imp_rules_table.tex``: scriptsize, longtable, vertical bars, \\hline rows.
    body_col = r">{\raggedright\arraybackslash\hspace{0pt}}p{"
    lines = [
        r"\begin{scriptsize}%",
        r"\begin{longtable}{|p{0.7cm}|"
        rf"{body_col}8.8cm}}|"
        rf"{body_col}4.5cm}}|}}",
        r" \caption{K-framework rewrite rules used to formalize \IMP. \label{tab:imp-k-rules}}\\",
        r" \toprule%",
        r" \textbf{Rule} & \textbf{Full rule} & \textbf{Description}\\",
        r" \midrule%",
        *rows,
        r" \bottomrule%",
        r"\end{longtable}%",
        r"\end{scriptsize}%",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
