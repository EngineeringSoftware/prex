<div align="center">
  <p>The 2nd International Workshop on Language Models and Programming Languages (LMPL 2026), Oakland, USA</p>
  <h1>Predicting Program Exit Code with LLMs and Programming Language Semantics</h1>

  <p style="font-size: 20px;">
    by
    <a href="https://l-ma.github.io/">Lara Marinov</a><sup>1</sup>,
    <a href="https://www.adityathimmaiah.com">Aditya Thimmaiah</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=HtNfeKYAAAAJ&hl=en">Jayanth Srinivasa</a><sup>2</sup>,
    <a href="https://www.jessyli.com">Junyi Jessy Li</a><sup>1</sup>,
    <a href="https://users.ece.utexas.edu/~gligoric/">Milos Gligoric</a><sup>1</sup>
  </p>

  <p>
    <sup>1</sup>The University of Texas at Austin &nbsp;&nbsp;&nbsp;
    <sup>2</sup>Cisco Research
  </p>
</div>

<p align="center"><a href="https://arxiv.org/pdf/2609.00579v1"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2510.03415v3-b31b1b.svg"></a> <a href="https://github.com/EngineeringSoftware/prex"><img alt="Code" src="https://img.shields.io/badge/Code-GitHub-black"></a></p>

## About
Do LLMs actually understand programming language semantics when completing software engineering tasks or do they just fall back on patterns they've memorized in training?

To answer this, we introduce Program Executability Prediction (PrEx): given a program’s syntax and operational semantics, decide whether the program is semantically valid or invalid—and if invalid, which formal rule it violates.

## Installation and Usage
The details for requirements and setup are contained in the [PLSemanticsBench](https://github.com/EngineeringSoftware/PLSemanticsBench/tree/main) repository:
  - [Installation](https://github.com/EngineeringSoftware/PLSemanticsBench#installation)
  - [Usage](https://github.com/EngineeringSoftware/PLSemanticsBench#detailed-usage)
<br>

Basic usage to rerun experiments:
  - `python src/main.py experiment <experiment_name>`

Example:
  - `python src/main.py experiment qwen_coder_32b_uk_pcp_imp_k`

For CoT/multi-seed runs, add --seed:
  - `python src/main.py experiment qwen_coder_3b_mk_pcp_cot_imp_sos --seed 73`
<br>

`<experiment_name>` is any registered subcommand in `src/llm_interpreter/experiments/functions.py` (e.g. `qwen_coder_*_pcp_*, ministral_*_pcp_*, deepseek_*_pcp_*`).

## Citation
```bibtex
@inproceedings{MarinovETAL26PrEx,
  author = {Marinov, Lara and Thimmaiah, Aditya and Srinivasa, Jayanth and Li, Junyi Jessy and Gligoric, Milos},
  title = {Predicting Program Exit Code with {LLMs} and Programming Language Semantics},
  booktitle = {International Workshop on Language Models and Programming Languages},
  pages = {To appear},
  year = {2026},
}
```
