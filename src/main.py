"""
Main script for running experiments, tests, data processing and analysis.

The code is organized into the following modules:
1. Data processing: data/functions.py - Dataset creation and preprocessing
2. Experiments: experiments/functions.py - Core experiment implementations
3. Results analysis: results/ - Analyzing and processing experiment results
4. LaTeX: latex/ - Scripts for generating paper tables and figures

Usage:
python main.py <category> <method_name>

Categories:
- DATA: Dataset operations
- EXPERIMENT: Run experiments
- RESULTS: Analyze results
- TEST: Run tests
- LATEX: Generate paper assets
"""

import os
import inspect
import seutil as su

from jsonargparse import CLI
from llm_interpreter.utils import Category
from llm_interpreter import Macros
from llm_interpreter.data import functions as data_functions
from llm_interpreter.experiments import functions as exp_functions
from llm_interpreter.results import functions as results_functions
from llm_interpreter.paper_latex import functions as latex_functions

logger = su.log.get_logger(__name__, su.log.INFO)


def make_subcommands():
    subcommands = {c.value: {"_help": c.help_msg} for c in Category}
    modules: list = [
        latex_functions,
        data_functions,
        exp_functions,
        results_functions,
    ]
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if getattr(obj, "_is_subcommand", False):
                category: str = getattr(obj, "_category")
                subcommands[category][name] = obj

    return subcommands


if __name__ == "__main__":
    if not os.path.isdir(Macros.tmp_dir):
        os.mkdir(Macros.tmp_dir)
    if not os.path.isdir(Macros.batch_dir):
        os.mkdir(Macros.batch_dir)
    su.log.setup(Macros.log_file)
    CLI(make_subcommands(), as_positional=False)
