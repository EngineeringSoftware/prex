import sys
import math
import statistics
import pandas as pd
from seutil import io, latex, log
from pathlib import Path
from typing import Any, List
from llm_interpreter.data.dataset_process import PCP_EXCLUDED_PROGRAMS
from llm_interpreter.macros import Macros
from llm_interpreter.language import Language
from llm_interpreter.results.results_analysis import OPResults, gen_op_regression_coefficients, dataset_code_complexity_metrics_to_df
from llm_interpreter.utils import num_tokens_from_string, read_from_json_file
from llm_interpreter.paper_latex.semantics_rules_table import (
    generate_imp_k_rules_table_tex,
)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import defaultdict
#from .figure import collect_metrics_from_op_results

DATA_DIR = Macros.data_dir

logger = log.get_logger(__name__, log.INFO)


class Table:
    ROW_SEP = "<ROW_SEP>"
    COL_SEP = "<COL_SEP>"
    llms2macros = {
        "meta-llama-Llama-3.3-70B-Instruct-da": "\llamaBig",
        "meta-llama-Llama-3.3-70B-Instruct-cot": "\llamaBig-\COT",
        "Qwen-Qwen2.5-Coder-3B-Instruct-da": r"\qwenCoder{3}",
        "Qwen-Qwen2.5-Coder-3B-Instruct-cot": r"\qwenCoder{3}-\COT",
        "Qwen-Qwen2.5-Coder-7B-Instruct-da": r"\qwenCoder{7}",
        "Qwen-Qwen2.5-Coder-7B-Instruct-cot": r"\qwenCoder{7}-\COT",
        "Qwen-Qwen2.5-Coder-14B-Instruct-da": r"\qwenCoder{14}",
        "Qwen-Qwen2.5-Coder-14B-Instruct-cot": r"\qwenCoder{14}-\COT",
        "Qwen-Qwen2.5-Coder-32B-Instruct-da": r"\qwenCoder{32}",
        "Qwen-Qwen2.5-Coder-32B-Instruct-cot": r"\qwenCoder{32}-\COT",
        "gpt-4o-mini-da": "\gptfo",
        "gpt-4o-mini-cot": "\gptfo-\COT",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da": r"\dpskLlama{70}",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da": r"\dpskQwen{14}",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da": r"\dpskQwen{32}",
        "Qwen-QwQ-32B-da": "\qwq",
        "o3-mini-da": "\othree",
        "gpt-5-mini-da": "\gptfivemini",
        "gemini-2.5-pro-da": "\gemini",
        "random-guesser-dataset-distribution-da": "Random",
#        "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct-da": r"\dpskCoderVtLite",
#        "qwq:32b-fp16-da": r"\qwq-fp16",
#        "qwen2.5-coder:14b-instruct-fp16-da": r"\qwenCoder{14}-instruct-fp16",
#        "qwen2.5-coder:32b-instruct-fp16-da": r"\qwenCoder{32}-instruct-fp16",
    }
    datasets2macros = {
        "human_written": r"\humanwrit",
        "synthetic_cpp": r"\llmtrans",
        "fuzzer_generated": r"\fuzzgen",
    }
    metrics2macros = {
        "pcp-uk-IMP-SOS-accuracy": r"\uk",
        "pcp-mk-IMP-SOS-addSub_mulDiv_negateRelation-accuracy": "\ksMk",
        "pcp-mk-IMP-SOS-unseen-accuracy": "\koMk",
        "pcp-mk-IMP-SOS-KeywordSwap-accuracy": r"\KeywordSwap",
        "pcp-mk-IMP-SOS-KeywordObf-accuracy": r"\KeywordObf",
        "pcp-mk-IMP-K-KeywordSwap-accuracy": r"\KeywordSwap",
        "pcp-mk-IMP-K-KeywordObf-accuracy": r"\KeywordObf",
        "op-nk-IMP-SOS-acc": r"\nk",
        "op-uk-IMP-SOS-acc": r"\uk",
        "op-mk-IMP-SOS-unseen-acc": "\koMk",
        "op-mk-IMP-SOS-addSub_mulDiv_negateRelation-acc": r"\ksMk",
        "op-nk-IMP-SOS-synthetic-cpp-acc": r"\nk",
        "op-uk-IMP-SOS-synthetic-cpp-acc": r"\uk",
        "op-mk-IMP-SOS-synthetic-cpp-unseen-acc": "\koMk",
        "op-mk-IMP-SOS-synthetic-cpp-addSub_mulDiv_negateRelation-acc": r"\ksMk",
        "op-nk-IMP-SOS-fuzzer-generated-acc": r"\nk",
        "op-uk-IMP-SOS-fuzzer-generated-acc": r"\uk",
        "op-mk-IMP-SOS-fuzzer-generated-unseen-acc": "\koMk",
        "op-mk-IMP-SOS-fuzzer-generated-addSub_mulDiv_negateRelation-acc": r"\ksMk",
        "op-nk-IMP-SOS-var-acc": r"\nk",
        "op-uk-IMP-SOS-var-acc": r"\uk",
        "op-mk-IMP-SOS-unseen-var-acc": "\koMk",
        "op-mk-IMP-SOS-addSub_mulDiv_negateRelation-var-acc": r"\ksMk",
        "op-nk-IMP-SOS-synthetic-cpp-var-acc": r"\nk",
        "op-uk-IMP-SOS-synthetic-cpp-var-acc": r"\uk",
        "op-mk-IMP-SOS-synthetic-cpp-unseen-var-acc": "\koMk",
        "op-mk-IMP-SOS-synthetic-cpp-addSub_mulDiv_negateRelation-var-acc": r"\ksMk",
        "op-nk-IMP-SOS-fuzzer-generated-var-acc": r"\nk",
        "op-uk-IMP-SOS-fuzzer-generated-var-acc": r"\uk",
        "op-mk-IMP-SOS-fuzzer-generated-unseen-var-acc": "\koMk",
        "op-mk-IMP-SOS-fuzzer-generated-addSub_mulDiv_negateRelation-var-acc": r"\ksMk",
        "pcp-uk-IMP-K-accuracy": r"\uk",
        "pcp-mk-IMP-K-addSub_mulDiv_negateRelation-accuracy": "\ksMk",
        "pcp-mk-IMP-K-unseen-accuracy": "\koMk",
        "op-uk-IMP-K-acc": r"\uk",
        "op-mk-IMP-K-unseen-acc": "\koMk",
        "op-mk-IMP-K-addSub_mulDiv_negateRelation-acc": r"\ksMk",
        "op-uk-IMP-K-var-acc": r"\uk",
        "op-mk-IMP-K-unseen-var-acc": "\koMk",
        "op-mk-IMP-K-addSub_mulDiv_negateRelation-var-acc": r"\ksMk",
        "op-uk-IMP-K-synthetic-cpp-acc": r"\uk",
        "op-mk-IMP-K-synthetic-cpp-unseen-acc": "\koMk",
        "op-mk-IMP-K-synthetic-cpp-addSub_mulDiv_negateRelation-acc": r"\ksMk",
        "op-uk-IMP-K-synthetic-cpp-var-acc": r"\uk",
        "op-mk-IMP-K-synthetic-cpp-unseen-var-acc": "\koMk",
        "op-mk-IMP-K-synthetic-cpp-addSub_mulDiv_negateRelation-var-acc": r"\ksMk",
        "op-uk-IMP-K-fuzzer-generated-acc": r"\uk",
        "op-mk-IMP-K-fuzzer-generated-unseen-acc": "\koMk",
        "op-mk-IMP-K-fuzzer-generated-addSub_mulDiv_negateRelation-acc": r"\ksMk",
        "op-uk-IMP-K-fuzzer-generated-var-acc": r"\uk",
        "op-mk-IMP-K-fuzzer-generated-unseen-var-acc": "\koMk",
        "op-mk-IMP-K-fuzzer-generated-addSub_mulDiv_negateRelation-var-acc": r"\ksMk",
        "srp-uk-IMP-SOS-xmatch-accuracy": r"\uk",
        "srp-mk-IMP-SOS-addSub_mulDiv_negateRelation-xmatch-accuracy": "\ksMk",
        "srp-mk-IMP-SOS-unseen-xmatch-accuracy": "\koMk",
        "srp-uk-IMP-K-xmatch-accuracy": r"\uk",
        "srp-mk-IMP-K-addSub_mulDiv_negateRelation-xmatch-accuracy": "\ksMk",
        "srp-mk-IMP-K-unseen-xmatch-accuracy": "\koMk",
        "etp-uk-IMP-SOS-xmatch-accuracy": r"\uk",
        "etp-uk-IMP-SOS-approx-match-accuracy": r"$\sim$\uk",
        "etp-uk-IMP-SOS-final-state-match": r"$\sigma_{\uk}$",
        "etp-uk-IMP-SOS-stmt-jaccard-avg": "Jaccard \OperationalSemantics",
        "etp-mk-IMP-SOS-addSub_mulDiv_negateRelation-xmatch-accuracy": "\ksMk",
        "etp-mk-IMP-SOS-addSub_mulDiv_negateRelation-stmt-jaccard-avg": "Jaccard \MutatedOperationalSemanticsA",
        "etp-mk-IMP-SOS-addSub_mulDiv_negateRelation-approx-match-accuracy": r"$\sim$\ksMk",
        "etp-mk-IMP-SOS-addSub_mulDiv_negateRelation-final-state-match": r"$\sigma_{\ksMk}$",
        "etp-mk-IMP-SOS-unseen-xmatch-accuracy": "\koMk",
        "etp-mk-IMP-SOS-unseen-stmt-jaccard-avg": "Jaccard \MutatedOperationalSemanticsU",
        "etp-mk-IMP-SOS-unseen-approx-match-accuracy": "$\sim$\koMk",
        "etp-mk-IMP-SOS-unseen-final-state-match": "$\sigma_{\koMk}$",
        "etp-uk-IMP-K-xmatch-accuracy": r"\uk",
        "etp-uk-IMP-K-approx-match-accuracy": r"$\sim$\uk",
        "etp-uk-IMP-K-final-state-match": r"$\sigma_{\uk}$",
        "etp-uk-IMP-K-stmt-jaccard-avg": "Jaccard \OperationalSemantics",
        "etp-mk-IMP-K-addSub_mulDiv_negateRelation-xmatch-accuracy": "\ksMk",
        "etp-mk-IMP-K-addSub_mulDiv_negateRelation-stmt-jaccard-avg": "Jaccard \MutatedOperationalSemanticsA",
        "etp-mk-IMP-K-addSub_mulDiv_negateRelation-approx-match-accuracy": r"$\sim$\ksMk",
        "etp-mk-IMP-K-addSub_mulDiv_negateRelation-final-state-match": r"$\sigma_{\ksMk}$",
        "etp-mk-IMP-K-unseen-xmatch-accuracy": "\koMk",
        "etp-mk-IMP-K-unseen-stmt-jaccard-avg": "Jaccard \MutatedOperationalSemanticsU",
        "etp-mk-IMP-K-unseen-approx-match-accuracy": "$\sim$\koMk",
        "etp-mk-IMP-K-unseen-final-state-match": "$\sigma_{\koMk}$",
        "imp-tokens": "\# Tokens",
        "imp-loc": "\# LOC",
        "imp-vars": "\# Variables",
        "srp-selected-stmts": "\# Chosen Stmt.",
        "srp-rules-per-stmt": "\# Rules/Stmt.",
        "etp-trace-length": "Len. exec. trace",
        ## ---------------------------------------------------------
        "ig-uk-IMP-SOS-EBNF-invalid_pass": "$P$",
        "ig-uk-IMP-SOS-EBNF-invalid_fail_semantic": "$F_{sem}$",
        "ig-uk-IMP-SOS-EBNF-invalid_fail_syntax": "$F_{syn}$",
        "ig-uk-IMP-SOS-EBNF-invalid_fail_other": "$F_{o}$",
        "ig-uk-IMP-SOS-EBNF-invalid_timeout": "$TO$",
        "ig-uk-IMP-SOS-EBNF-valid_pass": "$P$",
        "ig-uk-IMP-SOS-EBNF-valid_fail_syntax": "$F_{syn}$",
        "ig-uk-IMP-SOS-EBNF-valid_fail_semantic_w": "$F_{sem_w}$",
        "ig-uk-IMP-SOS-EBNF-valid_fail_semantic_r": "$F_{sem_r}$",
        "ig-uk-IMP-SOS-EBNF-valid_fail_other": "$F_{o}$",
        "ig-uk-IMP-SOS-EBNF-valid_timeout": "$TO$",
        ## ---------------------------------------------------------
        "ig-uk-IMP-SOS-ANTLR-invalid_pass": "$P$",
        "ig-uk-IMP-SOS-ANTLR-invalid_fail_semantic": "$F_{sem}$",
        "ig-uk-IMP-SOS-ANTLR-invalid_fail_syntax": "$F_{syn}$",
        "ig-uk-IMP-SOS-ANTLR-invalid_fail_other": "$F_{o}$",
        "ig-uk-IMP-SOS-ANTLR-invalid_timeout": "$TO$",
        "ig-uk-IMP-SOS-ANTLR-valid_pass": "$P$",
        "ig-uk-IMP-SOS-ANTLR-valid_fail_syntax": "$F_{syn}$",
        "ig-uk-IMP-SOS-ANTLR-valid_fail_semantic_w": "$F_{sem_w}$",
        "ig-uk-IMP-SOS-ANTLR-valid_fail_semantic_r": "$F_{sem_r}$",
        "ig-uk-IMP-SOS-ANTLR-valid_fail_other": "$F_{o}$",
        "ig-uk-IMP-SOS-ANTLR-valid_timeout": "$TO$",
        "translate-mk-IMP-SOS-KeyWordSwap-acc": "KeyWordSwap-acc",
        "translate-mk-IMP-SOS-KeyWordObf-acc": "KeyWordObf-acc",
        "translate-uk-IMP-SOS-acc": "Accuracy (\\%)",
        "translate-uk-IMP-SOS-malformed-count": "Malformed Count",
        "translate-mk-IMP-SOS-malformed-count": "Malformed Count",
        "TCap-models-translate-uk-IMP-SOS": "Translation metrics for \\LLMs on \\IMP programs under unmutated semantics.",
        "TCap-models-translate-mk-IMP-SOS": "Translation metrics for \\LLMs on \\IMP programs under mutated semantics.",
        "TCap-models-pcp-qwen-coder-IMP-K-IMP-SOS-accuracy": (
            "PCP accuracy (\\%) for \\QwenCoder models on \\IMP programs "
            "under unmutated (\\uk) and mutated (\\KeywordSwap, \\KeywordObf) semantics."
        ),
        ## ---------------------------------------------------------
        "iga-uk-IMP-SOS-invalid_pass": "$P$",
        "iga-uk-IMP-SOS-invalid_fail_semantic": "$F_{sem}$",
        "iga-uk-IMP-SOS-invalid_fail_syntax": "$F_{syn}$",
        "iga-uk-IMP-SOS-invalid_fail_other": "$F_{o}$",
        "iga-uk-IMP-SOS-invalid_timeout": "$TO$",
        "iga-uk-IMP-SOS-valid_pass": "$P$",
        "iga-uk-IMP-SOS-valid_fail_syntax": "$F_{syn}$",
        "iga-uk-IMP-SOS-valid_fail_semantic_w": "$F_{sem_w}$",
        "iga-uk-IMP-SOS-valid_fail_semantic_r": "$F_{sem_r}$",
        "iga-uk-IMP-SOS-valid_fail_other": "$F_{o}$",
        "iga-uk-IMP-SOS-valid_timeout": "$TO$",
        ## ---------------------------------------------------------
        "igaf-uk-IMP-SOS-invalid_pass": "$P$",
        "igaf-uk-IMP-SOS-invalid_fail_semantic": "$F_{sem}$",
        "igaf-uk-IMP-SOS-invalid_fail_syntax": "$F_{syn}$",
        "igaf-uk-IMP-SOS-invalid_fail_other": "$F_{o}$",
        "igaf-uk-IMP-SOS-invalid_timeout": "$TO$",
        "igaf-uk-IMP-SOS-valid_pass": "$P$",
        "igaf-uk-IMP-SOS-valid_fail_syntax": "$F_{syn}$",
        "igaf-uk-IMP-SOS-valid_fail_semantic_w": "$F_{sem_w}$",
        "igaf-uk-IMP-SOS-valid_fail_semantic_r": "$F_{sem_r}$",
        "igaf-uk-IMP-SOS-valid_fail_other": "$F_{o}$",
        "igaf-uk-IMP-SOS-valid_timeout": "$TO$",
        ## ---------------------------------------------------------
        "iga-mk-ks-IMP-SOS-invalid_pass": "$P$",
        "iga-mk-ks-IMP-SOS-invalid_fail_semantic": "$F_{sem}$",
        "iga-mk-ks-IMP-SOS-invalid_fail_syntax": "$F_{syn}$",
        "iga-mk-ks-IMP-SOS-invalid_fail_other": "$F_{o}$",
        "iga-mk-ks-IMP-SOS-invalid_timeout": "$TO$",
        "iga-mk-ks-IMP-SOS-valid_pass": "$P$",
        "iga-mk-ks-IMP-SOS-valid_fail_syntax": "$F_{syn}$",
        "iga-mk-ks-IMP-SOS-valid_fail_semantic_w": "$F_{sem_w}$",
        "iga-mk-ks-IMP-SOS-valid_fail_semantic_r": "$F_{sem_r}$",
        "iga-mk-ks-IMP-SOS-valid_fail_other": "$F_{o}$",
        "iga-mk-ks-IMP-SOS-valid_timeout": "$TO$",
        ## ---------------------------------------------------------
        "iga-mk-ko-IMP-SOS-invalid_pass": "$P$",
        "iga-mk-ko-IMP-SOS-invalid_fail_semantic": "$F_{sem}$",
        "iga-mk-ko-IMP-SOS-invalid_fail_syntax": "$F_{syn}$",
        "iga-mk-ko-IMP-SOS-invalid_fail_other": "$F_{o}$",
        "iga-mk-ko-IMP-SOS-invalid_timeout": "$TO$",
        "iga-mk-ko-IMP-SOS-valid_pass": "$P$",
        "iga-mk-ko-IMP-SOS-valid_fail_syntax": "$F_{syn}$",
        "iga-mk-ko-IMP-SOS-valid_fail_semantic_w": "$F_{sem_w}$",
        "iga-mk-ko-IMP-SOS-valid_fail_semantic_r": "$F_{sem_r}$",
        "iga-mk-ko-IMP-SOS-valid_fail_other": "$F_{o}$",
        "iga-mk-ko-IMP-SOS-valid_timeout": "$TO$",
        ## ---------------------------------------------------------
        "igaf-mk-ks-IMP-SOS-invalid_pass": "$P$",
        "igaf-mk-ks-IMP-SOS-invalid_fail_semantic": "$F_{sem}$",
        "igaf-mk-ks-IMP-SOS-invalid_fail_syntax": "$F_{syn}$",
        "igaf-mk-ks-IMP-SOS-invalid_fail_other": "$F_{o}$",
        "igaf-mk-ks-IMP-SOS-invalid_timeout": "$TO$",
        "igaf-mk-ks-IMP-SOS-valid_pass": "$P$",
        "igaf-mk-ks-IMP-SOS-valid_fail_syntax": "$F_{syn}$",
        "igaf-mk-ks-IMP-SOS-valid_fail_semantic_w": "$F_{sem_w}$",
        "igaf-mk-ks-IMP-SOS-valid_fail_semantic_r": "$F_{sem_r}$",
        "igaf-mk-ks-IMP-SOS-valid_fail_other": "$F_{o}$",
        "igaf-mk-ks-IMP-SOS-valid_timeout": "$TO$",
        ## ---------------------------------------------------------
        "igaf-mk-ko-IMP-SOS-invalid_pass": "$P$",
        "igaf-mk-ko-IMP-SOS-invalid_fail_semantic": "$F_{sem}$",
        "igaf-mk-ko-IMP-SOS-invalid_fail_syntax": "$F_{syn}$",
        "igaf-mk-ko-IMP-SOS-invalid_fail_other": "$F_{o}$",
        "igaf-mk-ko-IMP-SOS-invalid_timeout": "$TO$",
        "igaf-mk-ko-IMP-SOS-valid_pass": "$P$",
        "igaf-mk-ko-IMP-SOS-valid_fail_syntax": "$F_{syn}$",
        "igaf-mk-ko-IMP-SOS-valid_fail_semantic_w": "$F_{sem_w}$",
        "igaf-mk-ko-IMP-SOS-valid_fail_semantic_r": "$F_{sem_r}$",
        "igaf-mk-ko-IMP-SOS-valid_fail_other": "$F_{o}$",
        "igaf-mk-ko-IMP-SOS-valid_timeout": "$TO$",
    }
    non_reasoning_llms = [
        "meta-llama-Llama-3.3-70B-Instruct-da",
        "meta-llama-Llama-3.3-70B-Instruct-cot",
 #       "Qwen-Qwen2.5-Coder-3B-Instruct-da",
 #       "Qwen-Qwen2.5-Coder-3B-Instruct-cot",
 #       "Qwen-Qwen2.5-Coder-7B-Instruct-da",
 #       "Qwen-Qwen2.5-Coder-7B-Instruct-cot",
        "Qwen-Qwen2.5-Coder-14B-Instruct-da",
        "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
        "Qwen-Qwen2.5-Coder-32B-Instruct-da",
        "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
        "gpt-4o-mini-da",
        "gpt-4o-mini-cot",
    ]
    reasoning_llms = [
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
        "gemini-2.5-pro-da",
        "o3-mini-da",
        "Qwen-QwQ-32B-da",
#        "qwq:32b-fp16-da",  # Add this
#        "qwen2.5-coder:14b-instruct-fp16-da",  # Add this
#        "qwen2.5-coder:32b-instruct-fp16-da",  # Add this
#        "qwen2.5-coder:14b-da",  # Add this
#        "qwen2.5-coder:32b-da",  # Add this
#        "qwq:32b-da",  # Add this
        # "gpt-5-mini-da"
    ]
    exp2models = {
        "op-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
            "Qwen-QwQ-32B-da",
            "o3-mini-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "op-IMP-K": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
            "Qwen-QwQ-32B-da",
            "o3-mini-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "op-IMP-K-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
            "Qwen-QwQ-32B-da",
            "o3-mini-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "op-IMP-K-IMP-SOS-synthetic-cpp": [
            "Qwen-QwQ-32B-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "op-IMP-K-IMP-SOS-fuzzer-generated": [
            "Qwen-QwQ-32B-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "op-IMP-odds-per-iqr-SOS-K": [
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
        ],
        "op-IMP-odds-per-iqr-SOS-K-uk-human_written": [
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
        ],
        "op-IMP-odds-per-iqr-SOS-K-uk-synthetic_cpp": [
            "Qwen-QwQ-32B-da",
        ],
        "op-IMP-odds-per-iqr-SOS-K-uk-fuzzer_generated": [
            "Qwen-QwQ-32B-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "op-IMP-odds-per-iqr-SOS-K-nk-human_written": [
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
        ],
        "op-IMP-odds-per-iqr-SOS-K-nk-synthetic_cpp": [
            "Qwen-QwQ-32B-da",
        ],
        "op-IMP-odds-per-iqr-SOS-K-nk-fuzzer_generated": [
            "Qwen-QwQ-32B-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "op-nk-IMP-odds-per-iqr": [
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
        ],
        "pcp-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
            "Qwen-QwQ-32B-da",
            "o3-mini-da",
            # "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "pcp-qwen-coder-IMP-K-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-3B-Instruct-da",
            "Qwen-Qwen2.5-Coder-3B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-7B-Instruct-da",
            "Qwen-Qwen2.5-Coder-7B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
        ],
        "pcp-IMP-K": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
            "Qwen-QwQ-32B-da",
            "o3-mini-da",
            # "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "srp-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
            "Qwen-QwQ-32B-da",
            "o3-mini-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "srp-IMP-K": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
            "Qwen-QwQ-32B-da",
            "o3-mini-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "etp-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
            "Qwen-QwQ-32B-da",
            "o3-mini-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "etp-IMP-K": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct-da",
            "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
            "meta-llama-Llama-3.3-70B-Instruct-da",
            "meta-llama-Llama-3.3-70B-Instruct-cot",
            "gpt-4o-mini-da",
            "gpt-4o-mini-cot",
            ROW_SEP,
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B-da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B-da",
            "Qwen-QwQ-32B-da",
            "o3-mini-da",
            "gpt-5-mini-da",
            "gemini-2.5-pro-da",
        ],
        "ig-uk-IMP-SOS-EBNF": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct-da",
            "gpt-4o-mini-da",
            "o3-mini-da",
        ],
        "ig-uk-IMP-SOS-ANTLR": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct-da",
            "gpt-4o-mini-da",
            "o3-mini-da",
        ],
        "translate-mk-IMP-SOS": [
            "gpt-4o-mini-da",
            "gemini-2.5-pro-preview-05-06-da",
            "qwq:32b-fp16-da",
            "qwen2.5-coder:14b-instruct-fp16-da",
            "qwen2.5-coder:32b-instruct-fp16-da",
        ],
        "translate-uk-IMP-SOS": [
            "qwq:32b-fp16-da",
            "qwen2.5-coder:14b-instruct-fp16-da",
            "qwen2.5-coder:32b-instruct-fp16-da",
        ],
        "iga-uk-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct-da",
            # "gpt-4o-mini-da",
            # "o3-mini-da",
        ],
        "igaf-uk-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct-da",
            # "gpt-4o-mini-da",
            # "o3-mini-da",
        ],
        "iga-mk-ks-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct-da",
            # "gpt-4o-mini-da",
            # "o3-mini-da",
        ],
        "iga-mk-ko-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct-da",
            # "gpt-4o-mini-da",
            # "o3-mini-da",
        ],
        "igaf-mk-ks-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct-da",
            # "gpt-4o-mini-da",
            # "o3-mini-da",
        ],
        "igaf-mk-ko-IMP-SOS": [
            "Qwen-Qwen2.5-Coder-14B-Instruct-da",
            # "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct-da",
            # "gpt-4o-mini-da",
            # "o3-mini-da",
        ],
    }

    exps = {
        "pcp-IMP-SOS": ["pcp-uk-IMP-SOS", "pcp-mk-IMP-SOS"],
        "pcp-IMP-K": ["pcp-uk-IMP-K", "pcp-mk-IMP-K"],
        "op-IMP-SOS": [
            "op-nk-IMP-SOS",
            "op-uk-IMP-SOS",
            "op-mk-IMP-SOS",
        ],
        "op-IMP-SOS-synthetic-cpp": [
            "op-nk-IMP-SOS-synthetic-cpp",
            "op-uk-IMP-SOS-synthetic-cpp",
            "op-mk-IMP-SOS-synthetic-cpp",
        ],
        "op-IMP-K-synthetic-cpp": [
            "op-nk-IMP-SOS-synthetic-cpp",
            "op-uk-IMP-K-synthetic-cpp",
            "op-mk-IMP-K-synthetic-cpp",
        ],
        "op-IMP-SOS-fuzzer-generated": [
            "op-nk-IMP-SOS-fuzzer-generated",
            "op-uk-IMP-SOS-fuzzer-generated",
            "op-mk-IMP-SOS-fuzzer-generated",
        ],
        "op-IMP-K-fuzzer-generated": [
            "op-nk-IMP-SOS-fuzzer-generated",
            "op-uk-IMP-K-fuzzer-generated",
            "op-mk-IMP-K-fuzzer-generated",
        ],
        "op-IMP-K": [
            "op-nk-IMP-SOS",
            "op-uk-IMP-K",
            "op-mk-IMP-K",
        ],
        "srp-IMP-SOS": ["srp-uk-IMP-SOS", "srp-mk-IMP-SOS"],
        "srp-IMP-K": ["srp-uk-IMP-K", "srp-mk-IMP-K"],
        "etp-IMP-SOS": ["etp-uk-IMP-SOS", "etp-mk-IMP-SOS"],
        "etp-IMP-K": ["etp-uk-IMP-K", "etp-mk-IMP-K"],
        "ig-IMP-SOS-EBNF": ["ig-uk-IMP-SOS-EBNF"],
        "ig-IMP-SOS-ANTLR": ["ig-uk-IMP-SOS-ANTLR"],
        "translate-mk-IMP-SOS": ["translate-mk-IMP-SOS"],
        "translate-uk-IMP-SOS": ["translate-uk-IMP-SOS"],
    }
    tasks2metricsnum = {
        "pcp": 3,
        "op": 4,
        "srp": 3,
        "etp": 3,
    }
    exp2metrics = {
        "op-uk-IMP-SOS": {"acc": ["acc"], "var-acc": ["var-acc"]},
        "op-nk-IMP-SOS": {"acc": ["acc"], "var-acc": ["var-acc"]},
        "op-mk-IMP-SOS": {"acc": ["addSub_mulDiv_negateRelation-acc",  "unseen-acc"], "var-acc": ["addSub_mulDiv_negateRelation-var-acc", "unseen-var-acc"]},
        "op-uk-IMP-K": {"acc": ["acc"], "var-acc": ["var-acc"]},
        "op-mk-IMP-K": {"acc": ["addSub_mulDiv_negateRelation-acc",  "unseen-acc"], "var-acc": ["addSub_mulDiv_negateRelation-var-acc","unseen-var-acc"]},
        "op-uk-IMP-SOS-synthetic-cpp": {"acc": ["acc"], "var-acc": ["var-acc"]},
        "op-nk-IMP-SOS-synthetic-cpp": {"acc": ["acc"], "var-acc": ["var-acc"]},
        "op-mk-IMP-SOS-synthetic-cpp": {"acc": ["addSub_mulDiv_negateRelation-acc",  "unseen-acc"], "var-acc": ["addSub_mulDiv_negateRelation-var-acc", "unseen-var-acc"]},
        "op-uk-IMP-K-synthetic-cpp": {"acc": ["acc"], "var-acc": ["var-acc"]},
        "op-mk-IMP-K-synthetic-cpp": {"acc": ["addSub_mulDiv_negateRelation-acc",  "unseen-acc"], "var-acc": ["addSub_mulDiv_negateRelation-var-acc","unseen-var-acc"]},
        "op-uk-IMP-SOS-fuzzer-generated": {"acc": ["acc"], "var-acc": ["var-acc"]},
        "op-nk-IMP-SOS-fuzzer-generated": {"acc": ["acc"], "var-acc": ["var-acc"]},
        "op-mk-IMP-SOS-fuzzer-generated": {"acc": ["addSub_mulDiv_negateRelation-acc",  "unseen-acc"], "var-acc": ["addSub_mulDiv_negateRelation-var-acc", "unseen-var-acc"]},
        "op-uk-IMP-K-fuzzer-generated": {"acc": ["acc"], "var-acc": ["var-acc"]},
        "op-mk-IMP-K-fuzzer-generated": {"acc": ["addSub_mulDiv_negateRelation-acc",  "unseen-acc"], "var-acc": ["addSub_mulDiv_negateRelation-var-acc","unseen-var-acc"]},
        "pcp-mk-IMP-SOS": {
            "accuracy": [
                "addSub_mulDiv_negateRelation-accuracy",
                # "addSub_mulDiv_negateRelation-rule-accuracy",
                "unseen-accuracy",
                # "unseen-rule-accuracy",
            ]
        },
        "pcp-uk-IMP-SOS": {
            "accuracy": ["accuracy"],
            # "rule-accuracy",
        },
        "pcp-mk-IMP-K":  {
            "accuracy": [
                "addSub_mulDiv_negateRelation-accuracy",
                # "addSub_mulDiv_negateRelation-rule-accuracy",
                "unseen-accuracy",
                # "unseen-rule-accuracy",
            ]
        },
        "pcp-uk-IMP-K": {
            "accuracy": ["accuracy"],
            # "rule-accuracy",
        },
        "srp-uk-IMP-SOS": {"xmatch-accuracy": ["xmatch-accuracy"]},
        "srp-mk-IMP-SOS": {
            "xmatch-accuracy": [
                "addSub_mulDiv_negateRelation-xmatch-accuracy",
                "unseen-xmatch-accuracy",
            ]
        },
        "srp-uk-IMP-K": {"xmatch-accuracy": ["xmatch-accuracy"]},
        "srp-mk-IMP-K": {
            "xmatch-accuracy": [
                "addSub_mulDiv_negateRelation-xmatch-accuracy",
                "unseen-xmatch-accuracy",
            ]
        },
        "etp-uk-IMP-SOS": {"xmatch-accuracy":["xmatch-accuracy"], "approx-match-accuracy":["approx-match-accuracy"], "final-state-match":["final-state-match"]},
        "etp-mk-IMP-SOS": {
            "xmatch-accuracy": [
                "addSub_mulDiv_negateRelation-xmatch-accuracy",
                "unseen-xmatch-accuracy",
            ],
            "approx-match-accuracy": [
                "addSub_mulDiv_negateRelation-approx-match-accuracy",
                "unseen-approx-match-accuracy"
            ],
            "final-state-match": [
                "addSub_mulDiv_negateRelation-final-state-match",
                # "addSub_mulDiv_negateRelation-stmt-jaccard-avg",,
                "unseen-final-state-match",
                # "unseen-stmt-jaccard-avg",
            ]
        },
        "etp-uk-IMP-K": {"xmatch-accuracy":["xmatch-accuracy"], "approx-match-accuracy":["approx-match-accuracy"], "final-state-match":["final-state-match"]},
        "etp-mk-IMP-K": {
            "xmatch-accuracy": [
                "addSub_mulDiv_negateRelation-xmatch-accuracy",
                "unseen-xmatch-accuracy",
            ],
            "approx-match-accuracy": [
                "addSub_mulDiv_negateRelation-approx-match-accuracy",
                "unseen-approx-match-accuracy"
            ],
            "final-state-match": [
                "addSub_mulDiv_negateRelation-final-state-match",
                # "addSub_mulDiv_negateRelation-stmt-jaccard-avg",,
                "unseen-final-state-match",
                # "unseen-stmt-jaccard-avg",
            ]
        },
        "ig-uk-IMP-SOS-ANTLR": [
            "loc",
            "compilation_pass",
            "valid_pass",
            "valid_fail_semantic_w",
            "valid_fail_semantic_r",
            "valid_fail_syntax",
            "valid_fail_other",
            "valid_timeout",
            "invalid_pass",
            "invalid_fail_semantic",
            "invalid_fail_syntax",
            "invalid_fail_other",
            "invalid_timeout",
        ],
        "ig-uk-IMP-SOS-EBNF": [
            "loc",
            "compilation_pass",
            "valid_pass",
            "valid_fail_semantic_w",
            "valid_fail_semantic_r",
            "valid_fail_syntax",
            "valid_fail_other",
            "valid_timeout",
            "invalid_pass",
            "invalid_fail_semantic",
            "invalid_fail_syntax",
            "invalid_fail_other",
            "invalid_timeout",
        ],
        "translate-mk-IMP-SOS": [
            "KeyWordSwap-acc",
            "KeyWordObf-acc",
            "malformed-count",
        ],
        "translate-uk-IMP-SOS": ["acc", "malformed-count"],
        "iga-uk-IMP-SOS": [
            "loc",
            "compilation_pass",
            "valid_pass",
            "valid_fail_semantic_w",
            "valid_fail_semantic_r",
            "valid_fail_syntax",
            "valid_fail_other",
            "valid_timeout",
            "invalid_pass",
            "invalid_fail_semantic",
            "invalid_fail_syntax",
            "invalid_fail_other",
            "invalid_timeout",
        ],
        "igaf-uk-IMP-SOS": [
            "loc",
            "compilation_pass",
            "valid_pass",
            "valid_fail_semantic_w",
            "valid_fail_semantic_r",
            "valid_fail_syntax",
            "valid_fail_other",
            "valid_timeout",
            "invalid_pass",
            "invalid_fail_semantic",
            "invalid_fail_syntax",
            "invalid_fail_other",
            "invalid_timeout",
        ],
        "iga-mk-ks-IMP-SOS": [
            "loc",
            "compilation_pass",
            "valid_pass",
            "valid_fail_semantic_w",
            "valid_fail_semantic_r",
            "valid_fail_syntax",
            "valid_fail_other",
            "valid_timeout",
            "invalid_pass",
            "invalid_fail_semantic",
            "invalid_fail_syntax",
            "invalid_fail_other",
            "invalid_timeout",
        ],
        "iga-mk-ko-IMP-SOS": [
            "loc",
            "compilation_pass",
            "valid_pass",
            "valid_fail_semantic_w",
            "valid_fail_semantic_r",
            "valid_fail_syntax",
            "valid_fail_other",
            "valid_timeout",
            "invalid_pass",
            "invalid_fail_semantic",
            "invalid_fail_syntax",
            "invalid_fail_other",
            "invalid_timeout",
        ],
        "igaf-mk-ks-IMP-SOS": [
            "loc",
            "compilation_pass",
            "valid_pass",
            "valid_fail_semantic_w",
            "valid_fail_semantic_r",
            "valid_fail_syntax",
            "valid_fail_other",
            "valid_timeout",
            "invalid_pass",
            "invalid_fail_semantic",
            "invalid_fail_syntax",
            "invalid_fail_other",
            "invalid_timeout",
        ],
        "igaf-mk-ko-IMP-SOS": [
            "loc",
            "compilation_pass",
            "valid_pass",
            "valid_fail_semantic_w",
            "valid_fail_semantic_r",
            "valid_fail_syntax",
            "valid_fail_other",
            "valid_timeout",
            "invalid_pass",
            "invalid_fail_semantic",
            "invalid_fail_syntax",
            "invalid_fail_other",
            "invalid_timeout",
        ],
    }
    ds2metrics = {
        "imp-dataset": [
            "num-programs",
            "avg-loc",
            "avg-tokens",
            "avg-trace-length",
            "max-trace-length",
            "min-trace-length",
            "avg-rules-per-state",
            "max-rules-per-state",
            "min-rules-per-state",
        ],
        "mk-IMP-SOS": [
            "addSub",
            "mulDiv",
            "negateRelation",
            "addSub-mulDiv",
            "addSub-negateRelation",
            "mulDiv-negateRelation",
            "addSub-mulDiv-negateRelation",
        ],
    }
    ds2ds = {
        "imp-dataset": "imp-dataset",
        "mk-IMP-SOS": ["mk-IMP-SOS-mutation-rate", "mk-IMP-SOS-num-mutated-programs"],
    }

    data_tables = ["mutated-imp-dataset"]

    def __init__(self, paper: str = "arxiv"):
        if paper not in Macros.papers:
            raise ValueError(
                f"Paper {paper} is not supported. Available: {Macros.papers}"
            )
        self.paper = paper
        self.paper_dir = Macros.papers_dir / self.paper
        self.tables_dir: Path = self.paper_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir: Path = Macros.results_dir

    def make_numbers(self, task: str, exp_name: str, dataset_name: str = "human_written"):
        """
        Add numbers to latex macro file.
        """
        def get_mutation_counterpart(k: str):
            if 'addSub_mulDiv_negateRelation' in k:
                return k.replace('addSub_mulDiv_negateRelation', 'unseen')
            else:
                return k.replace('unseen', 'addSub_mulDiv_negateRelation')
            #fi
        #fed
        exp_id = f"{task}-{exp_name}" if dataset_name == "human_written" else f"{task}-{exp_name}-{dataset_name.replace('_','-')}"
        f = latex.File(
            self.tables_dir / f"numbers-{exp_id}.tex",
        )
        model_values: dict = {}
        for exp in self.exps[exp_id]:
            file_name = f"metrics-{exp}.json"
            results = io.load(self.results_dir / file_name)
            for model, res in results.items():
                for k, val in res.items():
                    if k == 'acc' or k == 'var-acc':
                        model_values[f"{model}-{exp}-{k}"] = round(val * 100)
                    elif 'xmatch-accuracy' in k and task == 'srp':
                        model_values[f"{model}-{exp}-{k}"] = round(val * 100)
                    #fi
                #rof
            #rof
        #rof


        for exp in self.exps[exp_id]:
            if task == 'op' and exp_name == 'IMP-K' and '-nk-' in exp:
                continue
            #fi
            file_name = f"metrics-{exp}.json"
            results = io.load(self.results_dir / file_name)

            for model, res in results.items():
                for k, val in res.items():
                    if isinstance(val, dict):
                        for sub_k, sub_val in val.items():
                            if "percentage-trace-match" in k:
                                sub_val = round(sub_val)
                            fmt = self.infer_fmt(sub_val)
                            f.append_macro(
                                latex.Macro(
                                    f"res-{model}-{exp}-{k}-{sub_k}",
                                    f"{sub_val:{fmt}}",
                                )
                            )
                        continue
                    if val <= 1:
                        val = round(val * 100)
                    #fi
                    fmt = self.infer_fmt(val)
                    if '-uk-' in exp and task == 'op' and (k == 'acc' or k == 'var-acc'):
                        if model_values[f"{model}-{exp}-{k}"] < model_values[f"{model}-{exp.replace('-uk-','-nk-').replace('IMP-K','IMP-SOS')}-{k}"]:
                            f.append_macro(
                                latex.Macro(f"res-{model}-{exp}-{k}", f"\\cellcolor{{red!10}}{val:{fmt}}")
                            )
                        else:
                            f.append_macro(
                                latex.Macro(f"res-{model}-{exp}-{k}", f"\\cellcolor{{green!10}}{val:{fmt}}")
                            )                            
                        #fi
                        continue
                    elif ('-mk-' in exp) and task == 'srp' and ('xmatch-accuracy' in k):
                        if model_values[f"{model}-{exp}-{k}"] < model_values[f"{model}-{exp}-{get_mutation_counterpart(k)}"]:
                            f.append_macro(
                                latex.Macro(f"res-{model}-{exp}-{k}", f"\\cellcolor{{red!10}}{val:{fmt}}")
                            )
                        else:
                            f.append_macro(
                                latex.Macro(f"res-{model}-{exp}-{k}", f"\\cellcolor{{green!10}}{val:{fmt}}")
                            )                            
                        #fi
                        continue
                    #fi
                    f.append_macro(
                        latex.Macro(f"res-{model}-{exp}-{k}", f"{val:{fmt}}")
                    )
        f.save()

    def make_ig_numbers(self, task: str, exp_name: str):
        exp_id = f"{task}-{exp_name}"
        f = latex.File(
            self.tables_dir / f"numbers-{exp_id}.tex",
        )

        file_name = f"metrics-{exp_id}.json"
        results = io.load(self.results_dir / file_name)

        for model, res in results.items():
            for impl_lang, metrics in res.items():
                for i in range(len(metrics)):
                    compiled = metrics[i]["compilation_pass"]
                    for k, v in metrics[i].items():
                        if isinstance(v, bool):
                            v = r"\ding{51}" if v else r"\ding{55}"
                            if impl_lang == "Python":
                                v = "-"
                        f.append_macro(
                            latex.Macro(
                                f"res-{model}-{exp_id}-{impl_lang}-{k}-{i}",
                                f"{v}" if compiled or "valid" not in k else "-",
                            )
                        )
        f.save()

    def make_dataset_numbers(self, dataset: str):
        """
        Add dataset stats numbers to latex macro file.
        """

        f = latex.File(
            self.tables_dir / f"numbers-{dataset}-stats.tex",
        )

        results = io.load(self.results_dir / f"{dataset}-stats.json")

        for k, val in results.items():
            if isinstance(val, dict):
                for sub_k, sub_val in val.items():
                    fmt = self.infer_fmt(sub_val)
                    f.append_macro(
                        latex.Macro(
                            f"dataset-{dataset}-{k}-{sub_k}", f"{sub_val:{fmt}}"
                        )
                    )
                continue
            fmt = self.infer_fmt(val)
            f.append_macro(latex.Macro(f"dataset-{dataset}-{k}", f"{val:{fmt}}"))
        f.save()

    def make_dataset_table(self, ds_name: str):
        data_list = self.ds2ds[ds_name]

        f = latex.File(self.tables_dir / f"table-{ds_name}-stats.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{ds_name}.tex")
        macros = []

        cols = ["dataset"] + self.ds2metrics[ds_name]
        rows = data_list

        f.append(r"\begin{table*}[t]")
        # f.append(r"\begin{small}")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-{ds_name}-dataset-stats").use()
        macros.append(
            latex.Macro(f"TCap-{ds_name}-dataset-stats", "Dataset statistics.")
        )
        label = r"\label{tab:dataset-stats}"
        # add caption and label
        f.append(r"\caption{" + caption + label + "}")
        f.append(r"\begin{tabular}{l |" + " c " * (len(cols) - 1) + "}")
        f.append(r"\toprule")
        f.append(
            r"\multirow{2}{*}{\textbf{"
            + latex.Macro(f"THead-{ds_name}-dataset").use()
            + r"}}"
        )
        first_col = cols.pop(0)
        f.append(r"\textbf{" + latex.Macro(f"THead-{ds_name}-{first_col}").use() + "}")
        macros.append(latex.Macro(f"THead-{first_col}", first_col))
        for col in cols:
            f.append(r" & \textbf{" + latex.Macro(f"THead-{col}").use() + "}")
            macros.append(latex.Macro(f"THead-{col}", col))
        f.append(r"\\")

        f.append(r"\midrule")
        for i, row in enumerate(rows):
            f.append(latex.Macro(f"THead-{row}").use())
            macros.append(latex.Macro(f"THead-{row}", row))
            for col in cols:
                f.append(r" & " + latex.Macro(f"dataset-dataset-{col}").use())
            f.append(r"\\")

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        # f.append(r"\end{small}")
        f.append(r"\end{table*}")
        f.save()

        if not mf.path.exists():
            for def_macro in macros:
                mf.append(def_macro)
            mf.save()

    def make_benchmark_dataset_table(self, ds_name: str):
        f = latex.File(self.tables_dir / f"table-{ds_name}-stats.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{ds_name}.tex")
        macros = []

        rows = ["imp", "srp", "etp"]
        subset2metrics = {
            "imp": ["tokens", "vars", "loc"],
            "srp": ["selected-stmts", "rules-per-stmt"],
            "etp": ["trace-length"],
        }
        cols = ["mean", "max"]

        caption = latex.Macro(f"TCap-{ds_name}-dataset-stats").use()
        label = r"\label{tab:dataset-stats}"
        # add caption and label
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")
        f.append(r"\begin{tabular}{l l c c}")
        f.append(r"\toprule")
        f.append(r" &")
        for col in cols:
            f.append(r"& \textbf{" + latex.Macro(f"THead-{ds_name}-{col}").use() + "}")
            macros.append(latex.Macro(f"THead-{ds_name}-{col}", col.capitalize()))
        f.append(r"\\")
        f.append(r"\midrule")
        for i, row in enumerate(rows):
            if len(subset2metrics[row]) > 1:
                metrics_num = len(subset2metrics[row])
                f.append(
                    r"\multirow{"
                    + str(metrics_num)
                    + r"}{*}{\textbf{"
                    + latex.Macro(f"THead-{row}").use()
                    + r"}}"
                )
            else:
                f.append(r"\textbf{" + latex.Macro(f"THead-{row}").use() + "}")
            macros.append(latex.Macro(f"THead-{row}", "\\" + row))
            # write in metrics
            for metric in subset2metrics[row]:
                f.append(r" & " + latex.Macro(f"THead-{row}-{metric}").use())
                macros.append(
                    latex.Macro(
                        f"THead-{row}-{metric}", self.metrics2macros[f"{row}-{metric}"]
                    )
                )
                for col in cols:
                    f.append(
                        r" & "
                        + latex.Macro(f"dataset-{ds_name}-{row}-{col}-{metric}").use()
                    )
                f.append(r"\\")
            # f.append(r"\\")
            if i != len(rows) - 1:
                f.append(r"\midrule")

            #
        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        # f.append(r"\end{center}")
        # f.append(r"\end{small}")
        # f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()

    PROGRAM_EXECUTABILITY_ERROR_TYPE_LABELS = {
        "continue_outside_loop": "Continue outside loop",
        "modulo_zero": "Modulo by zero",
        "break_outside_loop": "Break outside loop",
        "divide_by_zero": "Divide by zero",
        "var_use_before_declare": "Variable use before declare",
    }
    PROGRAM_EXECUTABILITY_ERROR_TYPE_RULES = {
        "break_outside_loop": {"K": "34", "S": "73"},
        "continue_outside_loop": {"K": "31", "S": "76"},
        "divide_by_zero": {"K": "7", "S": "19"},
        "modulo_zero": {"K": "9", "S": "23"},
        "var_use_before_declare": {"K": "2", "S": "2"},
    }

    def make_program_executability_stats_table(
        self,
        ds_name: str = "program-executability",
        dataset_path: Path | None = None,
    ):
        if dataset_path is None:
            dataset_path = DATA_DIR / "dataset" / "dataset-pcp-uk-IMP-SOS.jsonl"

        records: list[dict] = io.load(dataset_path)
        total = len(records)
        success_count = sum(1 for r in records if r["ans"] == "##success##")
        error_records = [r for r in records if r["ans"] != "##success##"]
        error_type_counts: dict[str, int] = defaultdict(int)
        for record in error_records:
            error_type_counts[record["semantic-error-type"]] += 1

        error_types = sorted(
            error_type_counts,
            key=lambda error_type: (-error_type_counts[error_type], error_type),
        )

        nf = latex.File(self.tables_dir / f"numbers-{ds_name}-stats.tex")
        nf.append_macro(
            latex.Macro(f"dataset-{ds_name}-total-programs", str(total))
        )
        nf.append_macro(
            latex.Macro(f"dataset-{ds_name}-success-count", str(success_count))
        )
        nf.append_macro(
            latex.Macro(
                f"dataset-{ds_name}-success-percent",
                f"{100 * success_count / total:.1f}\\%",
            )
        )
        for error_type in error_types:
            slug = error_type.replace("_", "-")
            count = error_type_counts[error_type]
            nf.append_macro(
                latex.Macro(f"dataset-{ds_name}-{slug}-count", str(count))
            )
            nf.append_macro(
                latex.Macro(
                    f"dataset-{ds_name}-{slug}-percent",
                    f"{100 * count / total:.1f}\\%",
                )
            )
        nf.save()

        mf = latex.File(self.tables_dir / f"macros-table-{ds_name}.tex")
        mf.append_macro(
            latex.Macro(
                f"TCap-{ds_name}-stats",
                r"Breakdown of program executability labels in the extended \pcp dataset.",
            )
        )
        mf.append_macro(latex.Macro("THead-executability", "Executability"))
        mf.append_macro(latex.Macro("THead-semantic-error-type", "Semantic error type"))
        mf.append_macro(latex.Macro("THead-corresponding-rule", "Corresponding Rule"))
        mf.append_macro(latex.Macro("THead-count", "Count"))
        mf.append_macro(latex.Macro("THead-percentage", "Percentage"))
        mf.append_macro(latex.Macro("THead-success", "success"))
        mf.append_macro(latex.Macro("THead-error", "error"))
        mf.append_macro(latex.Macro("THead-none", "---"))
        mf.append_macro(latex.Macro(f"dataset-{ds_name}-success-rule-k", "---"))
        mf.append_macro(latex.Macro(f"dataset-{ds_name}-success-rule-s", "---"))
        for error_type in error_types:
            slug = error_type.replace("_", "-")
            label = self.PROGRAM_EXECUTABILITY_ERROR_TYPE_LABELS.get(
                error_type, error_type.replace("_", " ").capitalize()
            )
            mf.append_macro(latex.Macro(f"THead-semantic-error-{slug}", label))
            rules = self.PROGRAM_EXECUTABILITY_ERROR_TYPE_RULES[error_type]
            mf.append_macro(
                latex.Macro(f"dataset-{ds_name}-{slug}-rule-k", rules["K"])
            )
            mf.append_macro(
                latex.Macro(f"dataset-{ds_name}-{slug}-rule-s", rules["S"])
            )
        mf.save()

        tf = latex.File(self.tables_dir / f"table-{ds_name}-stats.tex")
        tf.append(r"\TableFont")
        caption = latex.Macro(f"TCap-{ds_name}-stats").use()
        tf.append(r"\caption{" + caption + r"}")
        tf.append(r"\label{tab:program-executability-stats}")
        tf.append(r"\setlength{\tabcolsep}{2pt}")
        tf.append(r"\resizebox{\linewidth}{!}{%")
        tf.append(
            r"\begin{tabular}{l l >{\centering\arraybackslash}p{0.5in} "
            r">{\centering\arraybackslash}p{0.5in} r r}"
        )
        tf.append(r"\toprule")
        tf.append(
            r" & & \multicolumn{2}{c}{\textbf{"
            + latex.Macro("THead-corresponding-rule").use()
            + r"}} & & "
        )
        tf.append(r"\\")
        tf.append(r"\cmidrule(lr){3-4}")
        tf.append(
            r"\textbf{"
            + latex.Macro("THead-executability").use()
            + r"}"
            + r" & \textbf{"
            + latex.Macro("THead-semantic-error-type").use()
            + r"}"
            + r" & $\mathbb{K}$"
            + r" & $\mathbb{S}$"
            + r" & \textbf{"
            + latex.Macro("THead-count").use()
            + r"}"
            + r" & \textbf{"
            + latex.Macro("THead-percentage").use()
            + r"}"
        )
        tf.append(r"\\")
        tf.append(r"\midrule")
        tf.append(latex.Macro("THead-success").use())
        tf.append(r" & " + latex.Macro("THead-none").use())
        tf.append(
            r" & "
            + latex.Macro(f"dataset-{ds_name}-success-rule-k").use()
            + r" & "
            + latex.Macro(f"dataset-{ds_name}-success-rule-s").use()
            + r" & "
            + latex.Macro(f"dataset-{ds_name}-success-count").use()
            + r" & "
            + latex.Macro(f"dataset-{ds_name}-success-percent").use()
        )
        tf.append(r"\\")
        for error_type in error_types:
            slug = error_type.replace("_", "-")
            tf.append(latex.Macro("THead-error").use())
            tf.append(r" & " + latex.Macro(f"THead-semantic-error-{slug}").use())
            tf.append(
                r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-rule-k").use()
                + r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-rule-s").use()
                + r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-count").use()
                + r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-percent").use()
            )
            tf.append(r"\\")
        tf.append(r"\bottomrule")
        tf.append(r"\end{tabular}%")
        tf.append(r"}")
        tf.save()

    IMP_SPLIT_LOC_DIRS = {
        "human_written": (
            DATA_DIR / "imp/valid_imp_programs/human_written",
            "Human-Written",
        ),
        "synthetic_cpp": (
            DATA_DIR / "imp/valid_imp_programs/synthetic_cpp",
            "LLM-Translated",
        ),
        "fuzzer_generated": (
            DATA_DIR / "imp/fuzzer_generated",
            "Fuzzer-Generated",
        ),
    }

    def make_imp_split_loc_stats_table(self, ds_name: str = "imp-split-loc"):
        split_stats: dict[str, dict[str, int]] = {}
        for split_key, (imp_dir, _) in self.IMP_SPLIT_LOC_DIRS.items():
            if not imp_dir.is_dir():
                raise FileNotFoundError(f"Missing IMP split directory: {imp_dir}")
            imp_paths = [
                path
                for path in imp_dir.glob("*.imp")
                if path.name not in PCP_EXCLUDED_PROGRAMS
            ]
            programs = [path.read_text() for path in imp_paths]
            line_counts = [len(program.splitlines()) for program in programs]
            token_counts = [num_tokens_from_string(program) for program in programs]
            if not line_counts:
                raise ValueError(f"No IMP programs found in {imp_dir}")
            split_stats[split_key] = {
                "num-programs": len(line_counts),
                "min-loc": min(line_counts),
                "median-loc": int(round(statistics.median(line_counts))),
                "max-loc": max(line_counts),
                "min-tokens": min(token_counts),
                "median-tokens": int(round(statistics.median(token_counts))),
                "max-tokens": max(token_counts),
            }

        nf = latex.File(self.tables_dir / f"numbers-{ds_name}-stats.tex")
        for split_key, stats in split_stats.items():
            slug = split_key.replace("_", "-")
            for metric, value in stats.items():
                fmt = self.infer_fmt(value)
                nf.append_macro(
                    latex.Macro(
                        f"dataset-{ds_name}-{slug}-{metric}",
                        f"{value:{fmt}}",
                    )
                )
        nf.save()

        mf = latex.File(self.tables_dir / f"macros-table-{ds_name}.tex")
        mf.append_macro(
            latex.Macro(
                f"TCap-{ds_name}-stats",
                r"Program size per dataset split. LOC counts source lines in each valid base \IMP program. Token counts use the GPT-4o tokenizer",
            )
        )
        column_headers = {
            "split": "Split",
            "num-programs": r"\# programs",
            "loc": "LOC",
            "tokens": "Tokens",
            "min": "Min",
            "median": "Median",
            "max": "Max",
        }
        for header_key, header_label in column_headers.items():
            mf.append_macro(
                latex.Macro(f"THead-{ds_name}-{header_key}", header_label)
            )
        for split_key, (_, label) in self.IMP_SPLIT_LOC_DIRS.items():
            slug = split_key.replace("_", "-")
            mf.append_macro(latex.Macro(f"THead-{ds_name}-{slug}", label))
        mf.save()

        tf = latex.File(self.tables_dir / f"table-{ds_name}-stats.tex")
        tf.append(r"\TableFont")
        caption = latex.Macro(f"TCap-{ds_name}-stats").use()
        tf.append(r"\caption{" + caption + r"}")
        tf.append(r"\label{tab:imp-split-loc-stats}")
        tf.append(r"\setlength{\tabcolsep}{3pt}")
        tf.append(r"\begin{tabular}{l r r r r r r r}")
        tf.append(r"\toprule")
        tf.append(
            r" & & \multicolumn{3}{c}{\textbf{"
            + latex.Macro(f"THead-{ds_name}-loc").use()
            + r"}} & \multicolumn{3}{c}{\textbf{"
            + latex.Macro(f"THead-{ds_name}-tokens").use()
            + r"}}"
        )
        tf.append(r"\\")
        tf.append(r"\cmidrule(lr){3-5}")
        tf.append(r"\cmidrule(lr){6-8}")
        tf.append(
            r"\textbf{"
            + latex.Macro(f"THead-{ds_name}-split").use()
            + r"} & \textbf{"
            + latex.Macro(f"THead-{ds_name}-num-programs").use()
            + r"} & \textbf{"
            + latex.Macro(f"THead-{ds_name}-min").use()
            + r"} & \textbf{"
            + latex.Macro(f"THead-{ds_name}-median").use()
            + r"} & \textbf{"
            + latex.Macro(f"THead-{ds_name}-max").use()
            + r"} & \textbf{"
            + latex.Macro(f"THead-{ds_name}-min").use()
            + r"} & \textbf{"
            + latex.Macro(f"THead-{ds_name}-median").use()
            + r"} & \textbf{"
            + latex.Macro(f"THead-{ds_name}-max").use()
            + r"}"
        )
        tf.append(r"\\")
        tf.append(r"\midrule")
        for split_key in self.IMP_SPLIT_LOC_DIRS:
            slug = split_key.replace("_", "-")
            tf.append(latex.Macro(f"THead-{ds_name}-{slug}").use())
            tf.append(
                r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-num-programs").use()
                + r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-min-loc").use()
                + r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-median-loc").use()
                + r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-max-loc").use()
                + r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-min-tokens").use()
                + r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-median-tokens").use()
                + r" & "
                + latex.Macro(f"dataset-{ds_name}-{slug}-max-tokens").use()
            )
            tf.append(r"\\")
        tf.append(r"\bottomrule")
        tf.append(r"\end{tabular}")
        tf.save()

    @staticmethod
    def _primary_metrics(ems):
        # exp2metrics entries may be a flat list (legacy) or a dict mapping a
        # primary-metric key to its list of reported sub-metrics (e.g. acc -> [
        # "addSub_mulDiv_negateRelation-acc", "unseen-acc"]). Tables produced by
        # make_result_table / make_two_results_table only display the primary
        # metric, so flatten dicts to the first key's sub-metric list.
        if isinstance(ems, dict):
            if not ems:
                return []
            return list(next(iter(ems.values())))
        return list(ems)

    def make_result_table(self, task: str, exp_name: str):
        exp_id = f"{task}-{exp_name}"
        exps = self.exps[exp_id]
        model_list = self.exp2models[exp_id]

        metrics_2_best_models = {}
        for exp in exps:
            best_model_file = f"metrics-best-model-{exp}.json"
            best_models = io.load(self.results_dir / best_model_file)
            metrics_2_best_models.update(
                {f"{exp}-{metric}": models for metric, models in best_models.items()}
            )
        #

        f = latex.File(self.tables_dir / f"table-results-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []

        cols = ["models"]
        for exp in exps:
            ems = self._primary_metrics(self.exp2metrics[exp])
            cols += [f"{exp}-{metric}" for metric in ems]
        rows = model_list
        f.append(r"\begin{table*}[t]")
        # f.append(r"\begin{small}")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        macros.append(
            latex.Macro(
                f"TCap-models-{exp_id}",
                self.metrics2macros.get(
                    f"TCap-models-{exp_id}", "Results of the models."
                ),
            )
        )
        label = r"\label{tab:" + exp_id + "}"
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")

        f.append(r"\begin{tabular}{l |" + " c " * (len(cols) - 1) + "}")
        f.append(r"\toprule")
        first_col = cols.pop(0)
        f.append(r"\textbf{" + latex.Macro(f"THead-{exp_id}-{first_col}").use() + "}")
        macros.append(latex.Macro(f"THead-{exp_id}-{first_col}", "Models"))
        for ci, col in enumerate(cols):
            f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
            macros.append(
                latex.Macro(f"THead-{exp_id}-{col}", self.metrics2macros.get(col, col))
            )
        f.append(r"\\")

        f.append(r"\midrule")
        for i, row in enumerate(rows):
            if row == self.ROW_SEP:
                f.append(r"\midrule")
                continue
            f.append(latex.Macro(f"THead-{exp_id}-{row}").use())
            macros.append(
                latex.Macro(f"THead-{exp_id}-{row}", self.llms2macros.get(row, row))
            )
            for ci, col in enumerate(cols):
                if col in metrics_2_best_models:
                    best_models = metrics_2_best_models[col]
                    if row in best_models:
                        f.append(
                            r" & \textbf{"
                            + latex.Macro(f"res-{row}-{col}").use()
                            + r"}"
                        )
                    else:
                        f.append(r" & " + latex.Macro(f"res-{row}-{col}").use())
            f.append(r"\\")

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        # f.append(r"\end{small}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()

    def make_two_results_table(self, tasks: List[str], exp_name: str = "IMP-SOS"):
        model_list = self.non_reasoning_llms + [self.ROW_SEP] + self.reasoning_llms
        exps = []
        for task in tasks:
            exp_id = f"{task}-{exp_name}"
            exps.extend(self.exps[exp_id])

        metrics_2_best_models = {}
        for exp in exps:
            best_model_file = f"metrics-best-model-{exp}.json"
            best_models = io.load(self.results_dir / best_model_file)
            metrics_2_best_models.update(
                {f"{exp}-{metric}": models for metric, models in best_models.items()}
            )
        #
        exp_id = f"{'-'.join(tasks)}-{exp_name}"
        f = latex.File(self.tables_dir / f"table-results-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []

        cols = ["models"]
        task_prefix = ""
        first_half_cnt, second_half_cnt = 0, 0
        for exp in exps:
            if task_prefix and exp.split("-")[0] != task_prefix:
                first_half_cnt = len(cols) - 1
                cols.append(self.COL_SEP)
            #
            task_prefix = exp.split("-")[0]
            ems = self._primary_metrics(self.exp2metrics[exp])
            cols += [f"{exp}-{metric}" for metric in ems]
        second_half_cnt = len(cols) - first_half_cnt - 2
        rows = model_list
        f.append(r"\begin{table*}[t]")
        # f.append(r"\begin{small}")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        label = r"\label{tab:" + exp_id + "}"
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")

        f.append(r"\begin{tabular}{l l " + " c " * (len(cols) - 1) + "}")
        f.append(r"\toprule")
        f.append(r"&")  # empty cell for rotate box
        first_col = cols.pop(0)
        f.append(
            r"\multirow{2}{*}{\textbf{"
            + latex.Macro(f"THead-{exp_id}-{first_col}").use()
            + r"}}"
        )
        macros.append(latex.Macro(f"THead-{exp_id}-{first_col}", "Models"))
        f.append(
            r"& \multicolumn{"
            + str(first_half_cnt)
            + "}{c}{\\textbf{\\"
            + tasks[0]
            + "}}"
        )
        f.append(
            r"& \multicolumn{"
            + str(second_half_cnt)
            + "}{c}{\\textbf{\\"
            + tasks[1]
            + "}}"
        )
        f.append(r"\\")
        f.append(r"\cmidrule(lr){3-" + str(first_half_cnt + 2) + "}")
        f.append(
            r"\cmidrule(lr){" + str(first_half_cnt + 3) + "-" + str(len(cols) + 1) + "}"
        )
        f.append(r"&")  # empty cell for rotate box
        for ci, col in enumerate(cols):
            if col == self.COL_SEP:
                continue
            f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
            macros.append(
                latex.Macro(f"THead-{exp_id}-{col}", self.metrics2macros.get(col, col))
            )
        f.append(r"\\")

        f.append(r"\midrule")
        # rotate box
        f.append(
            r"\multirow{"
            + str(len(self.non_reasoning_llms))
            + r"}{*}{\rotatebox{90}{Non-reasoning}}"
        )
        for i, row in enumerate(rows):
            if row == self.ROW_SEP:
                f.append(r"\midrule")
                f.append(
                    r"\multirow{"
                    + str(len(self.reasoning_llms))
                    + r"}{*}{\rotatebox{90}{Reasoning}}"
                )
                continue
            f.append(r"&")
            f.append(latex.Macro(f"THead-{exp_id}-{row}").use())
            macros.append(
                latex.Macro(f"THead-{exp_id}-{row}", self.llms2macros.get(row, row))
            )
            for ci, col in enumerate(cols):
                if col in metrics_2_best_models:
                    best_models = metrics_2_best_models[col]
                    if row in best_models:
                        f.append(
                            r" & \textbf{"
                            + latex.Macro(f"res-{row}-{col}").use()
                            + r"}"
                        )
                    else:
                        f.append(r" & " + latex.Macro(f"res-{row}-{col}").use())
            f.append(r"\\")

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        # f.append(r"\end{small}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()


    def make_two_results_table_same_task(
        self,
        task: str,
        exp_names: List[str],
        metrics: str,
        headers: list = None,
        dataset_name: str = "human_written",
        model_list: list = None,
        exp_metric_columns: dict = None,
        table_suffix: str = "",
        row_footnote_marks: dict[str, str] | None = None,
        table_notes: list[tuple[str, str]] | None = None,
    ):
        if model_list is None:
            rows = self.non_reasoning_llms + [self.ROW_SEP] + self.reasoning_llms
        else:
            rows = model_list
        exps = []
        for exp_name in exp_names:
            exp_id = f"{task}-{exp_name}" if dataset_name == "human_written" else f"{task}-{exp_name}-{dataset_name.replace('_','-')}"
            exps.extend(self.exps[exp_id])

        metrics_2_best_models = {}
        for exp in exps:
            best_model_file = f"metrics-best-model-{exp}.json"
            best_models = io.load(self.results_dir / best_model_file)
            if model_list is not None:
                best_models = {
                    metric: [m for m in models if m in model_list]
                    for metric, models in best_models.items()
                }
            if exp_metric_columns is not None:
                for col_metric in exp_metric_columns.get(exp, []):
                    col_key = f"{exp}-{col_metric}"
                    if col_metric in best_models:
                        metrics_2_best_models[col_key] = best_models[col_metric]
            else:
                metrics_2_best_models.update(
                    {f"{exp}-{metric}": models for metric, models in best_models.items()}
                )
        #
        exp_id = f"{task}-{'-'.join(exp_names)}" if not metrics else f"{task}-{'-'.join(exp_names)}-{metrics}"
        exp_id = exp_id if dataset_name == "human_written" else f"{exp_id}-{dataset_name.replace('_','-')}"
        if table_suffix:
            exp_id = f"{exp_id}-{table_suffix}"
        f = latex.File(self.tables_dir / f"table-results-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []

        cols = ["models"]
        task_prefix = ""
        first_half_cnt, second_half_cnt = 0, 0
        for exp in exps:
            if task_prefix and exp.split("-")[-1] != task_prefix:
                first_half_cnt = len(cols) - 1
                cols.append(self.COL_SEP)
            #
            task_prefix = exp.split("-")[-1]
            if exp_metric_columns is not None:
                ems = exp_metric_columns.get(exp, self._primary_metrics(self.exp2metrics[exp]))
            else:
                ems = self.exp2metrics[exp][metrics]
            cols += [f"{exp}-{metric}" for metric in ems]
        #rof
        second_half_cnt = len(cols) - first_half_cnt - 2
        f.append(r"\begin{table*}[t]")
        # f.append(r"\begin{small}")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        macros.append(
            latex.Macro(
                f"TCap-models-{exp_id}",
                self.metrics2macros.get(
                    f"TCap-models-{exp_id}", "Results of the models."
                ),
            )
        )
        label = r"\label{tab:" + exp_id + "}"
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")
        if table_notes:
            f.append(r"\begin{threeparttable}")

        rotate_col = "" if model_list is not None else "l "
        f.append(r"\begin{tabular}{" + rotate_col + "l " + " c " * (len(cols) - 1) + "}")
        f.append(r"\toprule")
        if model_list is None:
            f.append(r"&")  # empty cell for rotate box
        first_col = cols.pop(0)
        f.append(
            r"\multirow{2}{*}{\textbf{"
            + latex.Macro(f"THead-{exp_id}-{first_col}").use()
            + r"}}"
        )
        macros.append(latex.Macro(f"THead-{exp_id}-{first_col}", "Models"))
        f.append(
            r"& \multicolumn{"
            + str(first_half_cnt)
            + "}{c}{\\textbf{\\"
            + (exp_names[0] if headers is None else headers[0])
            + "}}"
        )
        f.append(
            r"& \multicolumn{"
            + str(second_half_cnt)
            + "}{c}{\\textbf{\\"
            + (exp_names[1] if headers is None else headers[1])
            + "}}"
        )
        f.append(r"\\")
        col_offset = 3 if model_list is None else 2
        f.append(
            r"\cmidrule(lr){"
            + str(col_offset)
            + "-"
            + str(col_offset + first_half_cnt - 1)
            + "}"
        )
        f.append(
            r"\cmidrule(lr){"
            + str(col_offset + first_half_cnt)
            + "-"
            + str(col_offset + first_half_cnt + second_half_cnt - 1)
            + "}"
        )
        if model_list is None:
            f.append(r"&")  # empty cell for rotate box
        for ci, col in enumerate(cols):
            if col == self.COL_SEP:
                continue
            f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
            macros.append(
                latex.Macro(f"THead-{exp_id}-{col}", self.metrics2macros.get(col, col))
            )
        f.append(r"\\")

        f.append(r"\midrule")
        if model_list is None:
            # rotate box
            f.append(
                r"\multirow{"
                + str(len(self.non_reasoning_llms))
                + r"}{*}{\rotatebox{90}{Non-reasoning}}"
            )
        for i, row in enumerate(rows):
            if row == self.ROW_SEP:
                f.append(r"\midrule")
                if model_list is None:
                    f.append(
                        r"\multirow{"
                        + str(len(self.reasoning_llms))
                        + r"}{*}{\rotatebox{90}{Reasoning}}"
                    )
                continue
            if model_list is None:
                f.append(r"&")
            row_label = latex.Macro(f"THead-{exp_id}-{row}").use()
            if row_footnote_marks and row in row_footnote_marks:
                row_label += rf"\tnote{{{row_footnote_marks[row]}}}"
            f.append(row_label)
            macros.append(
                latex.Macro(f"THead-{exp_id}-{row}", self.llms2macros.get(row, row))
            )
            for ci, col in enumerate(cols):
                if col in metrics_2_best_models:
                    best_models = metrics_2_best_models[col]
                    if row in best_models:
                        f.append(
                            r" & \textbf{"
                            + latex.Macro(f"res-{row}-{col}").use()
                            + r"}"
                        )
                    else:
                        f.append(r" & " + latex.Macro(f"res-{row}-{col}").use())
            f.append(r"\\")

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        if table_notes:
            f.append(r"\begin{tablenotes}")
            f.append(r"\scriptsize")
            for mark, note in table_notes:
                f.append(rf"\item {mark} {note}")
            f.append(r"\end{tablenotes}")
            f.append(r"\end{threeparttable}")
        f.append(r"\end{center}")
        # f.append(r"\end{small}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()


    def make_two_results_table_same_task_op(self, exp_names: List[str], metrics: str, headers: list = None, dataset_names: list[str] = ["human_written","synthetic_cpp","fuzzer_generated"]):
        task: str = "op"
        exp_id = f"{task}-{'-'.join(exp_names)}-{'-'.join([dataset_name.replace('_','-') for dataset_name in dataset_names])}-{metrics}"
        f = latex.File(self.tables_dir / f"table-results-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []
        f.append(r"\begin{table*}[t]")
        # f.append(r"\begin{small}")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        label = r"\label{tab:" + exp_id + "}"
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")
        for idx, dataset_name in enumerate(dataset_names):
            exps = []
            for exp_name in exp_names:
                exp_id = f"{task}-{exp_name}" if dataset_name == "human_written" else f"{task}-{exp_name}-{dataset_name.replace('_','-')}"
                if 'IMP-K' == exp_name:
                    mod_exps = [exp for exp in self.exps[exp_id] if '-nk-' not in exp]
                else:
                    mod_exps = self.exps[exp_id]
                #fi
                exps.extend(mod_exps)
            #rof
            model_list = self.exp2models[f"{task}-{'-'.join(exp_names)}"] if dataset_name == "human_written" else self.exp2models[f"{task}-{'-'.join(exp_names)}-{dataset_name.replace('_','-')}"]
            metrics_2_best_models = {}
            for exp in exps:
                best_model_file = f"metrics-best-model-{exp}.json"
                best_models = io.load(self.results_dir / best_model_file)
                metrics_2_best_models.update(
                    {f"{exp}-{metric}": models for metric, models in best_models.items()}
                )
            #
            exp_id = f"{task}-{'-'.join(exp_names)}" if not metrics else f"{task}-{'-'.join(exp_names)}-{metrics}"
            exp_id = exp_id if dataset_name == "human_written" else f"{exp_id}-{dataset_name.replace('_','-')}"
            cols = ["models"]
            task_prefix = ""
            first_half_cnt, second_half_cnt = 0, 0
            for exp in exps:
                if task_prefix and exp.split("-")[-1] != task_prefix:
                    first_half_cnt = len(cols) - 1
                    cols.append(self.COL_SEP)
                #
                task_prefix = exp.split("-")[-1]
                ems = self.exp2metrics[exp][metrics]
                cols += [f"{exp}-{metric}" for metric in ems]
            #rof
            
            second_half_cnt = len(cols) - first_half_cnt - 3
            # Hack to add nk in its own column
            nk_index: int = -1
            for i, col in enumerate(cols):
                if '-nk-' in col:
                    nk_index = i
                #fi
            #rof
            if nk_index >= 0:
                nk_value = cols.pop(nk_index)
                cols.insert(1, nk_value)
            #fi 
            rows = model_list
            first_col = cols.pop(0)
            if idx == 0:
                f.append(r"\begin{tabular}{l l " + " c " * (len(cols) - 1) + "}")
                f.append(r"\toprule")
                f.append(r"&")  # empty cell for rotate box
                f.append(
                    r"\multirow{2}{*}{\textbf{"
                    + latex.Macro(f"THead-{exp_id}-{first_col}").use()
                    + r"}}"
                )
                macros.append(latex.Macro(f"THead-{exp_id}-{first_col}", "Models"))
                f.append(
                    r"& \multirow{2}{*}{\textbf{"
                    + latex.Macro(f"THead-{exp_id}-{cols[0]}").use()
                    + r"}}"
                )
                f.append(
                    r"& \multicolumn{"
                    + str(first_half_cnt)
                    + "}{c}{\\textbf{\\"
                    + (exp_names[0] if headers is None else headers[0])
                    + "}}"
                )
                f.append(
                    r"& \multicolumn{"
                    + str(second_half_cnt)
                    + "}{c}{\\textbf{\\"
                    + (exp_names[1] if headers is None else headers[1])
                    + "}}"
                )
                f.append(r"\\")
                f.append(r"\cmidrule(lr){4-" + str(first_half_cnt + 3) + "}")
                f.append(
                    r"\cmidrule(lr){" + str(first_half_cnt + 4) + "-" + str(len(cols) + 1) + "}"
                )
                f.append(r"&")  # empty cell for rotate box
                for ci, col in enumerate(cols):
                    if col == self.COL_SEP:
                        continue
                    macros.append(
                        latex.Macro(f"THead-{exp_id}-{col}", self.metrics2macros.get(col, col)))
                    if ci == 0:
                        f.append(r" & ")
                        continue
                    f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
                #rof
                f.append(r"\\")
            #fi

            f.append(r"\midrule")
            f.append(r"\multicolumn{"+ str(9) +r"}{c}{\textbf{" + latex.Macro(f"THead-{exp_id}-dataset").use() + "}}%")
            f.append(r"\\")
            f.append(r"\midrule")
            macros.append(
                latex.Macro(f"THead-{exp_id}-dataset", self.datasets2macros.get(dataset_name, dataset_name))
            )
            # rotate box
            if dataset_name == "human_written":
                f.append(
                    r"\multirow{"
                    + str(len(self.non_reasoning_llms))
                    + r"}{*}{\rotatebox{90}{Non-reasoning}}"
                )
            #fi 
            for i, row in enumerate(rows):
                if row == self.ROW_SEP:
                    f.append(r"\midrule")
                    if dataset_name == "human_written":
                        f.append(
                            r"\multirow{"
                            + str(len(self.reasoning_llms))
                            + r"}{*}{\rotatebox{90}{Reasoning}}"
                        )
                    #fi
                    continue
                #fi
                f.append(r"&")
                f.append(latex.Macro(f"THead-{exp_id}-{row}").use())
                macros.append(
                    latex.Macro(f"THead-{exp_id}-{row}", self.llms2macros.get(row, row))
                )
                for ci, col in enumerate(cols):
                    if col in metrics_2_best_models:
                        best_models = metrics_2_best_models[col]
                        if row in best_models:
                            f.append(
                                r" & \textbf{"
                                + latex.Macro(f"res-{row}-{col}").use()
                                + r"}"
                            )
                        else:
                            f.append(r" & " + latex.Macro(f"res-{row}-{col}").use())
                        #fi
                    #fi
                #rof
                f.append(r"\\")
            #rof
        #rof
        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()


    def make_two_results_table_same_task_reasoning_only(self, task: str, exp_names: List[str], metrics: str, headers: list = None, dataset_name: str = "synthetic_cpp"):
        model_list = self.exp2models[f"{task}-{'-'.join(exp_names)}-{dataset_name.replace('_','-')}"]
        exps = []
        for exp_name in exp_names:
            exp_id = f"{task}-{exp_name}-{dataset_name.replace('_','-')}"
            exps.extend(self.exps[exp_id])

        metrics_2_best_models = {}
        for exp in exps:
            best_model_file = f"metrics-best-model-{exp}.json"
            best_models = io.load(self.results_dir / best_model_file)
            metrics_2_best_models.update(
                {f"{exp}-{metric}": models for metric, models in best_models.items()}
            )
        #
        exp_id = f"{task}-{'-'.join(exp_names)}" if not metrics else f"{task}-{'-'.join(exp_names)}-{metrics}"
        exp_id = f"{exp_id}-{dataset_name.replace('_','-')}"
        f = latex.File(self.tables_dir / f"table-results-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []

        cols = ["models"]
        task_prefix = ""
        first_half_cnt, second_half_cnt = 0, 0
        for exp in exps:
            if task_prefix and exp.split("-")[-3] != task_prefix:
                first_half_cnt = len(cols) - 1
                cols.append(self.COL_SEP)
            #
            task_prefix = exp.split("-")[-3]
            ems = self.exp2metrics[exp][metrics]
            cols += [f"{exp}-{metric}" for metric in ems]
        #rof
        second_half_cnt = len(cols) - first_half_cnt - 3
        # Hack to add nk in its own column
        nk_index: int = -1
        for i, col in enumerate(cols):
            if '-nk-' in col:
                nk_index = i
            #fi
        #rof
        if nk_index >= 0:
            nk_value = cols.pop(nk_index)
            cols.insert(1, nk_value)
        #fi 
        rows = model_list
        f.append(r"\begin{table*}[t]")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        label = r"\label{tab:" + exp_id + "}"
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")

        f.append(r"\begin{tabular}{l " + " c " * (len(cols) - 1) + "}")
        f.append(r"\toprule")
        first_col = cols.pop(0)
        f.append(
            r"\multirow{2}{*}{\textbf{"
            + latex.Macro(f"THead-{exp_id}-{first_col}").use()
            + r"}}"
        )
        macros.append(latex.Macro(f"THead-{exp_id}-{first_col}", "Models"))
        f.append(
            r"& \multirow{2}{*}{\textbf{"
            + latex.Macro(f"THead-{exp_id}-{cols[0]}").use()
            + r"}}"
        )
        f.append(
            r"& \multicolumn{"
            + str(first_half_cnt)
            + "}{c}{\\textbf{\\"
            + (exp_names[0] if headers is None else headers[0])
            + "}}"
        )
        f.append(
            r"& \multicolumn{"
            + str(second_half_cnt)
            + "}{c}{\\textbf{\\"
            + (exp_names[1] if headers is None else headers[1])
            + "}}"
        )
        f.append(r"\\")
        f.append(r"\cmidrule(lr){3-" + str(first_half_cnt + 2) + "}")
        f.append(
            r"\cmidrule(lr){" + str(first_half_cnt + 3) + "-" + str(len(cols)) + "}"
        )
        for ci, col in enumerate(cols):
            if col == self.COL_SEP:
                continue
            macros.append(
                latex.Macro(f"THead-{exp_id}-{col}", self.metrics2macros.get(col, col)))
            if ci == 0:
                f.append(r" & ")
                continue
            f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
        f.append(r"\\")
        f.append(r"\midrule")
        for i, row in enumerate(rows):
            f.append(latex.Macro(f"THead-{exp_id}-{row}").use())
            macros.append(
                latex.Macro(f"THead-{exp_id}-{row}", self.llms2macros.get(row, row))
            )
            for ci, col in enumerate(cols):
                if col in metrics_2_best_models:
                    best_models = metrics_2_best_models[col]
                    if row in best_models:
                        f.append(
                            r" & \textbf{"
                            + latex.Macro(f"res-{row}-{col}").use()
                            + r"}"
                        )
                    else:
                        f.append(r" & " + latex.Macro(f"res-{row}-{col}").use())
            f.append(r"\\")

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        # f.append(r"\end{small}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()


    def make_etp_percentage_match_table(self, mutation: str, exp_name: str):
        model_list = self.non_reasoning_llms + [self.ROW_SEP] + self.reasoning_llms
        if mutation == 'uk':
            exp_id = f"etp-{mutation}-{exp_name}-percentage-trace-match"
        else:
            exp_id = f"etp-mk-{exp_name}-{mutation}-percentage-trace-match"
        #fi            

        f = latex.File(self.tables_dir / f"table-results-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []
        cols = ["models"]
        macros.append(latex.Macro(f"THead-{exp_id}-models", "Models"))
        for i in range(1, 11):
            macros.append(
                latex.Macro(f"THead-{exp_id}-{i * 10}", f"\\textbf{{<{(i * 10)}\\%}}")
            )
            cols.append(str(i * 10))
        #rof
        task_prefix = ""
        rows = model_list
        f.append(r"\begin{table*}[t]")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        label = r"\label{tab:" + exp_id + "}"
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")
        f.append(r"\begin{tabular}{l l " + " c " * (10) + "}")
        f.append(r"\toprule")
        for col in cols:
            f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
        f.append(r"\\")
        f.append(r"\midrule")
        # rotate box
        f.append(
            r"\multirow{"
            + str(len(self.non_reasoning_llms))
            + r"}{*}{\rotatebox{90}{Non-reasoning}}"
        )
        cols.pop(0)
        for i, row in enumerate(rows):
            if row == self.ROW_SEP:
                f.append(r"\midrule")
                f.append(
                    r"\multirow{"
                    + str(len(self.reasoning_llms))
                    + r"}{*}{\rotatebox{90}{Reasoning}}"
                )
                continue
            f.append(r"&")
            f.append(latex.Macro(f"THead-{exp_id}-{row}").use())
            macros.append(
                latex.Macro(f"THead-{exp_id}-{row}", self.llms2macros.get(row, row))
            )
            for ci, col in enumerate(cols):
                f.append(r" & " + latex.Macro(f"res-{row}-{exp_id}-{col}").use())
            f.append(r"\\")

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        # f.append(r"\end{small}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()
    #fed
    

    def make_ig_result_table(self, task: str, exp_name: str):
        exp_id = f"{task}-{exp_name}"
        model_list = self.exp2models[exp_id]

        # Get the number of impl languages from metrics file
        metrics_file = f"metrics-{exp_id}.json"
        metrics = io.load(self.results_dir / metrics_file)
        for model, res in metrics.items():
            impl_langs = res.keys()
            n_samples = len(res[next(iter(impl_langs))])
            break

        f = latex.File(self.tables_dir / f"table-results-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []

        cols = ["pl", "models"]
        ems = self.exp2metrics[exp_id]
        cols += [f"{exp_id}-{metric}" for metric in ems]
        valid_cnt = sum(1 for metric in ems if metric.startswith("valid_"))
        invalid_cnt = sum(1 for metric in ems if metric.startswith("invalid_"))
        rows = model_list
        # f.append(r"\begin{table*}[t]")
        f.append(r"\begin{table*}[!h]")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + "}")

        f.append(r"\begin{tabular}{l " + " c " * (len(cols) - 1) + "}")
        f.append(r"\toprule")
        f.append(r"\multirow{2}{*}{\textbf{" + "PL" + "}}")
        f.append(r"& \multirow{2}{*}{\textbf{" + "Models" + "}}")
        f.append(r"& \multirow{2}{*}{\textbf{" + "LOC" + "}}")
        f.append(r"& \multirow{2}{*}{\textbf{" + "Comp" + "}}")
        f.append(
            r"& \multicolumn{" + str(valid_cnt) + "}{c}{\\textbf{" + "Valid" + "}}"
        )
        f.append(
            r"& \multicolumn{" + str(invalid_cnt) + "}{c}{\\textbf{" + "Invalid" + "}}"
        )
        f.append(r"\\")
        f.append(r"\cmidrule(lr){5-" + str(valid_cnt + 4) + "}")
        f.append(r"\cmidrule(lr){" + str(valid_cnt + 5) + "-" + str(len(cols)) + "}")

        f.append("&\n&\n&\n")
        for _, col in enumerate(cols[4:]):
            f.append(r"& \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
            macros.append(
                latex.Macro(f"THead-{exp_id}-{col}", self.metrics2macros.get(col, col))
            )
        f.append(r"\\")

        f.append(r"\midrule")
        for i, impl_lang in enumerate(impl_langs):
            f.append(
                r"\multirow{"
                + str(len(rows) * n_samples)
                + r"}{*}{\rotatebox{90}{"
                + f"{impl_lang}"
                + r"}}"
            )

            for j, row in enumerate(rows):
                f.append(
                    r"& \multirow{"
                    + str(n_samples)
                    + "}{*}{"
                    + latex.Macro(f"THead-{exp_id}-{row}").use()
                    + r"}"
                )
                if i == 0:
                    macros.append(
                        latex.Macro(
                            f"THead-{exp_id}-{row}", self.llms2macros.get(row, row)
                        )
                    )
                for k in range(n_samples):
                    if k > 0:
                        f.append("&")
                    for _, col in enumerate(cols[2:]):
                        # insert impl_lang after the last '-' in col
                        col_parts = col.split("-")
                        col_parts.insert(-1, impl_lang)
                        col_parts.append(str(k))
                        col = "-".join(col_parts)
                        f.append(r"& " + latex.Macro(f"res-{row}-{col}").use())
                    f.append(r"\\")
                if j != len(rows) - 1:
                    f.append(r"\cmidrule(lr){2-" + str(len(cols)) + "}")
            if i != len(impl_langs) - 1:
                f.append(r"\midrule")

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()

    def infer_fmt(self, value: Any, float_precision: int = 1):
        if isinstance(value, int):
            return ",d"
        else:
            return f",.{float_precision}f"


    def make_op_odds_per_iqr_table(self, models: list, semantics_types: list, strategies: list, op_traces: list, dataset_file: str, metrics: list[str], headers: list[str], header_cat: dict, filter_while_count: int = -1, filter_if_count: int = -1, cats_map: dict = {'K': '\KTool', 'SOS': '\Sos'}):
        [_, regression_details] = gen_op_regression_coefficients(models, semantics_types, strategies, op_traces, dataset_file, metrics, filter_while_count, filter_if_count)
        model_data: dict = defaultdict(dict)
        model_list: list = self.exp2models['op-IMP-odds-per-iqr-SOS-K']
        keys = sorted(header_cat.keys())
        metrics = [elem for key in keys for elem in header_cat[key]]
        for model, model_regression_detail in regression_details.items():
            coef_table = model_regression_detail['coef_table']
            or_per_iqr: list = []
            for metric in metrics:
                or_per_iqr.append(round(100 * (coef_table['OR_per_IQR'][metric] - 1)))
            #rof
            model_splits: list = model.split('-')
            model_name: str = "-".join(model_splits[0: len(model_splits) - 1])
            semantics_type: str = model_splits[-1]
            model_data[semantics_type][model_name] = or_per_iqr
        #rof

        exp_id: str = "op-odds-per-iqr"
        f = latex.File(self.tables_dir / f"table-results-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []
        cols = ["models"] + metrics
        macros.append(
            latex.Macro(f"THead-{exp_id}-models", "\\multirow{2}{*}{\\textbf{Models}}")
        )
        for metric, header in zip(metrics, headers):
            macros.append(
                latex.Macro(f"THead-{exp_id}-{metric}", f"\\textbf{{{header}}}")
            )
        #rof
        for key in header_cat.keys():
            macros.append(
                latex.Macro(f"THead-{exp_id}-{key}", f"\\textbf{{{key}}}")
            )
        #rof
        task_prefix = ""
        rows = model_list
        f.append(r"\begin{table*}[t]")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        label = r"\label{tab:" + exp_id + "}"
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")
        f.append(r"\begin{tabular}{l l" + " c " * ((len(cols) - 1)) + "}")
        f.append(r"\toprule")
        f.append(r" & " + latex.Macro(f"THead-{exp_id}-{cols[0]}").use())
        cols.pop(0)
        for key in keys:
            f.append(r" & \multicolumn{" + str(len(header_cat[key])) +"}{c}{"  + latex.Macro(f"THead-{exp_id}-{key}").use() + "}")
        #rof
        f.append(r"\\")
        start_rule: int = 3
        for key in keys:
            f.append(r" \cmidrule(lr){" + str(start_rule) + "-" + str(len(header_cat[key]) - 1 + start_rule) + "}")
            start_rule += len(header_cat[key])
        #rof
        f.append(r"& ")
        for col in cols:
            f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
        f.append(r"\\")
        cat_sorted: list = sorted(cats_map.keys())
        for cat in cat_sorted:
            f.append(r"\midrule")
            # f.append(
            #     r"\multicolumn{" + str(len(cols) + 1) + "}{c}{\\textbf{" + cats_map[cat] + "}}\\\\"
            # )
            # f.append(r"\midrule")
            # rotate box
            f.append(
                r"\multirow{"
                + str(len(self.non_reasoning_llms) + 2)
                + r"}{*}{\rotatebox{90}{" + cats_map[cat] + "}}"
            )
            for i, row in enumerate(model_list):
                if row == self.ROW_SEP:
                    f.append(r"\cmidrule(lr){2-" + str(len(cols) + 2) + "}")
                    # f.append(
                    #     r"\multirow{"
                    #     + str(2)
                    #     + r"}{*}{\rotatebox{90}{Reasoning}}"
                    # )
                    continue
                f.append(r"&")
                f.append(latex.Macro(f"THead-{exp_id}-{row}-{cat}").use())
                macros.append(
                    latex.Macro(f"THead-{exp_id}-{row}-{cat}", self.llms2macros.get(row, row))
                )
                max_idx: int = max(enumerate(model_data[cat][row]), key=lambda x: abs(x[1]))[0]
                for idx, col in enumerate(cols):
                    macros.append(latex.Macro(f"res-{row}-{exp_id}-{col}-{cat}", model_data[cat][row][idx] if idx != max_idx else f"\\textbf{{{model_data[cat][row][idx]}}}"))
                    f.append(r" & " + latex.Macro(f"res-{row}-{exp_id}-{col}-{cat}").use())
                f.append(r"\\")
            #rof
        #rof

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        # f.append(r"\end{small}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()
    #fed


    def make_op_odds_per_iqr_table_combined(self, models_list: list, semantics_types_list: list, strategies_list: list, op_traces_list: list, dataset_files: list, dataset_names: list[str], mutation: str, metrics: list[str], headers: list[str], header_cat: dict, filter_while_count: int = -1, filter_if_count: int = -1, cats_map: dict = {'K': 'K', 'SOS': '\Sos'}):
        f = latex.File(self.tables_dir / f"table-results-odds-per-iqr-{mutation}-{'-'.join([dataset_name.replace('_','-') for dataset_name in dataset_names])}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-odds-per-iqr-{mutation}-{'-'.join([dataset_name.replace('_','-') for dataset_name in dataset_names])}.tex")
        macros = []
        for idx, (models, semantics_types, strategies, op_traces, dataset_file, dataset_name) in enumerate(zip(models_list, semantics_types_list, strategies_list, op_traces_list, dataset_files, dataset_names)):
            [_, regression_details] = gen_op_regression_coefficients(models, semantics_types, strategies, op_traces, dataset_file, metrics, [mutation] * len(models),  filter_while_count, filter_if_count)
            model_data: dict = defaultdict(dict)
            model_list: list = self.exp2models[f'op-IMP-odds-per-iqr-SOS-K-{mutation}-{dataset_name}']
            keys = sorted(header_cat.keys())
            metrics = [elem for key in keys for elem in header_cat[key]]
            for model, model_regression_detail in regression_details.items():
                coef_table = model_regression_detail['coef_table']
                or_per_iqr: list = []
                for metric in metrics:
                    or_per_iqr.append(round(100 * (coef_table['OR_per_IQR'][metric] - 1)))
                #rof
                model_splits: list = model.split('-')
                model_name: str = "-".join(model_splits[0: len(model_splits) - 2])
                semantics_type: str = model_splits[-2]
                model_data[semantics_type][model_name] = or_per_iqr
            #rof
            exp_id: str = f"op-odds-per-iqr-{mutation}-{dataset_name.replace('_','-')}"
            cols = metrics
            if idx == 0:
                cols = ["models"] + metrics
                macros.append(
                    latex.Macro(f"THead-{exp_id}-models", "\\multirow{2}{*}{\\textbf{Models}}")
                )
                for metric, header in zip(metrics, headers):
                    macros.append(
                        latex.Macro(f"THead-{exp_id}-{metric}", f"\\textbf{{{header}}}")
                    )
                #rof
                for key in header_cat.keys():
                    macros.append(
                        latex.Macro(f"THead-{exp_id}-{key}", f"\\textbf{{{key}}}")
                    )
                #rof

                f.append(r"\begin{table*}[t]")
                f.append(r"\begin{center}")
                caption = latex.Macro(f"TCap-models-{exp_id}").use()
                f.append(r"\TableFont")
                f.append(r"\caption{" + caption + "}")
                if mutation == 'nk':
                    f.append(r"\begin{tabular}{l" + " c " * ((len(cols) - 1)) + "}")
                else:
                    f.append(r"\begin{tabular}{l l" + " c " * ((len(cols) - 1)) + "}")
                #fi
                f.append(r"\toprule")
                if mutation != 'nk':
                    f.append(" & ")
                #fi
                f.append(latex.Macro(f"THead-{exp_id}-{cols[0]}").use())
                cols.pop(0)
                for key in keys:
                    f.append(r" & \multicolumn{" + str(len(header_cat[key])) +"}{c}{"  + latex.Macro(f"THead-{exp_id}-{key}").use() + "}")
                #rof
                f.append(r"\\")
                start_rule: int = 3 if mutation != 'nk' else 2
                for key in keys:
                    f.append(r" \cmidrule(lr){" + str(start_rule) + "-" + str(len(header_cat[key]) - 1 + start_rule) + "}")
                    start_rule += len(header_cat[key])
                #rof
                if mutation != 'nk':
                    f.append(r"& ")
                #fi
                for col in cols:
                    f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
                f.append(r"\\")
            #fi
            rows = model_list
            cat_sorted: list = sorted(cats_map.keys())
            f.append(r"\midrule")
            f.append(r"\multicolumn{"+ str(11 if mutation != 'nk' else 10) +r"}{c}{\textbf{" + latex.Macro(f"THead-{exp_id}-dataset").use() + "}}%")
            f.append(r"\\")
            macros.append(
                latex.Macro(f"THead-{exp_id}-dataset", self.datasets2macros.get(dataset_name, dataset_name))
            )
            for j, cat in enumerate(cat_sorted):
                if len(model_list) > 2 or j == 0:
                    f.append(r"\midrule")
                #fi
                # rotate box
                #if dataset_name == "human_written":
                if mutation != 'nk':
                    if len(model_list) < 2 and j > 0:                       
                        f.append(
                            r"\multirow{"
                            + str(2)
                            + r"}{*}{\rotatebox{90}{" + cats_map[cat] + "}}"
                        )
                    else:
                        f.append(
                            r"\multirow{"
                            + str(len(model_list))
                            + r"}{*}{\rotatebox{90}{" + cats_map[cat] + "}}"
                        )
                    #fi
                #fi
                for i, row in enumerate(model_list):
                    if row == self.ROW_SEP:
                        f.append(r"\cmidrule(lr){2-" + str(len(cols) + 2) + "}")
                        continue
                    if mutation != 'nk':
                        f.append(r"&")
                    #fi
                    f.append(latex.Macro(f"THead-{exp_id}-{row}-{cat}").use())
                    macros.append(
                        latex.Macro(f"THead-{exp_id}-{row}-{cat}", self.llms2macros.get(row, row))
                    )
                    max_idx: int = max(enumerate(model_data[cat][row]), key=lambda x: abs(x[1]))[0]
                    for idx2, col in enumerate(cols):
                        macros.append(latex.Macro(f"res-{row}-{exp_id}-{col}-{cat}", model_data[cat][row][idx2] if idx2 != max_idx else f"\\textbf{{{model_data[cat][row][idx2]}}}"))
                        f.append(r" & " + latex.Macro(f"res-{row}-{exp_id}-{col}-{cat}").use())
                    if len(model_list) < 2 and mutation != 'nk':
                        f.append(r"\\[0.15cm]")
                    else:
                        f.append(r"\\")
                    #fi
                #rof
                #if len(model_list) < 2:
                #    f.append(r"\\")
                #fi
            #rof
        #rof
        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()
    #fed


    def make_op_nk_odds_per_iqr_table(self, models: list, semantics_types: list, strategies: list, op_traces: list, dataset_file: str, metrics: list[str], headers: list[str], header_cat: dict, filter_while_count: int = -1, filter_if_count: int = -1):
        [_, regression_details] = gen_op_regression_coefficients(models, semantics_types, strategies, op_traces, dataset_file, metrics, filter_while_count, filter_if_count)
        model_data: dict = defaultdict(dict)
        model_list: list = self.exp2models['op-nk-IMP-odds-per-iqr']
        keys = sorted(header_cat.keys())
        metrics = [elem for key in keys for elem in header_cat[key]]
        for model, model_regression_detail in regression_details.items():
            coef_table = model_regression_detail['coef_table']
            or_per_iqr: list = []
            for metric in metrics:
                or_per_iqr.append(round(100 * (coef_table['OR_per_IQR'][metric] - 1)))
            #rof
            model_splits: list = model.split('-')
            model_name: str = "-".join(model_splits[0: len(model_splits) - 1])
            semantics_type: str = model_splits[-1]
            model_data[semantics_type][model_name] = or_per_iqr
        #rof

        exp_id: str = "op-nk-odds-per-iqr"
        f = latex.File(self.tables_dir / f"table-results-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []
        cols = ["models"] + metrics
        macros.append(
            latex.Macro(f"THead-{exp_id}-models", "\\multirow{2}{*}{\\textbf{Models}}")
        )
        for metric, header in zip(metrics, headers):
            macros.append(
                latex.Macro(f"THead-{exp_id}-{metric}", f"\\textbf{{{header}}}")
            )
        #rof
        for key in header_cat.keys():
            macros.append(
                latex.Macro(f"THead-{exp_id}-{key}", f"\\textbf{{{key}}}")
            )
        #rof
        task_prefix = ""
        rows = model_list
        f.append(r"\begin{table*}[t]")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        label = r"\label{tab:" + exp_id + "}"
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")
        f.append(r"\begin{tabular}{l l" + " c " * ((len(cols) - 1)) + "}")
        f.append(r"\toprule")
        f.append(r" & " + latex.Macro(f"THead-{exp_id}-{cols[0]}").use())
        cols.pop(0)
        for key in keys:
            f.append(r" & \multicolumn{" + str(len(header_cat[key])) +"}{c}{"  + latex.Macro(f"THead-{exp_id}-{key}").use() + "}")
        #rof
        f.append(r"\\")
        start_rule: int = 3
        for key in keys:
            f.append(r" \cmidrule(lr){" + str(start_rule) + "-" + str(len(header_cat[key]) - 1 + start_rule) + "}")
            start_rule += len(header_cat[key])
        #rof
        f.append(r"& ")
        for col in cols:
            f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
        f.append(r"\\")
        f.append(r"\midrule")
        # rotate box
        f.append(
            r"\multirow{"
            + str(len(self.non_reasoning_llms))
            + r"}{*}{\rotatebox{90}{Non-reasoning}}"
        )
        for i, row in enumerate(model_list):
            if row == self.ROW_SEP:
                f.append(r"\midrule")
                f.append(
                    r"\multirow{"
                    + str(3)
                    + r"}{*}{\rotatebox{90}{Reasoning}}"
                )
                continue
            f.append(r"&")
            f.append(latex.Macro(f"THead-{exp_id}-{row}-SOS").use())
            macros.append(
                latex.Macro(f"THead-{exp_id}-{row}-SOS", self.llms2macros.get(row, row))
            )
            max_idx: int = max(enumerate(model_data['SOS'][row]), key=lambda x: abs(x[1]))[0]
            for idx, col in enumerate(cols):
                macros.append(latex.Macro(f"res-{row}-{exp_id}-{col}-SOS", model_data['SOS'][row][idx] if idx != max_idx else f"\\textbf{{{model_data['SOS'][row][idx]}}}"))
                f.append(r" & " + latex.Macro(f"res-{row}-{exp_id}-{col}-SOS").use())
            f.append(r"\\")
        #rof

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        # f.append(r"\end{small}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()
    #fed
    

    def make_imp_k_rules_table(self):
        """Generate the appendix table of numbered K-framework semantics rules."""
        generate_imp_k_rules_table_tex(self.tables_dir / "imp_k_rules_table.tex")

    # fed

    def make_dataset_stats_table(self, dataset_files: list[str], dataset_names: list[str], headers: list[str], header_cat: dict):
        X: list = [dataset_code_complexity_metrics_to_df(read_from_json_file(dataset_file)) for dataset_file in dataset_files]
        keys = sorted(header_cat.keys())
        metrics = [elem for key in keys for elem in header_cat[key]]
        datasets_metrics: dict = defaultdict(dict)
        
        for x, name in zip(X, dataset_names):
            dataset_metric: list = [len(x)]
            dataset_first_quantile: list = [len(x)]
            dataset_third_quantile: list = [len(x)]
            for metric in metrics:
                median_val: int = round(x[metric].median())
                first_quantile: int = round(x[metric].quantile(0.25))
                third_quantile: int = round(x[metric].quantile(0.75))
                median_val = f"{round(median_val // 1000)}K" if median_val >= 1000 else median_val
                first_quantile = f"{round(first_quantile // 1000)}K" if first_quantile >= 1000 else first_quantile
                third_quantile = f"{round(third_quantile // 1000)}K" if third_quantile >= 1000 else third_quantile
                dataset_metric.append(median_val)
                dataset_first_quantile.append(first_quantile)
                dataset_third_quantile.append(third_quantile)
            #rof
            datasets_metrics[name]['median'] = dataset_metric
            datasets_metrics[name]['first-quantile'] = dataset_first_quantile
            datasets_metrics[name]['third-quantile'] = dataset_third_quantile
        #rof

        metrics.insert(0, "NumPoints")
        headers.insert(0, "\#Points")
        
        exp_id: str = "datasets-stats"
        f = latex.File(self.tables_dir / f"table-{exp_id}.tex")
        mf = latex.File(self.tables_dir / f"macros-table-{exp_id}.tex")
        macros = []
        cols = ["datasets"] + metrics
        macros.append(
            latex.Macro(f"THead-{exp_id}-{cols[0]}", "\\multirow{2}{*}{\\textbf{Dataset}}")
        )
        macros.append(
            latex.Macro(f"THead-{exp_id}-{cols[1]}", "\\multirow{2}{*}{\\textbf{\#Prog}}")
        )
        for metric, header in zip(metrics, headers):
            if metric == "NumPoints":
                continue
            #fi
            macros.append(
                latex.Macro(f"THead-{exp_id}-{metric}", f"\\textbf{{{header}}}")
            )
        #rof
        for key in header_cat.keys():
            macros.append(
                latex.Macro(f"THead-{exp_id}-{key}", f"\\textbf{{{key}}}")
            )
        #rof
        f.append(r"\begin{table*}[t]")
        f.append(r"\begin{center}")
        caption = latex.Macro(f"TCap-models-{exp_id}").use()
        label = r"\label{tab:" + exp_id + "}"
        f.append(r"\TableFont")
        f.append(r"\caption{" + caption + label + "}")
        f.append(r"\begin{tabular}{l " + " c " * ((len(cols) - 1)) + "}")
        f.append(r"\toprule")
        f.append(latex.Macro(f"THead-{exp_id}-{cols[0]}").use())
        f.append(r" & " + latex.Macro(f"THead-{exp_id}-{cols[1]}").use())
        cols.pop(0)
        cols.pop(0)
        for key in keys:
            f.append(r" & \multicolumn{" + str(len(header_cat[key])) +"}{c}{"  + latex.Macro(f"THead-{exp_id}-{key}").use() + "}")
        #rof
        f.append(r"\\")
        start_rule: int = 3
        for key in keys:
            f.append(r" \cmidrule(lr){" + str(start_rule) + "-" + str(len(header_cat[key]) - 1 + start_rule) + "}")
            start_rule += len(header_cat[key])
        #rof
        f.append(" & ")
        for col in cols:
            f.append(r" & \textbf{" + latex.Macro(f"THead-{exp_id}-{col}").use() + "}")
        f.append(r"\\")        
        f.append(r"\midrule")
        for name in dataset_names:
            f.append(latex.Macro(f"THead-{exp_id}-{name}").use())
            macros.append(
                latex.Macro(f"THead-{exp_id}-{name}", self.datasets2macros[name])
            )
            for met in ['median']:
                for i, metric in enumerate(metrics):
                    if met == "median":
                        macros.append(latex.Macro(f"res-{name}-{exp_id}-{metric}-{met}", datasets_metrics[name][met][i]))
                        f.append(r" & " + latex.Macro(f"res-{name}-{exp_id}-{metric}-{met}").use())
                        # if i == 0:
                        #     f.append(r" & \multirow{2}{*}{" + latex.Macro(f"res-{name}-{exp_id}-{metric}-{met}").use() + "}")
                        # else:
                    #fi
                    # else:
                    #     if i == 0:
                    #         f.append(" & ")
                    #         continue
                    #     #fi
                    #     macros.append(latex.Macro(f"res-{name}-{exp_id}-{metric}-first-quantile", datasets_metrics[name]['first-quantile'][i]))
                    #     macros.append(latex.Macro(f"res-{name}-{exp_id}-{metric}-third-quantile", datasets_metrics[name]['third-quantile'][i]))
                        #f.append(r" & \tiny{\Delta " + latex.Macro(f"res-{name}-{exp_id}-{metric}-first-quantile").use() + " - " + latex.Macro(f"res-{name}-{exp_id}-{metric}-third-quantile").use() + "}")     
                #rof
                f.append(r"\\")
            #rof
        #rof

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        f.append(r"\end{table*}")
        f.save()

        for def_macro in macros:
            mf.append(def_macro)
        mf.save()
    #fed

