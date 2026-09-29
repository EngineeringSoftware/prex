import seutil as su
import re
import Levenshtein
import numpy as np
import pandas as pd
from itertools import zip_longest
from statistics import mean
from collections import Counter
import traceback
from itertools import combinations
from pathlib import Path


import ast
from collections import defaultdict
from typing import Dict, List, DefaultDict, Tuple, Any
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from llm_interpreter.macros import Macros
from llm_interpreter.language.metrics import HalsteadMetric, ExtendedCyclomaticMetric, DepDegreeMetric
from llm_interpreter.utils import (
    extract_content_between_tags,
    extract_boxed_answers,
    parse_op_prediction,
    parse_srp_prediction,
    parse_etp_prediction,
    read_from_json_file,
)
from llm_interpreter.results.results_ig_helper import IGResultsHelper


logger = su.log.get_logger(__name__)

REASONING_MODELS = [
    "deepseek-ai-DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B",
    "Qwen-QwQ-32B",
    "o3-mini",
    "gemini-2.5-pro",
    "gpt-5-mini",
    "gpt-5.4-mini",
    "gpt-5.4",
    "claude-sonnet-4-6",
    "Qwen-Qwen3-Coder-30B-A3B-Instruct",
]


def strip_in_place(text: str) -> str:
    return text.strip().rstrip("\n").strip('"')


@dataclass
class SRPResults:
    model_name: str = ""
    malformed_cnt: int = 0
    pred_rules: List[List[str]] = field(default_factory=list)
    true_rules: List[List[str]] = field(default_factory=list)
    first_mismatch_rules: List[str] = field(default_factory=list)


@dataclass
class ETPResults:
    model_name: str = ""
    malformed_cnt: int = 0
    pred_exec_trace: List[Dict[str, str]] = field(default_factory=list)
    true_exec_trace: List[Dict[str, str]] = field(default_factory=list)
    raw_exec_trace: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PCPResults:
    model_name: str = ""
    y_true: List[int] = field(default_factory=list)
    y_pred: List[int] = field(default_factory=list)
    rule_true: List[str] = field(default_factory=list)
    rule_pred: List[str] = field(default_factory=list)
    rules_pred: Dict[str, List[str]] = field(default_factory=dict)
    malformed_cnt: int = 0
    mutation_pattern: str = None
    error_rule_correct_cnt: int = 0

@dataclass
class NL2RuleResults:
    model_name: str = ""
    true_ans: List[int] = field(default_factory=list)
    pred_ans: List[int] = field(default_factory=list)
    true_mapped_ans: List[int] = field(default_factory=list)
    pred_mapped_ans: List[int] = field(default_factory=list)
    normal_cnt: int = 0
    malformed_cnt: int = 0
    correct_cnt: int = 0

@dataclass
class Rule2NLResults:
    model_name: str = ""
    true_ans: List[int] = field(default_factory=list)
    pred_ans: List[int] = field(default_factory=list)
    true_mapped_ans: List[int] = field(default_factory=list)
    pred_mapped_ans: List[int] = field(default_factory=list)
    normal_cnt: int = 0
    malformed_cnt: int = 0
    correct_cnt: int = 0

@dataclass
class MutateResults:
    model_name: str = ""
    normal_cnt: int = 0
    pattern_correct_cnt: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    pattern_var_correct_cnt_percentage: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    pattern_scores: DefaultDict[str, List[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    malformed_cnt: int = 0


@dataclass
class OPResults:
    model_name: str = ""
    normal_cnt: int = 0
    correct_cnt: int = 0
    var_correct_cnt_percentage: float = 0
    true_ans: List[List[int]] = field(default_factory=list)
    pred_ans: List[List[int]] = field(default_factory=list)
    true_ans_dict: List[Dict[str,int]] = field(default_factory=list)
    pred_ans_dict: List[Dict[str,int]] = field(default_factory=list)
    src_filename: List[str] = field(default_factory=list)
    malformed_cnt: int = 0


@dataclass
class IGResults:
    model_name: str = ""
    pgms: DefaultDict[str, list] = field(default_factory=lambda: defaultdict(list))
    valid_pgm_run_states: DefaultDict[str, list] = field(
        default_factory=lambda: defaultdict(list)
    )
    invalid_pgm_run_states: DefaultDict[str, list] = field(
        default_factory=lambda: defaultdict(list)
    )


class ResultsAnalyzer:
    condition_rules = {"Rule 65", "Rule 66", "Rule 69", "Rule 70"}

    def __init__(self, task: str, setup_name: str, exp_name: str, model_list: str, filter_dataset: Path = None, dataset_name="human_written"):
        self.results_dir = Macros.results_dir
        self.task = task
        self.setup_name = setup_name
        self.exp_name = exp_name
        self.model_list = model_list
        self.filter_dataset = filter_dataset
        self.dataset_name = dataset_name
        self.filter_programs: set = set()
        if self.filter_dataset:
            filter_dataset: list = su.io.load(self.filter_dataset)
            for data in filter_dataset:
                self.filter_programs.add(data['src-filename'])
            #rof
        #fi

    def raw_results_trace(self, model_name: str, strategy: str = "da"):
        exp_output_dir = (
            Macros.results_dir
            / self.task
            / self.setup_name
            / (self.exp_name if self.dataset_name == "human_written" else f"{self.exp_name}-{self.dataset_name}")
            / f"{model_name}-{strategy}"
        )
        if not exp_output_dir.exists():
            logger.warning(f"Directory '{exp_output_dir}' does not exist.")
            return

        if not exp_output_dir.is_dir():
            logger.warning(f"'{exp_output_dir}' is not a directory.")
            return

        # Find all JSONL files in the directory
        jsonl_files = list(exp_output_dir.glob("*.jsonl"))

        if not jsonl_files:
            logger.warning(f"No results files found in '{exp_output_dir}'")
            return
        results_across_runs = []
        for file_path in jsonl_files:
            res = su.io.load(file_path)
            new_res: list = []
            if self.filter_dataset:
                for r in res:
                    if r['src-filename'] in self.filter_programs:
                        new_res.append(r)
                    #rof
                #rof
                res = new_res
            #rof
            model_result = self._eval(
                res, model_name, strategy, True
            )
            results_across_runs.append(model_result)
        #
        return results_across_runs

        
    def analyze_results(
        self, prompt_strategies: List = None, merge_existing: bool = False
    ):
        experiments_results = {}
        experiments_scores = {}

        for model_name in self.model_list:
            for prompt_strategy in prompt_strategies:
                if model_name in REASONING_MODELS and prompt_strategy.value in [
                    "cot",
                    "few-shot-cot",
                ]:
                    continue
                exp_output_dir = (
                    Macros.results_dir
                    / self.task
                    / self.setup_name
                    / (self.exp_name if self.dataset_name == "human_written" else f"{self.exp_name}-{self.dataset_name}")
                    / (model_name.replace("/", "-") + f"-{prompt_strategy.value}")
                )
                logger.info(f"Analyzing results for {exp_output_dir}...")
                # load the jsonl files in the  exp_output_dir one by one
                if not exp_output_dir.exists():
                    logger.warning(f"Directory '{exp_output_dir}' does not exist.")
                    continue

                if not exp_output_dir.is_dir():
                    logger.warning(f"'{exp_output_dir}' is not a directory.")
                    continue

                # Find all JSONL files in the directory
                jsonl_files = list(exp_output_dir.glob("*.jsonl"))

                if not jsonl_files:
                    logger.warning(f"No results files found in '{exp_output_dir}'")
                    continue
                results_across_runs, scores_across_runs = [], []
                for file_path in jsonl_files:
                    res = su.io.load(file_path)    
                    new_res: list = []
                    if self.filter_dataset:
                        for r in res:
                            if r['src-filename'] in self.filter_programs:
                                new_res.append(r)
                            #rof
                        #rof
                        res = new_res
                    #rof
                    model_results, model_scores = self._eval(
                        res, model_name, prompt_strategy.value
                    )
                    su.io.dump(
                        exp_output_dir / f"metrics-{Path(file_path).stem}.json",
                        model_results,
                    )
                    su.io.dump(
                        exp_output_dir / f"scores-{Path(file_path).stem}.json",
                        model_scores,
                    )
                    results_across_runs.append(model_results)
                    scores_across_runs.append(model_scores)
                #
                # merge results across runs
                avg_results = average_metrics(merge_metrics(results_across_runs))
                merged_scores = merge_metrics(scores_across_runs)

                experiments_results[f"{model_name}-{prompt_strategy}"] = avg_results
                experiments_scores[f"{model_name}-{prompt_strategy}"] = merged_scores
        #
        self._save_metrics(
            experiments_results, experiments_scores, merge_existing=merge_existing
        )

    def _eval(self, results: List[dict], model_name: str, prompt_strategy: str, return_raw_trace: bool = False):
        try:
            if self.task == "op":
                if self.setup_name == "uk" or self.setup_name == "nk" or self.setup_name == "mk-unseen-gpt4o-1-token" or self.setup_name == "mk-unseen-caucasian-albanian":
                    model_result = self._analyze_op_uk(results, model_name)
                elif self.setup_name == "mk":
                    model_result = self._analyze_op_mk(results, model_name)
            elif self.task == "nl2rule":
                if self.setup_name == "uk":
                    model_result = self._analyze_nl2rule_uk(results, model_name)
                elif self.setup_name == "mk":
                    model_result = self._analyze_nl2rule_mk(results, model_name)
            elif self.task == "rule2nl":
                if self.setup_name == "uk":
                    model_result = self._analyze_rule2nl_uk(results, model_name)
                elif self.setup_name == "mk":
                    model_result = self._analyze_rule2nl_mk(results, model_name)
            elif self.task == "translate":
                if self.setup_name == "mk":
                    model_result = self._analyze_translate_mk(results, model_name)
            elif self.task == "pcp":
                if self.setup_name != "mk":
                    model_result = self._analyze_pcp_uk(results, model_name)
                else:
                    model_result = self._analyze_pcp_mk(results, model_name)
            elif self.task == "srp":
                if self.setup_name != "mk":
                    model_result = self._analyze_srp_uk(results, model_name)
                    assert len(model_result.true_rules) == len(model_result.pred_rules)
                else:
                    model_result = self._analyze_srp_mk(results, model_name)
            elif self.task == "etp":
                if self.setup_name != "mk":
                    model_result = self._analyze_etp_uk(results, model_name)
                else:
                    model_result = self._analyze_etp_mk(results, model_name)
            elif self.task == "ig":
                if self.setup_name != "mk":
                    model_result = self._analyze_ig_uk(results, model_name)
                else:
                    raise NotImplementedError(
                        "IG task is not implemented for mk setup yet"
                    )
            elif self.task == "iga":
                if self.setup_name != "mk":
                    model_result = self._analyze_iga_uk(results, model_name)
                else:
                    model_result = self._analyze_iga_mk(results, model_name)
            elif self.task == "igaf":
                if self.setup_name == "uk":
                    model_result = self._analyze_iga_uk(results, model_name, False)
                else:
                    model_result = self._analyze_iga_mk(results, model_name, False)
            else:
                raise NotImplementedError("Task not implemented")
            if return_raw_trace:
                return model_result
            (
                model_result,
                model_scores,
            ) = self._compute_metrics(model_result)
        except Exception as e:
            logger.error(
                f"Error analyzing {model_name} with {prompt_strategy}: {e} {traceback.format_exc()}"
            )
        return (model_result, model_scores)

    def _save_metrics(
        self,
        experiments_results: dict,
        experiments_scores: dict,
        merge_existing: bool = False,
    ):
        """
        Save the metrics and best model results to files.
        When merge_existing is True, update only keys present in experiments_*
        and leave other models' entries in the aggregated files unchanged.
        """
        file_name_pattern = f"{self.task}-{self.setup_name}-{self.exp_name}" if self.dataset_name == "human_written" else f"{self.task}-{self.setup_name}-{self.exp_name}-{self.dataset_name.replace('_','-')}"
        metrics_path = self.results_dir / f"metrics-{file_name_pattern}.json"
        scores_path = self.results_dir / f"scores-{file_name_pattern}.json"
        if merge_existing:
            if metrics_path.exists():
                merged = su.io.load(metrics_path)
                merged.update(experiments_results)
                experiments_results = merged
            if scores_path.exists():
                merged = su.io.load(scores_path)
                merged.update(experiments_scores)
                experiments_scores = merged
        best_model_file_name = f"metrics-best-model-{file_name_pattern}.json"
        best_model = self._find_best_model(experiments_results)
        su.io.dump(
            self.results_dir / best_model_file_name,
            best_model,
            su.io.Fmt.jsonNoSort,
        )
        # dump models' results
        su.io.dump(
            metrics_path,
            experiments_results,
            su.io.Fmt.jsonNoSort,
        )
        su.io.dump(
            scores_path,
            experiments_scores,
        )

    def _find_best_model(self, results: dict) -> dict:
        """
        Find the best models across each metric.
        """
        # Skip best model for ig for now
        if self.task == "ig" or self.task == "iga":
            return {}
        best_model = defaultdict(list)
        for model in list(results.keys()):
            for metric in results[model]:
                if isinstance(results[model][metric], dict):
                    continue
                if metric not in best_model:
                    best_model[metric].append(model)
                else:
                    if results[model][metric] > results[best_model[metric][0]][metric]:
                        best_model[metric] = [model]
                    elif (
                        results[model][metric] == results[best_model[metric][0]][metric]
                    ):
                        best_model[metric].append(model)
        return best_model
        #

    def _analyze_pcp_uk(
        self,
        results: List[dict],
        model_name: str,
    ) -> PCPResults:
        analysis = PCPResults(model_name=model_name)
        pattern = re.compile(r"(?:rule[\s-]*)?(\d+)", re.IGNORECASE) #re.compile(r"\b\d+\b") #r"rule\s*(\d+)"
        for result in results:
            if result["ans"] == "##success##":
                analysis.y_true.append(1)
                analysis.rule_true.append("None")
            else:
                analysis.y_true.append(0)
                analysis.rule_true.append(result["semantic-error-rule"])
                if result["semantic-error-rule"] not in analysis.rules_pred:
                    analysis.rules_pred[result["semantic-error-rule"]] = []
                # fi
            # fi
            pred = result["model-prediction"]
            pred_return_code, pred_rule = extract_pcp_prediction(pred)
            if pred_return_code == "":
                analysis.malformed_cnt += 1
            if pred_return_code == "##success##":
                analysis.y_pred.append(1)
                analysis.rule_pred.append("None")
            else:
                analysis.y_pred.append(0)
                #matches = re.findall(pattern, pred_rule, flags=re.IGNORECASE)
                matches = pattern.search(pred_rule)
                if matches:
                    analysis.rule_pred.append(f"Rule {matches.group(1)}")
                #if len(matches) == 1:
                #    analysis.rule_pred.append(f"Rule {matches[0]}")
                else:
                    analysis.rule_pred.append(pred_rule)
                # fi
                if result["semantic-error-rule"] in analysis.rules_pred:
                    analysis.rules_pred[result["semantic-error-rule"]].append(pred_rule)
                # fi
            # fi
        #

        return analysis

    def _analyze_pcp_mk(
        self,
        results: List[dict],
        model_name: str,
    ) -> Dict[str, PCPResults]:
        """
        Analyze the PCP task results for each mutation pattern.
        """
        result_by_pattern = defaultdict(list)
        mk_pcp_patterns = {
            "unseen",
            "addSub_mulDiv_negateRelation",
            "KeywordSwap",
            "KeywordObf",
        }
        for result in results:
            if result["mutation-pattern"] not in mk_pcp_patterns:
                continue
            result_by_pattern[result["mutation-pattern"]].append(result)
        #
        model_result_by_pattern = {}
        for pattern, pattern_results in result_by_pattern.items():
            model_result = self._analyze_pcp_uk(pattern_results, model_name)
            model_result_by_pattern[pattern] = model_result
        #
        return model_result_by_pattern

    def _analyze_op_uk(
        self,
        results: List[dict],
        model_name: str,
    ) -> OPResults:
        """
        Collect results for op-uk task.
        """
        analysis = OPResults(model_name=model_name)
        for result in results:
            analysis.normal_cnt += 1
            true_list: list = []
            if "ground-truth" in result:
                [_, true_dict] = extract_op_prediction(result["ground-truth"])
            else:
                true_dict = result["final-state"]
            sorted_keys = sorted(true_dict.keys())
            for key in sorted_keys:
                true_list.append(true_dict[key])
            # rof
            analysis.true_ans_dict.append(true_dict)
            analysis.true_ans.append(true_list)
            [pred_list, pred_dict] = extract_op_prediction(result["model-prediction"])
            analysis.pred_ans.append(pred_list)
            analysis.pred_ans_dict.append(pred_dict)
            analysis.src_filename.append(result["src-filename"])
            if len(pred_list) == 0:
                analysis.malformed_cnt += 1
            elif true_list == pred_list:
                analysis.correct_cnt += 1
                analysis.var_correct_cnt_percentage += 1.0
            else:
                var_correct_cnt: int = 0
                non_zero_var: int = 0
                for key in sorted_keys:
                    if true_dict[key] != 0:
                        non_zero_var += 1
                        if key in pred_dict and true_dict[key] == pred_dict[key]:
                            var_correct_cnt += 1
                        #fi
                    #fi
                #rof
                analysis.var_correct_cnt_percentage += (var_correct_cnt / non_zero_var if non_zero_var != 0 else 1.0)
            #fi
        assert len(analysis.true_ans) == len(analysis.pred_ans)
        return analysis

    def _analyze_op_mk(
        self,
        results: List[dict],
        model_name: str,
    ) -> MutateResults:
        analysis = MutateResults(model_name=model_name)
        cur_pattern = ""
        normal_cnt = 0
        for result in results:
            # NOTE: in the Neurips25 paper, we only report two patterns.
            if result["mutation-pattern"] not in [
                "unseen",
                "addSub_mulDiv_negateRelation",
                "unseen-gpt4o-1-token",
                "KeywordSwap",
                "KeywordObf"    
            ]:
                continue
            if cur_pattern and result["mutation-pattern"] != cur_pattern:
                analysis.normal_cnt = normal_cnt
                normal_cnt = 0
                assert len(analysis.pattern_scores[cur_pattern]) == analysis.normal_cnt
            cur_pattern = result["mutation-pattern"]
            normal_cnt += 1
            #
            true_list: list = []
            if "ground-truth" in result:
                [_, true_dict] = extract_op_prediction(result["ground-truth"])
            else:
                true_dict = result["final-state"]
            sorted_keys = sorted(true_dict.keys())
            for key in sorted_keys:
                true_list.append(true_dict[key])
            # rof
            [pred_list, pred_dict] = extract_op_prediction(result["model-prediction"])
            analysis.pattern_scores[cur_pattern].append(0)
            if len(pred_list) == 0:
                analysis.malformed_cnt += 1
            elif true_list == pred_list:
                analysis.pattern_correct_cnt[cur_pattern] += 1
                analysis.pattern_var_correct_cnt_percentage[cur_pattern] += 1.0
                analysis.pattern_scores[cur_pattern][-1] = 1
            else:
                var_correct_cnt: int = 0
                non_zero_var: int = 0
                for key in sorted_keys:
                    if true_dict[key] != 0:
                        non_zero_var += 1
                        if key in pred_dict and true_dict[key] == pred_dict[key]:
                            var_correct_cnt += 1
                        #fi
                    #fi
                #rof
                analysis.pattern_var_correct_cnt_percentage[cur_pattern] += (var_correct_cnt / non_zero_var if non_zero_var != 0 else 1.0)
            #fi
                
        return analysis

    def _analyze_nl2rule_uk(
        self,
        results: List[dict],
        model_name: str,
    ) -> NL2RuleResults:
        """
        Collect results for nl2rule-uk task.
        """

        def extract_rule_number(s: str) -> int:
            m = re.search(r"\brule\s*(\d+)\b", s, flags=re.IGNORECASE)
            if not m:
                raise ValueError(f"No rule number found in: {s!r}")
            return int(m.group(1))
        #fed
        
        analysis = NL2RuleResults(model_name=model_name)
        for result in results:
            pred = extract_llm_prediction(llm_output=result["model-prediction"], tag="answer")
            try:
                pred = extract_rule_number(pred)
            except:
                analysis.malformed_cnt += 1
                continue
            #yrt
            analysis.normal_cnt += 1
            true = result["answer_index"]
            analysis.true_ans.append(true)
            analysis.true_mapped_ans.append(result["answer_rule_id"])
            analysis.pred_ans.append(pred)
            analysis.pred_mapped_ans.append(result["option_rule_ids"][pred])
            if true == pred:
                analysis.correct_cnt += 1
            #fi
        #rof
        assert len(analysis.true_ans) == len(analysis.pred_ans)
        return analysis

    def _analyze_nl2rule_mk(
        self,
        results: List[dict],
        model_name: str,
    ) -> NL2RuleResults:
        """
        Collect results for nl2rule-mk task.
        """

        analysis_by_pattern = defaultdict(list)
        for result in results:
            analysis_by_pattern[result["mutation-pattern"]].append(result)
        #rof

        model_result_by_pattern = {}
        for pattern, pattern_results in analysis_by_pattern.items():
            model_result = self._analyze_nl2rule_uk(pattern_results, model_name)
            model_result_by_pattern[pattern] = model_result
        #
        return model_result_by_pattern

    def _analyze_rule2nl_uk(
        self,
        results: List[dict],
        model_name: str,
    ) -> Rule2NLResults:
        """
        Collect results for rule2nl-uk task.
        """

        def extract_rule_number(s: str) -> int:
            m = re.search(r"\bdescription\s*(\d+)\b", s, flags=re.IGNORECASE)
            if not m:
                raise ValueError(f"No description number found in: {s!r}")
            return int(m.group(1))
        #fed
        
        analysis = Rule2NLResults(model_name=model_name)
        for result in results:
            pred = extract_llm_prediction(llm_output=result["model-prediction"], tag="answer")
            try:
                pred = extract_rule_number(pred)
            except:
                analysis.malformed_cnt += 1
                continue
            #yrt
            analysis.normal_cnt += 1
            true = result["answer_index"]
            analysis.true_ans.append(true)
            analysis.true_mapped_ans.append(result["answer_rule_id"])
            analysis.pred_ans.append(pred)
            analysis.pred_mapped_ans.append(result["option_rule_ids"][pred])
            if true == pred:
                analysis.correct_cnt += 1
            #fi
        #rof
        assert len(analysis.true_ans) == len(analysis.pred_ans)
        return analysis

    def _analyze_rule2nl_mk(
        self,
        results: List[dict],
        model_name: str,
    ) -> NL2RuleResults:
        """
        Collect results for rule2nl-mk task.
        """

        analysis_by_pattern = defaultdict(list)
        for result in results:
            analysis_by_pattern[result["mutation-pattern"]].append(result)
        #rof

        model_result_by_pattern = {}
        for pattern, pattern_results in analysis_by_pattern.items():
            model_result = self._analyze_rule2nl_uk(pattern_results, model_name)
            model_result_by_pattern[pattern] = model_result
        #
        return model_result_by_pattern


    def _analyze_translate_mk(
        self,
        results: List[dict],
        model_name: str,
    ) -> MutateResults:
        """
        Extract the generated Python code and execute it to get the stdout.
        """
        analysis = MutateResults(model_name=model_name)
        cur_pattern = ""
        normal_cnt = 0
        for result in results:
            if cur_pattern and result["mutation-pattern"] != cur_pattern:
                analysis.normal_cnt = normal_cnt
                normal_cnt = 0
                assert len(analysis.pattern_scores[cur_pattern]) == analysis.normal_cnt
            cur_pattern = result["mutation-pattern"]
            normal_cnt += 1
            #
            pred = result["model-prediction"]
            python_code = extract_python_code(pred)
            stdout = None
            if python_code:
                stdout = execute_python_code(python_code)
            analysis.pattern_scores[cur_pattern].append(0)
            if not stdout:
                analysis.malformed_cnt += 1
            else:
                if str(stdout) == str(result["ans"]):
                    analysis.pattern_correct_cnt[cur_pattern] += 1
                    analysis.pattern_scores[cur_pattern][-1] = 1
        return analysis

    def _analyze_translate_uk(
        self,
        results: List[dict],
        model_name: str,
    ) -> OPResults:
        """
        Extract the generated Python code and execute it to get the stdout for UK setup.
        """
        analysis = OPResults(model_name=model_name)
        for result in results:
            analysis.normal_cnt += 1
            analysis.true_ans.append(int(result["ans"]))
            pred = result["model-prediction"]
            python_code = extract_python_code(pred)
            stdout = None
            if python_code:
                stdout = execute_python_code(python_code)
            if not stdout:
                analysis.malformed_cnt += 1
                analysis.pred_ans.append("")
            else:
                analysis.pred_ans.append(stdout)
                if str(stdout) == str(result["ans"]):
                    analysis.correct_cnt += 1
        assert len(analysis.true_ans) == len(analysis.pred_ans)
        return analysis

    def _analyze_srp_uk(
        self,
        results: List[dict],
        model_name: str,
    ):
        analysis = SRPResults(model_name=model_name)
        for result in results:
            pred_rules_list: List[List[str]] = extract_srp_prediction(
                result["model-prediction"]
            )  # extract the list of list of rules
            gt_rules_list = []
            for d in result["sampled-statements"]:
                rule_list = [r.replace("Rule", "").strip() for r in d["rules"]]
                gt_rules_list.append(rule_list)
            #
            gt_rules_num = len(gt_rules_list)
            pred_rules_num = len(pred_rules_list)
            if not pred_rules_list:
                analysis.malformed_cnt += 1
                analysis.pred_rules.extend([[] for _ in range(gt_rules_num)])
            elif pred_rules_num < gt_rules_num:
                analysis.malformed_cnt += 1
                analysis.pred_rules.extend(pred_rules_list)
                # add empty lists for the missing rules
                analysis.pred_rules.extend(
                    [[] for _ in range(gt_rules_num - pred_rules_num)]
                )
            elif pred_rules_num > gt_rules_num:
                analysis.malformed_cnt += 1
                analysis.pred_rules.extend(pred_rules_list[:gt_rules_num])
            else:
                analysis.pred_rules.extend(pred_rules_list)
            analysis.true_rules.extend(gt_rules_list)
        #
        return analysis

    def _analyze_srp_mk(
        self,
        results: List[dict],
        model_name: str,
    ) -> Dict[str, SRPResults]:
        result_by_pattern = defaultdict(list)
        for result in results:
            if result["mutation-pattern"] not in [
                "unseen",
                "addSub_mulDiv_negateRelation",
                "KeywordSwap",
                "KeywordObf"    
            ]:
                continue
            result_by_pattern[result["mutation-pattern"]].append(result)
        #
        model_result_by_pattern = {}
        for pattern, pattern_results in result_by_pattern.items():
            model_result = self._analyze_srp_uk(pattern_results, model_name)
            model_result_by_pattern[pattern] = model_result
        #
        return model_result_by_pattern

    def _analyze_etp_uk(
        self,
        results: List[dict],
        model_name: str,
    ):
        analysis = ETPResults(model_name=model_name)
        for result in results:
            pred_exec_trace = extract_etp_prediction(result["model-prediction"])
            if pred_exec_trace == []:
                analysis.malformed_cnt += 1
            gt_exec_trace = extract_etp_prediction(result["ground-truth"])
            analysis.pred_exec_trace.append(pred_exec_trace)
            analysis.true_exec_trace.append(gt_exec_trace)
            analysis.raw_exec_trace.append(result["ground-truth"])
        #
        return analysis

    def _analyze_etp_mk(
        self,
        results: List[dict],
        model_name: str,
    ) -> Dict[str, ETPResults]:
        result_by_pattern = defaultdict(list)
        for result in results:
            if result["mutation-pattern"] not in [
                "unseen",
                "addSub_mulDiv_negateRelation",
                "KeywordSwap",
                "KeywordObf"    
            ]:
                continue
            result_by_pattern[result["mutation-pattern"]].append(result)
        #
        model_result_by_pattern = {}
        for pattern, pattern_results in result_by_pattern.items():
            model_result = self._analyze_etp_uk(pattern_results, model_name)
            model_result_by_pattern[pattern] = model_result
        #
        return model_result_by_pattern

    def _analyze_ig_uk(
        self,
        results: List[dict],
        model_name: str,
    ) -> IGResults:
        tmp_out_dir = f"{self.task}-{self.setup_name}-{self.exp_name}-{model_name}"
        helper = IGResultsHelper(
            res_dir=tmp_out_dir,
            task=self.task,
            setup_name=self.setup_name,
            exp_name=self.exp_name,
        )
        analysis = IGResults(model_name=model_name)
        for result in results:
            impl_lang = result["impl-language"]
            if isinstance(result["model-prediction"], str):
                output_file = helper.extract_program(result, result["model-prediction"])
                vp_states, ivp_states = helper.interpret_program(output_file)
                analysis.pgms[impl_lang].append(result["model-prediction"])
                analysis.valid_pgm_run_states[impl_lang].append(vp_states)
                analysis.invalid_pgm_run_states[impl_lang].append(ivp_states)
            elif isinstance(result["model-prediction"], list):
                for i, program in enumerate(result["model-prediction"]):
                    output_file = helper.extract_program(result, program, idx1=i)
                    vp_states, ivp_states, _ = helper.interpret_program(
                        output_file, idx1=i
                    )
                    analysis.pgms[impl_lang].append(output_file)
                    analysis.valid_pgm_run_states[impl_lang].append(vp_states)
                    analysis.invalid_pgm_run_states[impl_lang].append(ivp_states)
            else:
                raise ValueError("Unsupported model-prediction type")
        return analysis

    def _analyze_iga_uk(
        self,
        results: List[dict],
        model_name: str,
        save_feedback: bool = True,
    ) -> IGResults:
        tmp_out_dir = f"{self.task}-{self.setup_name}-{self.exp_name}-{model_name}"
        helper = IGResultsHelper(
            res_dir=tmp_out_dir,
            task=self.task,
            setup_name=self.setup_name,
            exp_name=self.exp_name,
        )
        analysis = IGResults(model_name=model_name)
        feedbacks = []
        for result in results:
            impl_lang = result["impl-language"]
            idx = result.get("idx", 0)
            if impl_lang == "Python":
                visitor_path = Macros.antlr_imp_python_uk_dir / "EvalVisitor.py"
            elif impl_lang == "Java":
                visitor_path = Macros.antlr_imp_java_uk_dir / "EvalVisitor.java"
            else:
                raise ValueError(f"Unsupported impl language: {impl_lang}")

            feedback_dict = {
                "model-name": model_name,
                "task": self.task,
                "setup-name": self.setup_name,
                "exp-name": self.exp_name,
                "impl-language": impl_lang,
            }
            if isinstance(result["model-prediction"], str):
                output_file = helper.extract_program(result, result["model-prediction"])
                helper.copy_to_dest(output_file, visitor_path)
                vp_states, ivp_states, feedback = helper.interpret_program(visitor_path)
                feedback_dict["idx"] = 0
                feedback_dict["feedback"] = feedback
                feedbacks.append(feedback_dict.copy())
                key = f"{impl_lang}-{idx}" if self.task == "igaf" else f"{impl_lang}"
                analysis.pgms[key].append(output_file)
                analysis.valid_pgm_run_states[key].append(vp_states)
                analysis.invalid_pgm_run_states[key].append(ivp_states)
            elif isinstance(result["model-prediction"], list):
                for i, program in enumerate(result["model-prediction"]):
                    output_file = helper.extract_program(
                        result, program, idx1=i, idx2=idx
                    )
                    helper.copy_to_dest(output_file, visitor_path)
                    vp_states, ivp_states, feedback = helper.interpret_program(
                        visitor_path, idx1=i, idx2=idx
                    )
                    feedback_dict["idx"] = i
                    feedback_dict["feedback"] = feedback
                    feedbacks.append(feedback_dict.copy())
                    key = (
                        f"{impl_lang}-{idx}" if self.task == "igaf" else f"{impl_lang}"
                    )
                    analysis.pgms[key].append(output_file)
                    analysis.valid_pgm_run_states[key].append(vp_states)
                    analysis.invalid_pgm_run_states[key].append(ivp_states)
            else:
                raise ValueError("Unsupported model-prediction type")

        if save_feedback:
            su.io.dump(
                Macros.data_dir
                / "dataset"
                / f"dataset-igaf-{self.setup_name}-{self.exp_name}.jsonl",
                feedbacks,
            )
        return analysis

    def _analyze_iga_mk(
        self,
        results: List[dict],
        model_name: str,
        save_feedback: bool = True,
    ) -> Dict[str, IGResults]:
        tmp_out_dir = f"{self.task}-{self.setup_name}-{self.exp_name}-{model_name}"
        helper = IGResultsHelper(
            res_dir=tmp_out_dir,
            task=self.task,
            setup_name=self.setup_name,
            exp_name=self.exp_name,
        )
        analysis = {
            "ks": IGResults(model_name=model_name),
            "ko": IGResults(model_name=model_name),
        }
        feedbacks = []
        for result in results:
            mutation_pattern = result.get("mutation-pattern", "")
            if mutation_pattern == "":
                mutation_pattern = result["feedback"].get("mutation_pattern", "")
            assert mutation_pattern in [
                "ks",
                "ko",
            ], f"Unsupported mutation pattern: {mutation_pattern}"
            impl_lang = result["impl-language"]
            idx = result.get("idx", 0)
            if impl_lang == "Python":
                if mutation_pattern == "ks":
                    visitor_path = Macros.antlr_imp_python_ks_dir / "EvalVisitor.py"
                elif mutation_pattern == "ko":
                    visitor_path = Macros.antlr_imp_python_ko_dir / "EvalVisitor.py"
                else:
                    raise ValueError(
                        f"Unsupported mutation pattern: {mutation_pattern}"
                    )
            elif impl_lang == "Java":
                if mutation_pattern == "ks":
                    visitor_path = Macros.antlr_imp_java_ks_dir / "EvalVisitor.java"
                elif mutation_pattern == "ko":
                    visitor_path = Macros.antlr_imp_java_ko_dir / "EvalVisitor.java"
                else:
                    raise ValueError(
                        f"Unsupported mutation pattern: {mutation_pattern}"
                    )
            else:
                raise ValueError(f"Unsupported impl language: {impl_lang}")

            feedback_dict = {
                "model-name": model_name,
                "task": self.task,
                "setup-name": self.setup_name,
                "exp-name": self.exp_name,
                "impl-language": impl_lang,
            }
            if isinstance(result["model-prediction"], str):
                output_file = helper.extract_program(
                    result,
                    result["model-prediction"],
                    mutation_pattern=mutation_pattern,
                )
                helper.copy_to_dest(output_file, visitor_path)
                vp_states, ivp_states, feedback = helper.interpret_program(
                    visitor_path,
                    mutation_pattern=mutation_pattern,
                )
                feedback_dict["idx"] = 0
                feedback_dict["feedback"] = feedback
                feedbacks.append(feedback_dict.copy())
                key = f"{impl_lang}-{idx}" if self.task == "igaf" else f"{impl_lang}"
                analysis[mutation_pattern].pgms[key].append(output_file)
                analysis[mutation_pattern].valid_pgm_run_states[key].append(vp_states)
                analysis[mutation_pattern].invalid_pgm_run_states[key].append(
                    ivp_states
                )
            elif isinstance(result["model-prediction"], list):
                for i, program in enumerate(result["model-prediction"]):
                    output_file = helper.extract_program(
                        result,
                        program,
                        idx1=i,
                        idx2=idx,
                        mutation_pattern=mutation_pattern,
                    )
                    helper.copy_to_dest(output_file, visitor_path)
                    vp_states, ivp_states, feedback = helper.interpret_program(
                        visitor_path,
                        idx1=i,
                        idx2=idx,
                        mutation_pattern=mutation_pattern,
                    )
                    feedback_dict["idx"] = i
                    feedback_dict["feedback"] = feedback
                    feedbacks.append(feedback_dict.copy())
                    key = (
                        f"{impl_lang}-{idx}" if self.task == "igaf" else f"{impl_lang}"
                    )
                    analysis[mutation_pattern].pgms[key].append(output_file)
                    analysis[mutation_pattern].valid_pgm_run_states[key].append(
                        vp_states
                    )
                    analysis[mutation_pattern].invalid_pgm_run_states[key].append(
                        ivp_states
                    )
            else:
                raise ValueError("Unsupported model-prediction type")

        if save_feedback:
            su.io.dump(
                Macros.data_dir
                / "dataset"
                / f"dataset-igaf-{self.setup_name}-{self.exp_name}.jsonl",
                feedbacks,
            )
        return analysis

    def _compute_metrics(
        self,
        analysis: OPResults | MutateResults | PCPResults | SRPResults | IGResults | NL2RuleResults | Rule2NLResults,
    ) -> Tuple[dict, dict]:
        """Calculate metrics from analysis results.
        Returns:
            Tuple:
                - dict containing computed metrics
                - dict containing scores of each example
        """
        if (self.task == "op" and self.setup_name == "mk") or (
            self.task == "translate" and self.setup_name == "mk"):
            metrics_dict = defaultdict(dict)
            scores_dict = defaultdict(dict)
            # pattern-wise results
            for mutation_pattern in ['addSub_mulDiv_negateRelation', 'unseen', 'unseen-gpt4o-1-token', 'KeywordSwap', 'KeywordObf']:
                metrics_dict[mutation_pattern]['acc'] = 0
                metrics_dict[mutation_pattern]['var-acc'] = 0
                scores_dict[mutation_pattern]['acc'] = 0
            #rof
            for mutation_pattern in analysis.pattern_correct_cnt:
                metrics_dict[mutation_pattern] = {
                    "acc": analysis.pattern_correct_cnt[mutation_pattern]
                    / analysis.normal_cnt,
                    "var-acc": analysis.pattern_var_correct_cnt_percentage[mutation_pattern]
                    / analysis.normal_cnt
                }
                scores_dict[mutation_pattern] = {
                    "acc": analysis.pattern_scores[mutation_pattern]
                }
            # aggreagate pattern-wise results
            final_metrics_dict, final_scores_dict = (
                self._aggregate_pattern_wise_metrics(metrics_dict, scores_dict)
            )
            final_metrics_dict["malformed-count"] = analysis.malformed_cnt
            return final_metrics_dict, final_scores_dict
        elif self.task == "op" or (
            self.task == "translate" and self.setup_name == "uk"
        ):
            if self.setup_name in {"uk", "nk", "mk-unseen-gpt4o-1-token", "mk-unseen-caucasian-albanian"}:
                metrics_dict = {
                    "acc": analysis.correct_cnt / analysis.normal_cnt,
                    "var-acc": analysis.var_correct_cnt_percentage / analysis.normal_cnt,
                    "malformed-count": analysis.malformed_cnt,
                }
                scores_dict = {
                    "acc": [
                        1 if analysis.true_ans[i] == analysis.pred_ans[i] else 0
                        for i in range(len(analysis.true_ans))
                    ],
                }
                return metrics_dict, scores_dict
        elif (self.task == "nl2rule" or self.task == "rule2nl") and self.setup_name == "uk":
            metrics_dict = {
                "acc": analysis.correct_cnt / analysis.normal_cnt,
                "malformed-count": analysis.malformed_cnt,
            }
            scores_dict = {
                "acc": [1 if true == pred else 0 for true, pred in zip(analysis.true_ans, analysis.pred_ans)],
            }
            return metrics_dict, scores_dict
        elif (self.task == "nl2rule" or self.task == "rule2nl") and self.setup_name == "mk":
            metrics_dict = defaultdict(dict)
            scores_dict = defaultdict(dict)
            # pattern-wise results
            for mutation_pattern, result in analysis.items():
                metrics_dict[mutation_pattern] = {
                    "acc": result.correct_cnt / result.normal_cnt,
                    "malformed-count": result.malformed_cnt,
                }
                scores_dict[mutation_pattern] = {
                    "acc": [1 if true == pred else 0 for true, pred in zip(result.true_ans, result.pred_ans)]
                }
            # aggreagate pattern-wise results
            final_metrics_dict, final_scores_dict = (
                self._aggregate_pattern_wise_metrics(metrics_dict, scores_dict)
            )
            return final_metrics_dict, final_scores_dict
        elif self.task == "pcp":
            if self.setup_name != "mk":
                return self.evaluate_pcp_task(analysis)
            else:
                metrics_dict = {}
                scores_dict = {}
                for pattern, result in analysis.items():
                    metrics_dict[pattern], scores_dict[pattern] = (
                        self.evaluate_pcp_task(result)
                    )
                return self._aggregate_pattern_wise_metrics(metrics_dict, scores_dict)
        elif self.task == "srp":
            if self.setup_name != "mk":
                return self.evaluate_srp_task(analysis)
            else:
                metrics_dict = {}
                scores_dict = {}
                for pattern, result in analysis.items():
                    metrics_dict[pattern], scores_dict[pattern] = (
                        self.evaluate_srp_task(result)
                    )
                return self._aggregate_pattern_wise_metrics(metrics_dict, scores_dict)
        elif self.task == "etp":
            if self.setup_name != "mk":
                return self.evaluate_etp_task(analysis)
            else:
                metrics_dict = {}
                scores_dict = {}
                for pattern, result in analysis.items():
                    metrics_dict[pattern], scores_dict[pattern] = (
                        self.evaluate_etp_task(result)
                    )
                return self._aggregate_pattern_wise_metrics(metrics_dict, scores_dict)
        elif self.task == "ig":
            if self.setup_name != "mk":
                return IGResultsHelper.compute_metrics(analysis)
            else:
                raise NotImplementedError("IG task is not implemented for mk setup yet")
        elif self.task == "iga" or self.task == "igaf":
            if self.setup_name != "mk":
                return IGResultsHelper.compute_metrics(analysis)
            else:
                return {
                    mutation_pattern: IGResultsHelper.compute_metrics(
                        analysis[mutation_pattern]
                    )[0]
                    for mutation_pattern in analysis.keys()
                }, {}
        else:
            raise NotImplementedError("Task not implemented")

    def evaluate_srp_task(self, results: SRPResults) -> dict:
        """
        Evaluate the SRP task results.
        """
        edit_distance_scores = []
        xmatch_scores = []
        first_mismatch_rule = Counter()
        most_correct_rules = Counter()
        rule_occurrence_count = Counter()
        for pred_rules, true_rules in zip(results.pred_rules, results.true_rules):
            if pred_rules == true_rules:
                xmatch_scores.append(1)
            else:
                xmatch_scores.append(0)
            for x in true_rules:
                rule_occurrence_count[x] += 1
            # rof
            for x, y in zip_longest(pred_rules, true_rules, fillvalue=""):
                if x != y:
                    first_mismatch_rule[y] += 1
                    break
                else:
                    most_correct_rules[y] += 1
            # rof
            # Calculate edit distance for each pair of predicted and true rules
            try:
                pred_rules_str = " ".join(pred_rules)
                true_rules_str = " ".join(true_rules)
                score = edit_distance(pred_rules_str, true_rules_str)
                edit_distance_scores.append(score)
            except Exception as e:
                edit_distance_scores.append(0)
                logger.warning(
                    f"Error calculating edit distance: {e}. Predicted: {pred_rules}, True: {true_rules}"
                )
        #
        # only keep first mismatch rules that have mismatch rate > 10%
        first_mismatch_rule = {
            k: (v / rule_occurrence_count[k])
            for k, v in first_mismatch_rule.items()
            if k in rule_occurrence_count
        }
        most_correct_rules = {
            k: (v / rule_occurrence_count[k])
            for k, v in most_correct_rules.items()
            if k in rule_occurrence_count
        }
        metrics_dict = {
            "edit-distance-avg": mean(edit_distance_scores),
            "xmatch-accuracy": mean(xmatch_scores),
            "malformed-count": results.malformed_cnt,
            "first-mismatch-rule": dict(
                sorted(first_mismatch_rule.items(), key=lambda x: x[1], reverse=True)
            ),
            "most-correct-rules": dict(
                sorted(most_correct_rules.items(), key=lambda x: x[1], reverse=True)
            ),
        }
        scores_dict = {
            "edit-distance": edit_distance_scores,
            "xmatch-accuracy": xmatch_scores,
        }
        return metrics_dict, scores_dict

    def evaluate_etp_task(self, results: ETPResults) -> dict:
        """
        Evaluate the ETP task results.
        """
        xmatch_scores = []
        var_stmt_sim_scores = []
        stmt_sim_scores = []
        percent_trace_match = []
        approx_match_scores = []
        final_state_match = []
        state_mismatch: int = 0
        #state_rule_mismatch: int = 0
        rule_mismatch: dict = defaultdict(int)
        rule_mismatch_pred: dict = defaultdict(int)
        state_rule_mismatch: dict = defaultdict(int)
        mismatch_count: int = 0
        
        # control_flow_errors: int = 0
        # computation_errors: int = 0
        # long_execution_step: int = 0
        # condition_error: int = 0
        # total_data_points: int = 0
        for pred_exe, true_exe, raw_exe in zip(
            results.pred_exec_trace, results.true_exec_trace, results.raw_exec_trace
        ):
            if pred_exe == true_exe:
                xmatch_scores.append(1)
                approx_match_scores.append(1)
                final_state_match.append(1)
                percent_trace_match.append(100)
            else:
                index_mismatch: int = -1
                min_iter_range: int = min(len(pred_exe), len(true_exe))
                max_iter_range: int = max(len(pred_exe), len(true_exe))
                for j in range(min_iter_range):
                    if pred_exe[j] != true_exe[j]:
                        mismatch_count += 1
                        if pred_exe[j]['program_state'] != true_exe[j]['program_state'] and pred_exe[j]['rule'] == true_exe[j]['rule']:
                            state_mismatch += 1
                        elif pred_exe[j]['program_state'] == true_exe[j]['program_state'] and pred_exe[j]['rule'] != true_exe[j]['rule']:
                            rule_mismatch[f"Rule {true_exe[j]['rule']}-{pred_exe[j]['rule']}"] += 1
                        else:
                            state_rule_mismatch[f"RuleBoth {true_exe[j]['rule']}-{pred_exe[j]['rule']}"] += 1
#                            rule_mismatch[f"Rule {true_exe[j]['rule']}"] += 1
#                            rule_mismatch_pred[f"RulePred {pred_exe[j]['rule']}"] += 1                            
                        #fi
                        index_mismatch = j + 1
                        break
                    # fi
                # rof
                match index_mismatch:
                    case -1:
                        percent_trace_match.append(
                            (min_iter_range / max_iter_range) * 100
                        )
                    case _:
                        percent_trace_match.append((j / max_iter_range) * 100)
                xmatch_scores.append(0)
                if len(pred_exe) >= len(true_exe):
                    approx_match_scores.append(0)
                    if pred_exe[-1] == true_exe[-1]:
                        final_state_match.append(1)
                    else:
                        final_state_match.append(0)
                elif len(pred_exe) == 0:
                    approx_match_scores.append(0)
                    final_state_match.append(0)
                else:
                    if pred_exe[-1] != true_exe[-1]:
                        approx_match_scores.append(0)
                        final_state_match.append(0)
                    else:
                        i: int = 0
                        j: int = 0
                        match_count: int = 0
                        while i < len(true_exe) and j < len(pred_exe):
                            if pred_exe[j] == true_exe[i]:
                                match_count += 1
                                i += 1
                                j += 1
                            else:
                                i += 1
                            # fi
                        #
                        if match_count == len(pred_exe):
                            approx_match_scores.append(1)
                        else:
                            approx_match_scores.append(0)
                        # fi
                        final_state_match.append(1)
            var_stmt_sim_scores.append(variable_statement_overlap(pred_exe, true_exe))
            stmt_sim_scores.append(statement_overlap(pred_exe, true_exe))

            # We are computing what is the first-point of failure: is it computation or control-flow error
            # We only check first-point since control-flow and computation errors beyond the first-point are
            # related to each other.
        #     i = 0
        #     for pe, te in zip_longest(pred_exe, true_exe, fillvalue={}):
        #         print(i)
        #         gt_exec = raw_exe[i - 1] if i > 0 else None
        #         if pe != te:
        #             total_data_points += 1
        #             if not te.get("rule", 0):
        #                 long_execution_step += 1
        #             elif pe.get("rule", 0) != te.get("rule", -1):
        #                 if (
        #                     gt_exec
        #                     and len(set(gt_exec["rule"]) & self.condition_rules) > 0
        #                 ):
        #                     # the previous statement is related to conditions
        #                     condition_error += 1
        #                 else:
        #                     control_flow_errors += 1
        #             elif pe["program_state"] != te["program_state"]:
        #                 computation_errors += 1
        #             break
        #         # fi
        #         i += 1
        #     # rof
        # #

        # Group percentages of traces that match
        percent_trace_dict: dict = {}
        percent_trace_dict_str_keys: dict = {}
        for i in range(1, 11):
            percent_trace_dict[i * 10] = 0
        # rof

        for percent_match in percent_trace_match:
            for i in range(1, 11):
                if percent_match >= (i * 10):
                    percent_trace_dict[i * 10] += 1
                # fi
            # rof
        # rof

        for key, value in percent_trace_dict.items():
            percent_trace_dict_str_keys[key] = (
                percent_trace_dict[key] / len(results.raw_exec_trace)
            ) * 100
        # rof

        metrics_dict = {
            "xmatch-accuracy": mean(xmatch_scores),
            "var-stmt-jaccard-avg": mean(var_stmt_sim_scores),
            "stmt-jaccard-avg": mean(stmt_sim_scores),
            "malformed-count": results.malformed_cnt,
            "percentage-trace-match": percent_trace_dict_str_keys,
            "approx-match-accuracy": mean(approx_match_scores),
            "final-state-match": mean(final_state_match),
            "computation-errors": state_mismatch / mismatch_count,
            "rule-errors": sum(list(rule_mismatch.values())) / mismatch_count,
            "computation-and-rule-errors": sum(list(state_rule_mismatch.values())) / mismatch_count,
            
            # "control-flow-errors": control_flow_errors / total_data_points,
            # "computation-errors": computation_errors / total_data_points,
            # "long-execution-errors": long_execution_step / total_data_points,
            # "condition-errors": condition_error / total_data_points,
        } | rule_mismatch | state_rule_mismatch
        scores_dict = {
            "xmatch-accuracy": xmatch_scores,
            "var-stmt-jaccard-avg": var_stmt_sim_scores,
            "stmt-jaccard-avg": stmt_sim_scores,
        }
        return metrics_dict, scores_dict

    def evaluate_pcp_task(self, results: PCPResults) -> Tuple[dict, dict]:
        """
        Computes and prints various evaluation metrics for binary classification.

        Args:
            y_true (array-like): Ground truth (correct) labels.
            y_pred (array-like): Predicted labels as returned by the classifier.
        """

        y_true = results.y_true
        y_pred = results.y_pred
        # rules_pred = results.rules_pred

        precision_positive = precision_score(y_true, y_pred, pos_label=1)
        recall_positive = recall_score(y_true, y_pred, pos_label=1)
        f1_positive = f1_score(y_true, y_pred, pos_label=1)

        precision_negative = precision_score(y_true, y_pred, pos_label=0)
        recall_negative = recall_score(y_true, y_pred, pos_label=0)
        f1_negative = f1_score(y_true, y_pred, pos_label=0)

        # compute the recall for each violated rule
        recalls = {}
        rules_set = sorted(list(set(results.rule_true)))
        confusion_matrix: dict = {}
        for rule in rules_set:
            #if rule == "None":
            #    continue
            # Indices where the ground truth is the current item
            confusion_matrix[rule] = {}
            for r in rules_set:
                if r != rule:
                    confusion_matrix[rule][r] = 0
                #fi
            #rof
            idxs = [i for i, t in enumerate(results.rule_true) if t == rule]
            correct = sum(1 for i in idxs if results.rule_pred[i] == rule)
            for i in idxs:
                if results.rule_pred[i] != rule and results.rule_pred[i] in confusion_matrix[rule]:
                    confusion_matrix[rule][results.rule_pred[i]] += 1
                #fi
            #rof
            recalls[rule] = 1 - (correct / len(idxs))
        #
        scores_dict = {
            "accuracy": [
                1 if results.rule_true[i] == results.rule_pred[i] else 0
                for i in range(len(results.rule_pred))
            ],
        }
        metrics_dict = {
            "accuracy": accuracy_score(results.rule_true, results.rule_pred),
            "precision-positive": precision_positive,
            "recall-positive": recall_positive,
            "f1-positive": f1_positive,
            "precision-negative": precision_negative,
            "recall-negative": recall_negative,
            "f1-negative": f1_negative,
            "malformed-count": results.malformed_cnt,
        } | recalls

        return metrics_dict, scores_dict

    def _aggregate_pattern_wise_metrics(
        self, metrics_dict: dict, scores_dict: dict
    ) -> Tuple[dict, dict]:
        """
        Aggregate the 1) pattern-wise metrics and 2) scores into two dicts.
        """
        final_metrics_dict = {}
        final_scores_dict = {}
        for pattern, metrics in metrics_dict.items():
            for metric, value in metrics.items():
                final_metrics_dict[f"{pattern}-{metric}"] = value
            # rof
        # rof
        for pattern, scores in scores_dict.items():
            for metric, score_list in scores.items():
                final_scores_dict[f"{pattern}-{metric}"] = score_list
            # rof
        return final_metrics_dict, final_scores_dict

    def _result_keys_for_model_list(self) -> set[str] | None:
        if not self.model_list:
            return None
        keys = set()
        for model_name in self.model_list:
            keys.add(f"{model_name}-da")
            keys.add(f"{model_name}-cot")
        return keys

    def compute_bootstrap_confidence_interval(self, output_suffix: str = ""):
        """
        Compute bootstrap confidence intervals for model scores.
        This function loads the model scores from a JSON file, performs bootstrap sampling,
        and computes the 95% confidence intervals for each model's accuracy scores.
        The results are saved to a new JSON file.
        """
        # load scores
        scores_file = (
            Macros.results_dir
            / f"scores-{self.task}-{self.setup_name}-{self.exp_name}{output_suffix}.json"
        )
        if output_suffix and not scores_file.exists():
            scores_file = (
                Macros.results_dir
                / f"scores-{self.task}-{self.setup_name}-{self.exp_name}.json"
            )
        try:
            models_scores = su.io.load(scores_file)
        except FileNotFoundError:
            logger.error(f"File not found: {scores_file}")
            return

        allowed_keys = self._result_keys_for_model_list()
        if allowed_keys is not None:
            models_scores = {
                k: v for k, v in models_scores.items() if k in allowed_keys
            }

        model_list = list(models_scores.keys())
        bootstrap_results = {}

        for model in model_list:
            bootstrap_results[model] = {}
            n_bootstrap = 1000
            rng = np.random.default_rng(42)  # for reproducibility
            for metric in models_scores[model]:
                if "accuracy" not in metric and "acc" not in metric:
                    continue
                # Perform bootstrap sampling
                boot_means = []
                for _ in range(n_bootstrap):
                    sample = rng.choice(
                        models_scores[model][metric],
                        size=len(models_scores[model][metric]),
                        replace=True,
                    )
                    boot_means.append(np.mean(sample))

                # Compute 2.5th and 97.5th percentiles
                ci_lower = np.percentile(boot_means, 2.5)
                ci_upper = np.percentile(boot_means, 97.5)

                bootstrap_results[model][metric] = {
                    "mean": np.mean(models_scores[model][metric]),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            # breakpoint()
        #
        # dump the results
        su.io.dump(
            Macros.results_dir
            / f"bootstrap-results-{self.task}-{self.setup_name}-{self.exp_name}{output_suffix}.json",
            bootstrap_results,
            su.io.Fmt.jsonNoSort,
        )

    def compute_significance_test(self, output_suffix: str = ""):
        scores_file = (
            Macros.results_dir
            / f"scores-{self.task}-{self.setup_name}-{self.exp_name}{output_suffix}.json"
        )
        if output_suffix and not scores_file.exists():
            scores_file = (
                Macros.results_dir
                / f"scores-{self.task}-{self.setup_name}-{self.exp_name}.json"
            )
        try:
            models_scores = su.io.load(scores_file)
        except FileNotFoundError:
            logger.error(f"File not found: {scores_file}")
            return

        allowed_keys = self._result_keys_for_model_list()
        if allowed_keys is not None:
            models_scores = {
                k: v for k, v in models_scores.items() if k in allowed_keys
            }

        model_list = list(models_scores.keys())
        significance_tests_results = {}
        # get pairs of models
        pairs = list(combinations(model_list, 2))

        for model_A, model_B in pairs:
            significance_tests_results[f"{model_A},{model_B}"] = {}
            for metric in models_scores[model_A]:
                # compute the siginificance test for the metric
                is_sign = is_significantly_different(
                    models_scores[model_A][metric],
                    models_scores[model_B][metric],
                )
                significance_tests_results[f"{model_A},{model_B}"][metric] = is_sign
        #

        # dump the results
        su.io.dump(
            Macros.results_dir
            / f"significance-tests-{self.task}-{self.setup_name}-{self.exp_name}{output_suffix}.json",
            significance_tests_results,
            su.io.Fmt.jsonNoSort,
        )


# ---------------#
# Helper functions
# ----------------#
class PLSProjector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5, scale=False):
        self.n_components = n_components
        self.scale = scale
        self._pls = None

    def fit(self, X, y):
        self._pls = PLSRegression(n_components=self.n_components, scale=self.scale)
        self._pls.fit(X, y)
        return self

    def transform(self, X):
        # return only X-scores (n_samples, n_components)
        return self._pls.transform(X)

    # expose trained PLS if you need back-projection/VIP later
    @property
    def pls_(self):
        return self._pls


def is_significantly_different(
    scores_A: List,
    scores_B: List,
    alpha: float = 0.05,
    n_trial: int = 10000,
    verbose: bool = False,
) -> bool:
    """Determine if the two lists of model performance are significantly
    different from each other by conducting paired bootstrapping test.

    ! Note: `scores_A` and `scores_B` need to be paired; otherwise, the result is not meaningful.

    Args:
        scores_A (List): First list of score.
        scores_B (List): Second list of score.
        alpha (float, optional): threshold for p-value (below which to be significant). Defaults to 0.05.
        n_trial (int, optional): number of bootstrap sampling to conduct. Defaults to 10000.
        verbose (bool, optional): Whether to print some intermediate results. Defaults to False.

    Returns:
        bool: whether scores_A and scores_B are significantly different from each other.
    """
    scores_A = np.array(scores_A)
    scores_B = np.array(scores_B)
    assert len(scores_A) == len(scores_B)

    # Get the inequality direction (or null hypothesis) we want to validate
    # (by calculating the raw average difference).
    # In this context, let's just call it "the ranking".
    scores_A_mean = scores_A.mean()
    scores_B_mean = scores_B.mean()
    delta = scores_B_mean - scores_A_mean

    count = 0
    n_boostrap = len(scores_A)
    for _ in range(n_trial):
        rand_ids = np.random.choice(len(scores_A), size=n_boostrap, replace=True)
        bootstrapped_scores_A = scores_A[rand_ids]
        bootstrapped_scores_B = scores_B[rand_ids]

        # Count how many times that the bootstrapped average *follows* the ranking
        if delta > 0:
            count += bootstrapped_scores_B.mean() > bootstrapped_scores_A.mean()
        else:
            count += bootstrapped_scores_B.mean() < bootstrapped_scores_A.mean()

    # how many times that the randomness (from bootstrap) causes the ranking to be violated.
    p = 1 - count / n_trial
    # if the amount of violation is below the specified threshold,
    # then it's significant difference.
    is_sig_diff = p <= alpha

    if verbose:
        logger.info(f"Score_A avg: {np.round(scores_A_mean, 3)}")
        logger.info(f"Score_B avg: {np.round(scores_B_mean, 3)}")
        logger.info(f"Delta (B - A): {np.round(delta, 3)}")
        logger.info(f"p: {p} (threshold = {alpha})")
        if is_sig_diff:
            logger.info("Significant")
        else:
            logger.info("*Not* Significant")

    return is_sig_diff


def extract_llm_prediction(llm_output: str, tag: str = "ans") -> str:
    """Extract the predicted return code from the LLM output."""
    if tag in llm_output:
        pred_return_code = extract_content_between_tags(llm_output, tag).strip()
    else:
        pred_return_code = ""

    return pred_return_code


def extract_python_code(llm_output: str) -> str:
    """Extract the Python code from the LLM output."""
    if "```python" in llm_output:
        return llm_output.split("```python")[1].split("```")[0]
    else:
        return ""


def execute_python_code(python_code: str) -> int:
    """Execute the Python code and return the result."""
    try:
        # execute the Python code
        with su.TimeUtils.time_limit(3):
            su.io.dump(Macros.tmp_dir / "tmp.py", python_code, su.io.Fmt.txt)
            stdout = su.bash.run(
                f"python {Macros.tmp_dir / 'tmp.py'}", check_returncode=0
            ).stdout.strip()
    except su.TimeoutException:
        logger.warning("Time limit exceeded when executing Python code.")
        return None
    except Exception as e:
        logger.warning(f"Error executing Python code: {e}")
        return None
    return stdout


def std_dev_numpy(arr: List):
    """
    Calculate standard deviation using NumPy's built-in function
    """
    return np.std(arr)


def edit_distance(pred: List[str], truth: List[str]) -> float:
    """
    Calculate the similarity between two strings using Levenshtein distance.
    Returns a value between 0 and 1, where 1 means identical.
    """
    similarity = 1 - Levenshtein.distance(pred, truth) / max(len(pred), len(truth))
    return similarity


def extract_rules_from_model_preds(pred: str) -> List[str]:
    try:
        return ast.literal_eval(pred)
    except Exception as e:
        logger.warning(f"Error extracting rules from model predictions: {e}")
        return []


def replace_outer_ans_tags(text, start_tag="<answer>", end_tag="</answer>"):
    if "</think>" in text:
        text = text.split("</think>")[1]
    if "<answer>" in text:
        return text
    # Replace the first <ans>
    text = re.sub(r"<ans>", start_tag, text, count=1)
    # Replace the last </ans>
    matches = list(re.finditer(r"</ans>", text))
    if matches:
        last_match = matches[-1]
        text = text[: last_match.start()] + end_tag + text[last_match.end() :]
    return text


def exec_to_str(d: dict):
    """Convert a dictionary-like execution trace to a consistent string representation."""
    program_state_str = ""
    for var, value in d["program_state"].items():
        program_state_str += f"{var}:{value} "
    return f"{d['rule']}:{program_state_str.strip()}"


def multiset_jaccard(pred_list: List[Any], true_list: List[Any]) -> float:
    counter_pred = Counter(pred_list)
    counter_true = Counter(true_list)
    intersection_size = sum((counter_pred & counter_true).values())
    union_size = sum((counter_pred | counter_true).values())
    if union_size == 0:
        raise ValueError("Union size is zero, cannot compute Jaccard similarity.")
    return intersection_size / union_size


def statement_overlap(
    pred_exec_trace: List[dict], true_exec_trace: List[dict]
) -> float:
    """
    Compute the similarity between two sequences of dictionaries using
    the Gestalt pattern matching algorithm.
    Each dictionary must have exactly two keys: 'key1' and 'key2'.
    """
    str_seq_a = [exec_to_str(d) for d in pred_exec_trace]
    str_seq_b = [exec_to_str(d) for d in true_exec_trace]

    return multiset_jaccard(str_seq_a, str_seq_b)


def variable_statement_overlap(
    pred_exec_trace: List[dict], true_exec_trace: List[dict]
) -> float:
    """
    Compute the overlap percentage of (variable, statement) pairs between
    the LLM's prediction and ground truth execution traces.
    """
    pred_var_stmt_list = []
    true_var_stmt_list = []

    for pred_exec in pred_exec_trace:
        program_state = pred_exec["program_state"]
        for var, value in program_state.items():
            pred_var_stmt_list.append((pred_exec["rule"], var, value))
    #

    for true_exec in true_exec_trace:
        program_state = true_exec["program_state"]
        for var, value in program_state.items():
            true_var_stmt_list.append((true_exec["rule"], var, value))
    #

    return multiset_jaccard(pred_var_stmt_list, true_var_stmt_list)


## Code to extract the model prediction from the LLM output
def extract_pcp_prediction(model_output: str) -> Tuple[str]:
    """
    Return the model predicted return code and the violated rule.
    """
    pred_return_code = extract_llm_prediction(model_output)
    pred_rule = extract_llm_prediction(model_output, "rule")
    # handle the case where the rule is wrapped by []
    match = re.search(r"\[([^\]]+)\]", pred_rule)
    if match:
        pred_rule = match.group(1).strip()
    return (pred_return_code, pred_rule)


def extract_op_prediction(model_output: str):
    """
    Return the model predicted program outcome.
    """
    pred_program_outcome = extract_llm_prediction(model_output, "answer")
    if not pred_program_outcome:
        pred_program_outcome = extract_boxed_answers(model_output)
    # handle the case where the value is wrapped by []
    match = re.search(r"\[([^\]]+)\]", pred_program_outcome)
    if match:
        pred_program_outcome = match.group(1).strip()
    if "error" in pred_program_outcome:
        return (["error"], {})
    #fi
    if "timeout" in pred_program_outcome:
        return (["timeout"], {})
    #fi
    return parse_op_prediction(f"<answer>{pred_program_outcome}</answer>")


def extract_srp_prediction(model_output: str) -> List[str]:
    """
    Return the model predicted rules.
    """
    pred = extract_llm_prediction(model_output)
    pred_rules_list = parse_srp_prediction(f"<ans>\n{pred}\n</ans>")
    return pred_rules_list


def extract_etp_prediction(model_output: str) -> List[dict]:
    """
    Return the model predicted execution trace.
    """
    pred = extract_llm_prediction(replace_outer_ans_tags(model_output), tag="answer")
    pred_exec_trace = parse_etp_prediction(f"<answer>{pred}</answer>")
    return pred_exec_trace


def is_number(string: str):
    try:
        float(string)
        return True
    except ValueError:
        return False


def average_metrics(metrics_list: Dict[str, List[Any]]):
    metrics = {}
    for k, lst in metrics_list.items():
        if isinstance(lst, list):
            if all(x == lst[0] for x in lst):
                logger.warning(
                    f"Values for {k} for all runs are the same, this might be a bug"
                )
            metrics[k] = np.mean(lst).item()
            metrics[f"{k}-std"] = np.std(lst).item()
        elif isinstance(lst, dict):
            metrics[k] = {}
            for key2, lst2 in lst.items():
                metrics[k][key2] = np.mean(lst2).item()
                metrics[k][f"{key2}-std"] = np.std(lst2).item()
            # rof
        # fi
    return metrics


def merge_metrics(dict_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Merge a list of dictionaries with the same keys into a single dictionary
    where each key maps to a list of values from all dictionaries.

    Args:
        dict_list: List of dictionaries with the same keys

    Returns:
        Dictionary with keys mapping to lists of values
    """
    if not dict_list:
        return {}

    # Method 1: Using defaultdict (recommended)
    merged = defaultdict(list)

    for d in dict_list:
        for key, value in d.items():
            if isinstance(value, dict):
                if key not in merged:
                    merged[key] = {}
                for key2, value2 in value.items():
                    if key2 in merged[key]:
                        merged[key][key2].append(value2)
                    else:
                        merged[key][key2] = [value2]
                    # fi
                # rof
            else:
                if isinstance(value, list):
                    merged[key].extend(value)
                else:
                    merged[key].append(value)

    # Convert back to regular dict if desired
    return dict(merged)


def dataset_code_complexity_metrics_to_df(dataset_metrics: dict) -> pd.DataFrame:
    columns_dict: dict = {
        "ProgramName": "ProgramName",
        "halstead-metrics": [
            HalsteadMetric.Measures.VOCABULARY.value,
            HalsteadMetric.Measures.VOLUME.value, 
            HalsteadMetric.Measures.DIFFICULTY.value,
            HalsteadMetric.Measures.EFFORT.value,
            HalsteadMetric.Measures.DIFFICULTY.value,
        ],
        "cyclomatic-complexity": [
            ExtendedCyclomaticMetric.Measures.MAX_BLOCK_DEPTH.value,
            ExtendedCyclomaticMetric.Measures.MAX_NESTED_LOOP.value,
            ExtendedCyclomaticMetric.Measures.MAX_NESTED_IF.value,
            ExtendedCyclomaticMetric.Measures.CC.value,
        ],
        "dep-degree": [
            DepDegreeMetric.Measures.DEP_DEGREE.value,
            DepDegreeMetric.Measures.DEP_DEGREE_PER_VARIABLE.value,
        ],
        "num-assignments": "NumAssignments",
        "trace-len": "TraceLength",
        "max-taken-loop-depth": "MaxTakenLoop",
        "max-taken-if-depth": "MaxTakenIf",
    }
    data: list = []
    columns = [
        "ProgramName",
        "LOC",
        HalsteadMetric.Measures.VOCABULARY.value,
        HalsteadMetric.Measures.VOLUME.value, 
        HalsteadMetric.Measures.DIFFICULTY.value,
        HalsteadMetric.Measures.EFFORT.value,
        HalsteadMetric.Measures.DIFFICULTY.value,
        ExtendedCyclomaticMetric.Measures.MAX_BLOCK_DEPTH.value,
        ExtendedCyclomaticMetric.Measures.MAX_NESTED_LOOP.value,
        ExtendedCyclomaticMetric.Measures.MAX_NESTED_IF.value,
        ExtendedCyclomaticMetric.Measures.CC.value,
        DepDegreeMetric.Measures.DEP_DEGREE.value,
        DepDegreeMetric.Measures.DEP_DEGREE_PER_VARIABLE.value,
        "NumAssignments",
        "TraceLength",
        "MaxTakenLoop",
        "MaxTakenIf",
    ]
    for key, value in dataset_metrics.items():
        data_row: list = [key]
        data_row.append(value['loc'])
        for column in columns_dict['halstead-metrics']:
            data_row.append(value['halstead-metrics'][column])
        #rof
        for column in columns_dict['cyclomatic-complexity']:
            data_row.append(value['cyclomatic-complexity'][column])
        #rof
        for column in columns_dict['dep-degree']:
            data_row.append(value['dep-degree'][column])
        #rof
        total_var_assignments: int = 0
        for num_var_assign in value['num-assignments'].values():
            total_var_assignments += num_var_assign
        #rof
        data_row.append(total_var_assignments)
        data_row.append(value['trace-len'])
        data_row.append(value['max-taken-loop-depth'])
        data_row.append(value['max-taken-if-depth'])
        data.append(data_row)
    #rof
    return pd.DataFrame(data, columns=columns)
#fed


def collect_metrics_for_op_results(op_results: list[OPResults], per_var: bool = False):
    outputs: list = []
    program_names: list = []
    num_dataset: int = len(op_results)

    for op_result in op_results:
        for (true_ans, pred_ans, true_ans_dict, pred_ans_dict, src_filename) in zip(op_result.true_ans, op_result.pred_ans, op_result.true_ans_dict, op_result.pred_ans_dict, op_result.src_filename):
            all_vars: set = set(true_ans_dict.keys())
            correct_vars: set = set([pred_var for pred_var, pred_value in pred_ans_dict.items() if pred_var in true_ans_dict and true_ans_dict[pred_var] == pred_value])
            if not per_var:
                outputs.append(1 if len(correct_vars) == len(all_vars) else 0)
                program_names.append(src_filename)
            else:
                sorted_keys = sorted(all_vars)
                for var in sorted_keys:
                    outputs.append(1 if var in correct_vars else 0)
                    program_names.append(src_filename)
                #rof
            #fi
        #rof
    #rof
    
    if num_dataset == 1:
        return (outputs, program_names)
    #fi
    new_outputs: list = []
    m: int = len(outputs)
    n: int = m // num_dataset
    
    for output in outputs:
        for i in range(0, n):
            sum_res: int = 0
            for j in range(0, num_dataset):
                sum_res += outputs[i + j * n]
            #rof
            new_outputs.append(1 if round(sum_res / num_dataset, 2) > 0.6 else 0)
        #rof
    #rof
    return (new_outputs, program_names[0 : n])
#fed

def prepare_dataset_and_op_output_dfs(models: list, semantics_types: list, strategies: list, op_traces: list, dataset_file: str, metrics: list[str], mutations: list = None, filter_while_count: int = -1, filter_if_count: int = -1):
    full_model_names: list = []
    X: pd.DataFrame = dataset_code_complexity_metrics_to_df(read_from_json_file(dataset_file))
    Y: pd.DataFrame = pd.DataFrame()
    if filter_while_count != -1:
        X = X[X['MaxNestedLoop'] < filter_while_count]
    #fi
    if filter_if_count != -1:
        X = X[X['MaxNestedIf'] < filter_if_count]
    #fi
    for model, semantics_type, strategy, op_trace, mutation in zip(models, semantics_types, strategies, op_traces, mutations):
        [outputs, program_names] = collect_metrics_for_op_results(op_trace, False)
        df_column_names: list = list(X['ProgramName'])
        index_map = {val: i for i, val in enumerate(program_names)}
        program_names_sorted = [df_column_names[i] for i in range(len(df_column_names))]
        outputs_sorted = [outputs[index_map[val]] for val in df_column_names]
        Y[f"{model}-{strategy}-{semantics_type}-{mutation}"] = outputs_sorted
    #rof
    X = X.drop(columns=[col for col in X.columns if col not in metrics], axis=1)

    # Drop models that have only 1 class
    columns_to_drop = []
    for col in full_model_names:
        ones_count = (Y[col] == 1).sum()
        zeros_count = (Y[col] == 0).sum()

        if ones_count < 10 or zeros_count < 10:
            columns_to_drop.append(col)
        #fi
    #rof
    Y = Y.drop(columns=columns_to_drop, axis=1)
    print(len(Y))
    return X, Y
#fed

def gen_op_regression_coefficients(models: list, semantics_types: list, strategies: list, op_traces: list, dataset_file: str, metrics: list[str], mutations: list = None, filter_while_count: int = -1, filter_if_count: int = -1):
    [X, Y] = prepare_dataset_and_op_output_dfs(models, semantics_types, strategies, op_traces, dataset_file, metrics, mutations, filter_while_count, filter_if_count)
    full_model_names = list(Y.columns)

    def backproject_logit_from_pls(pipe: Pipeline, X_final: pd.DataFrame):
        scaler = pipe.named_steps["scaler"]
        pls    = pipe.named_steps["pls"]
        pls = pls._pls
        clf    = pipe.named_steps["clf"]

        # PLS matrices
        W = pls.x_weights_            # (p, a)
        P = pls.x_loadings_           # (p, a)
        # W* = W (P^T W)^{-1}
        Wstar = W @ np.linalg.inv(P.T @ W)     # (p, a)

        # Logistic coef on components (gamma)
        if clf.coef_.ndim == 2:      # binary → shape (1, a)
            gamma = clf.coef_.ravel()
            intercept = clf.intercept_.ravel()[0]
        else:
            gamma = clf.coef_        # multinomial: handle per class

        # Coefs on standardized original features, then on original units
        beta_std  = Wstar @ gamma                    # (p,)
        sigma     = pd.Series(scaler.scale_, index=X_final.columns).astype(float)
        beta_orig = pd.Series(beta_std, index=X_final.columns) / sigma

        # Intercept in original space
        mu = pd.Series(scaler.mean_, index=X_final.columns).astype(float)
        intercept_orig = float(intercept - (beta_std * (mu / sigma)).sum())

        # Helpful effect sizes
        iqr = X_final.quantile(0.75) - X_final.quantile(0.25)
        or_per_iqr = np.exp(beta_orig * iqr)

        coef_table = pd.DataFrame({
            "beta_orig": beta_orig,
            "OR_per_IQR": or_per_iqr,
            "beta_per_SD": beta_std
        }).sort_values("OR_per_IQR", ascending=False)
        return coef_table, intercept_orig
    #fed

    regression_details: dict = {}
    for model in Y.columns:
        model_regression_details: dict = {}
        # Z-score normalization + PLS + Logistic Regression
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSProjector(n_components=1, scale=False)),
            ("clf", LogisticRegressionCV(
                solver="saga",
                penalty="elasticnet",
                l1_ratios=[0.1, 0.5, 0.9],
                Cs=np.logspace(-3, 1, 12),
                cv=cv,
                scoring="neg_log_loss",
                class_weight="balanced",
                max_iter=10000,
                n_jobs=-1,
                refit=True
            ))
        ])
        y = Y[model]
        pipe.fit(X, y)
        
        # Evaluate on holdout
        proba = pipe.predict_proba(X)[:, 1]
        [coef_table, intercept_orig] = backproject_logit_from_pls(pipe, X)
        pi = y.mean()                      # prevalence
        brier = brier_score_loss(y, proba) # model's probabilities
        baseline = pi*(1-pi)
        BSS = 1 - brier/baseline
        model_regression_details['coef_table'] = coef_table
        model_regression_details['intercept_orig'] = intercept_orig
        model_regression_details['performance'] = {
            "AUC": roc_auc_score(y, proba),
            "LogLoss": log_loss(y, proba),
            "Brier": brier,
            "pi": pi,
            "Baseline": baseline,
            "BSS": BSS,
        }
        regression_details[model] = model_regression_details
    #rof
    return X, regression_details
#fed
