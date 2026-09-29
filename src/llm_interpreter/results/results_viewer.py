import seutil as su
import os
import glob
from typing import Dict, List, Tuple
from pathlib import Path
import pprint
import random

from llm_interpreter.macros import Macros
from llm_interpreter.experiments.args import ExperimentArgs
from llm_interpreter.experiments.prompt_maker import (
    make_srp_prompt,
    make_pep_prompt,
    make_op_prompt,
    make_etp_prompt,
)
from llm_interpreter.results.results_analysis import (
    extract_pcp_prediction,
    extract_op_prediction,
    extract_etp_prediction,
    extract_srp_prediction,
    ResultsAnalyzer,
)

logger = su.log.get_logger(__name__)


class ResultsViewer:
    def __init__(self, exp_args: ExperimentArgs):
        self.args = exp_args
        self.result_analyzer = ResultsAnalyzer(
            task=exp_args.task,
            setup_name=exp_args.setup_name,
            exp_name=exp_args.expr_name,
            model_list=[],
        )

    def find_failure_mode(
        self, model_list: List[str], task: str, sample_size: int = 10
    ):
        """
        Find the failure examples for the models in the model_list.
        Sample failure examples for each semantics.
        """
        self.args.task = task
        self.result_analyzer.task = task
        self.result_analyzer.model_list = model_list
        model_results_list: dict = {}

        dataset = self.load_dataset()
        for model_name in model_list:
            self.args.model_name = model_name
            results_path: str = f"{Macros.results_dir}/{self.args.task}/{self.args.setup_name}/{self.args.expr_name}/{self.args.model_name}-{self.args.prompt_strategy}"
            files: list = glob.glob(f"{results_path}/*.jsonl")
            for jsonl_file in files:
                model_results_list[jsonl_file] = self.load_results_from_path(
                    Path(jsonl_file)
                )
            # rof
        failure_examples = []
        for i in range(len(dataset)):
            dt = dataset[i]
            # add the example to failure_examples if the all the model predictions are incorrect
            for jsonl_file, result in model_results_list.items():
                if not self.is_correct(dt, result[i], task):
                    failure_examples.append(
                        (
                            dt,
                            [(result[i], model_name, jsonl_file)],
                        )
                    )
        # log the failure examples
        logger.info(f"Found {len(failure_examples)} failure examples.")
        # if the setup is mk, classifiy based on two mutation patterns
        if self.args.setup_name == "mk":
            ks = [
                (dt, res)
                for (dt, res) in failure_examples
                if dt["mutation-pattern"]
                in ["KeyWordSwap", "addSub_mulDiv_negateRelation"]
                and "(+ " not in dt["mutated-program"]
            ]
            ko = [
                (dt, res)
                for (dt, res) in failure_examples
                if dt["mutation-pattern"] in ["KeyWordObf", "unseen"]
                and "(+ " not in dt["mutated-program"]
            ]
            # sample from each category
            if sample_size > 0:
                failure_examples = random.sample(ks, sample_size) + random.sample(
                    ko, sample_size
                )
            else:
                failure_examples = ks + ko
        else:
            # sample failure examples
            if sample_size > 0:
                failure_examples = random.sample(failure_examples, sample_size)
        # write results to file
        text = ""
        for dt, results in failure_examples:
            text += self.present_multi_model_result(dt, results)
            text += "\n"
        models = "-".join(model_list)
        su.io.dump(
            Macros.results_dir
            / f"examples/failure-{self.args.task}-{self.args.setup_name}-{models}.md",
            text,
            su.io.Fmt.txt,
        )

    def write_results_to_file(self):
        model_results = self.load_results()
        dataset = self.load_dataset()
        entire_text_result = ""
        for result, dt in zip(model_results, dataset):
            result_text = self.present_result(dt, result)
            entire_text_result += result_text
            entire_text_result += "\n"
        # write to file
        with open(
            Macros.results_dir / f"examples/results-{str(self.args)}.md", "w"
        ) as f:
            f.write(entire_text_result)
            f.write("\n")

    def present_result(self, dt: dict, result: dict) -> str:
        """
        Present the result in a readable format.
        """
        text = ""
        text += f"# Data id: {dt['id']}\n"
        prompt = self.prepare_prompt(dt)[-1]["content"]
        text += f"# Prompt:\n{prompt}\n"
        # model prediction
        text += f"# Model prediction:\n{result['model-prediction']}\n"
        # processed prediction
        pred = result["model-prediction"]
        if self.args.task == "pcp":
            text += f"# Processed prediction:\n{extract_pcp_prediction(pred)}\n"
        elif self.args.task == "op":
            text += f"# Processed prediction:\n{extract_op_prediction(pred)}\n"
        elif self.args.task == "srp":
            text += f"# Processed prediction:\n{extract_srp_prediction(pred)}\n"
        elif self.args.task == "etp":
            text += f"# Processed prediction:\n{extract_etp_prediction(pred)}\n"
        else:
            raise NotImplementedError(
                f"Prompt for {str(self.args)} is not implemented yet."
            )
        # ground truth
        text += f"# Ground truth:\n{self.get_ground_truth(dt)}\n"
        return text

    def present_multi_model_result(
        self, dt: dict, model_results: List[Tuple[dict, str]]
    ) -> str:
        """
        Present the result in a readable format.
        """
        text = ""
        text += f"# Data id: {dt['id']}\n"
        text += f"## Task : {self.args.task}\n"
        # prompt = self.prepare_prompt(dt)[-1]["content"]
        text += f"## Program:\n{dt['program']}\n"
        for result, model_name, jsonl_file in model_results:
            text += f"## Model: {model_name}\n"
            text += f"## File: {jsonl_file}\n"
            text += f"## Model prediction:\n{result['model-prediction']}\n\n"
            pred = result["model-prediction"]
            if self.args.task == "pcp":
                text += f"## Processed prediction:\n{extract_pcp_prediction(pred)}\n"
            elif self.args.task == "op":
                text += f"## Processed prediction:\n{extract_op_prediction(pred)}\n"
            elif self.args.task == "srp":
                text += f"## Processed prediction:\n{extract_srp_prediction(pred)}\n"
            elif self.args.task == "etp":
                text += f"## Processed prediction:\n{pprint.pformat(extract_etp_prediction(pred), indent=4)}\n"
            else:
                raise NotImplementedError(
                    f"Prompt for {str(self.args)} is not implemented yet."
                )
        text += "\n"
        # ground truth
        text += f"## Ground truth:\n{self.get_ground_truth(dt)}\n"
        text += "\n"
        return text

    def get_ground_truth(self, dt: dict) -> str:
        """
        Get the ground truth from the data.
        """
        if self.args.task == "pcp":
            return dt["ans"] + dt["semantic-error-rule"]
        elif self.args.task == "op":
            return dt["final-state"]
        elif self.args.task == "srp":
            gt_rules_list = []
            for d in dt["sampled-statements"]:
                rule_list = [r.replace("Rule", "").strip() for r in d["rules"]]
                gt_rules_list.append(rule_list)
            return gt_rules_list
        elif self.args.task == "etp":
            return pprint.pformat(extract_etp_prediction(dt["etp-ans"]), indent=4)
        else:
            raise NotImplementedError(
                f"Prompt for {str(self.args)} is not implemented yet."
            )

    # Helper functions

    def load_dataset(self):
        data_file_path = (
            Macros.data_dir
            / "dataset"
            / f"dataset-{self.args.task}-{self.args.setup_name}-{self.args.expr_name}.jsonl"
        )
        dataset = su.io.load(data_file_path)
        return dataset

    def prepare_prompt(self, dt: Dict) -> List[dict]:
        """
        Return the prompt string given the one data.
        """
        chat = []
        if self.args.task == "pcp":
            chat = make_pep_prompt(self.args, dt)
        elif self.args.task == "op":
            chat = make_op_prompt(self.args, dt)
        elif self.args.task == "srp":
            chat = make_srp_prompt(self.args, dt)
        elif self.args.task == "etp":
            chat = make_etp_prompt(self.args, dt)
        else:
            raise NotImplementedError(
                f"Prompt for {str(self.args)} is not implemented yet."
            )

        return chat

    def is_correct(self, dt: Dict, result: Dict, task: str) -> bool:
        """
        Check if the model prediction is correct.
        """
        if task == "pcp":
            pred_return_code, pred_rule = extract_pcp_prediction(
                result["model-prediction"]
            )
            return pred_return_code == dt["ans"]
        elif task == "op":
            pred_return_code = extract_op_prediction(result["model-prediction"])
            sorted_keys = sorted(result["final-state"])
            true_list: list = []
            for key in sorted_keys:
                true_list.append(result["final-state"][key])
            # rof
            return pred_return_code == true_list
        elif task == "srp":
            srp_result = self.result_analyzer._analyze_srp_uk([result], model_name="")
            xmatch_scores = []
            for pred_rules, true_rules in zip(
                srp_result.pred_rules, srp_result.true_rules
            ):
                if pred_rules == true_rules:
                    xmatch_scores.append(1)
                else:
                    xmatch_scores.append(0)
            return all(xmatch_scores) == 1
        elif task == "etp":
            etp_result = self.result_analyzer._analyze_etp_uk([result], model_name="")
            xmatch_scores = []
            for pred_exe, true_exe, raw_exe in zip(
                etp_result.pred_exec_trace,
                etp_result.true_exec_trace,
                etp_result.raw_exec_trace,
            ):
                if pred_exe == true_exe:
                    xmatch_scores.append(1)
                else:
                    xmatch_scores.append(0)
            return all(xmatch_scores) == 1
        else:
            raise NotImplementedError(f"Task {task} is not implemented yet.")

    def load_results(self):
        results_file_path = Macros.results_dir / f"results-{str(self.args)}.jsonl"
        if os.path.exists(results_file_path):
            results = su.io.load(results_file_path)
        else:
            raise FileNotFoundError(f"Results file {results_file_path} not found.")
        return results

    def load_results_from_path(self, results_file_path):
        if os.path.exists(results_file_path):
            results = su.io.load(results_file_path)
        else:
            raise FileNotFoundError(f"Results file {results_file_path} not found.")
        return results

    # fed
