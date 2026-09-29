import seutil as su
from statistics import mean
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
from dataclasses import asdict
from llm_interpreter.macros import Macros
from llm_interpreter.language import Language, IMP
from llm_interpreter.compiler_runners import KFramework, KResult
from llm_interpreter.utils import write_to_tmp, num_tokens_from_string, read_from_json_file
from llm_interpreter.experiments.prompts import etp_imp_sos_prompt
from llm_interpreter.language.metrics import HalsteadMetric, ExtendedCyclomaticMetric, DepDegreeMetric


logger = su.log.get_logger(__name__)
_LANGUAGE = None

class DataAnalysis:
    def collect_imp_data_stats(self):
        """
        Collect the statistics of the IMP dataset:
        # programs, loc, # tokens, len traces, rules
        """
        imp_data_dir = Macros.data_dir / "imp/valid_imp_programs/human_written/"
        stats = {
            "num-programs": 0,
            "avg-loc": 0,
            "avg-tokens": 0,
            "avg-trace-length": 0,
            "max-trace-length": 0,
            "min-trace-length": 0,
            "avg-rules-per-state": 0,
            "max-rules-per-state": 0,
            "min-rules-per-state": 0,
        }

        loc, tokens, trace_length, rules = [], [], [], []

        # prepare Kframework
        imp_lang1 = IMP("IMP1_Unmutated", Language.SEMANTICS_TYPE.K)
        imp_lang1.build_visitors()
        write_to_tmp(imp_lang1.get_semantics(), "IMP1_Unmutated.k")
        k_framework = KFramework()
        k_framework.compile_k_specification(
            k_file=f"{Macros.tmp_dir}/IMP1_Unmutated.k",
            output_dir=f"{Macros.tmp_dir}/IMP1",
        )

        for file_name in os.listdir(imp_data_dir):
            if file_name.endswith(".imp"):
                stats["num-programs"] += 1
                file_path = imp_data_dir / file_name
                imp_program = su.io.load(file_path, fmt=su.io.Fmt.txt)
                lines = imp_program.splitlines()
                loc.append(len(lines))
                # tokens
                tokens.append(num_tokens_from_string(imp_program))
                # execute
                output: str = k_framework.run_program(program_file=file_path)
                result: KResult = KFramework.parse_k_framework_output(
                    output, imp_lang1, Language.SEMANTICS_TYPE.SOS
                )
                exec_trace = result.get_execution_trace()
                trace_length.append(len(exec_trace))
                rules.extend([len(exec.rule) for exec in exec_trace])
        #
        # calculate averages
        stats["avg-loc"] = mean(loc)
        stats["avg-tokens"] = mean(tokens)
        stats["avg-trace-length"] = mean(trace_length)
        stats["max-trace-length"] = max(trace_length)
        stats["min-trace-length"] = min(trace_length)
        stats["avg-rules-per-state"] = mean(rules)
        stats["max-rules-per-state"] = max(rules)
        stats["min-rules-per-state"] = min(rules)

        # save stats
        su.io.dump(
            Macros.results_dir / "imp-dataset-stats.json",
            stats,
            fmt=su.io.Fmt.jsonPretty,
        )
        logger.info(f"IMP Dataset Stats: {stats}")

    def collect_imp_mutation_rate(self):
        data_file = Macros.data_dir / "dataset" / "dataset-op-mk-IMP-SOS.jsonl"
        stats = {}
        mutated_dataset = su.io.load(data_file)
        mutation_rate_list = defaultdict(list)
        mutation_rate = defaultdict(float)
        mutated_programs = defaultdict(int)

        for dt in tqdm(mutated_dataset, total=len(mutated_dataset)):
            if dt["mutated"]:
                mutation_rate_list[dt["mutation-pattern"].replace("_", "-")].append(1)
                mutated_programs[dt["mutation-pattern"].replace("_", "-")] += 1
            else:
                mutation_rate_list[dt["mutation-pattern"].replace("_", "-")].append(0)
        # calculate averages
        for key in mutation_rate_list.keys():
            mutation_rate[key] = mean(mutation_rate_list[key]) * 100
        #
        stats = {
            "mutation-rate": mutation_rate,
            "num-mutated-programs": mutated_programs,
        }
        su.io.dump(
            Macros.results_dir / "mk-IMP-SOS-stats.json",
            stats,
            fmt=su.io.Fmt.jsonPretty,
        )

    def collect_etp_stats(self, data_dir: str):
        stats = {}
        imp_lang1 = IMP("IMP1_Unmutated", Language.SEMANTICS_TYPE.K)
        imp_lang1.build_visitors()
        write_to_tmp(imp_lang1.get_semantics(), "IMP1_Unmutated.k")
        k_framework = KFramework()
        k_framework.compile_k_specification(
            k_file=f"{Macros.tmp_dir}/IMP1_Unmutated.k",
            output_dir=f"{Macros.tmp_dir}/IMP1",
        )

        trace_length, rules = [], []
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".imp"):
                file_path = data_dir / file_name
                output: str = k_framework.run_program(program_file=file_path)
                result: KResult = KFramework.parse_k_framework_output(
                    output, imp_lang1, Language.SEMANTICS_TYPE.SOS
                )
                exec_trace = result.get_execution_trace()
                trace_length.append(len(exec_trace))
                rules.extend([len(exec.rule) for exec in exec_trace])
        #

        stats["avg-trace-length"] = mean(trace_length)
        stats["max-trace-length"] = max(trace_length)
        stats["min-trace-length"] = min(trace_length)
        stats["avg-rules-per-state"] = mean(rules)
        stats["max-rules-per-state"] = max(rules)
        stats["min-rules-per-state"] = min(rules)

        # save stats
        su.io.dump(
            Macros.results_dir / "etp-dataset-stats.json",
            stats,
            fmt=su.io.Fmt.jsonPretty,
        )
        logger.info(f"ETP Dataset Stats: {stats}")

    def collect_etp_prompt_tokens(self):
        """
        Collect the statistics of the ETP dataset:
        # prompt tokens, # result tokens
        """
        # load dataset
        etp_dataset = su.io.load(
            Macros.data_dir / "dataset" / "dataset-etp-uk-IMP-SOS.jsonl"
        )
        prompt_tokens, result_tokens = [], []
        for dt in etp_dataset:
            prompt = etp_imp_sos_prompt.format(
                language=dt["language"],
                syntax=dt["syntax"],
                semantics=dt["semantics"],
                program=dt["program"],
            )
            ans = dt["etp-ans"]
            prompt_tokens.append(num_tokens_from_string(prompt))
            result_tokens.append(num_tokens_from_string(ans))
        #
        stats = {
            "avg-prompt-tokens": mean(prompt_tokens),
            "max-prompt-tokens": max(prompt_tokens),
            "min-prompt-tokens": min(prompt_tokens),
            "avg-result-tokens": mean(result_tokens),
            "max-result-tokens": max(result_tokens),
            "min-result-tokens": min(result_tokens),
        }
        print(stats)

    def collect_dataset_stats(self):
        """
        Collect the statistics of the dataset:
        IMP: # programs, loc, # tokens, # variables.
        pcp: # pos programs, # neg programs;
        srp: # selecte stmts, # rules
        etp: len. execution traces
        """
        stats_counter = defaultdict(list)
        # load dataset
        srp_dataset = su.io.load(
            Macros.data_dir / "dataset" / "dataset-srp-uk-IMP-SOS.jsonl"
        )
        etp_dataset = su.io.load(
            Macros.data_dir / "dataset" / "dataset-etp-uk-IMP-SOS.jsonl"
        )
        for edt, sdt in zip(etp_dataset, srp_dataset):
            loc = len(edt["mutated-program"].split("\n"))
            tokens = num_tokens_from_string(edt["mutated-program"])
            vars_nums = len(edt["final-state"])
            trace_length = len(edt["exec-trace"])
            selected_stmts = len(sdt["sampled-statements"])
            rules_per_stmt = [len(stmt["rules"]) for stmt in sdt["sampled-statements"]]
            # add to stats
            stats_counter["imp-loc"].append(loc)
            stats_counter["imp-tokens"].append(tokens)
            stats_counter["imp-vars"].append(vars_nums)
            stats_counter["etp-trace-length"].append(trace_length)
            stats_counter["srp-selected-stmts"].append(selected_stmts)
            stats_counter["srp-rules-per-stmt"].extend(rules_per_stmt)
            #

        # calculate averages
        stats = {
            "imp-num-programs": len(stats_counter["imp-loc"]),
            "imp-mean-loc": mean(stats_counter["imp-loc"]),
            "imp-max-loc": max(stats_counter["imp-loc"]),
            "imp-min-loc": min(stats_counter["imp-loc"]),
            "imp-mean-tokens": mean(stats_counter["imp-tokens"]),
            "imp-max-tokens": max(stats_counter["imp-tokens"]),
            "imp-min-tokens": min(stats_counter["imp-tokens"]),
            "imp-mean-vars": mean(stats_counter["imp-vars"]),
            "imp-max-vars": max(stats_counter["imp-vars"]),
            "imp-min-vars": min(stats_counter["imp-vars"]),
            "etp-mean-trace-length": mean(stats_counter["etp-trace-length"]),
            "etp-max-trace-length": max(stats_counter["etp-trace-length"]),
            "etp-min-trace-length": min(stats_counter["etp-trace-length"]),
            "srp-mean-selected-stmts": mean(stats_counter["srp-selected-stmts"]),
            "srp-max-selected-stmts": max(stats_counter["srp-selected-stmts"]),
            "srp-min-selected-stmts": min(stats_counter["srp-selected-stmts"]),
            "srp-mean-rules-per-stmt": mean(stats_counter["srp-rules-per-stmt"]),
            "srp-max-rules-per-stmt": max(stats_counter["srp-rules-per-stmt"]),
            "srp-min-rules-per-stmt": min(stats_counter["srp-rules-per-stmt"]),
        }
        su.io.dump(
            Macros.results_dir / "benchmark-stats.json",
            stats,
            fmt=su.io.Fmt.jsonNoSort,
        )



    # def collect_dataset_code_complexity_metrics(self, dataset_file: str) -> dict:
    #     dataset = su.io.load(dataset_file)
    #     language: Language = IMP("IMP1_Unmutated", Language.SEMANTICS_TYPE.SOS)
    #     dataset_metrics: dict = {}

    #     for dt in tqdm(dataset, total=len(dataset)):
    #         dt_metrics: dict = {}
    #         # get program and exec trace
    #         program = dt["program"]
    #         exec_trace = dt["exec-trace"]
    #         final_state = dt["final-state"]
    #         # get static metrics from program
    #         halstead_metrics = language.get_halstead_metrics(program)
    #         cyclomatic_complexity = language.get_extended_cyclomatic_metrics(program)
    #         dep_degree = language.get_depdegree_metrics(program)
    #         loc = language.get_loc_metrics(program)
    #         dt_metrics["program"] = program
    #         dt_metrics["halstead-metrics"] = halstead_metrics
    #         dt_metrics["cyclomatic-complexity"] = cyclomatic_complexity
    #         dt_metrics["dep-degree"] = dep_degree
    #         dt_metrics["loc"] = loc
    #         dt_metrics["trace-len"] = len(exec_trace)
    #         # get dynamic metrics from exec trace
    #         decl_vars = set(final_state.keys())
    #         num_updates_per_variable = defaultdict(int)
    #         var_initial_values = defaultdict(int)
    #         for elem in exec_trace:
    #             exec_state: dict = elem['state']
    #             for var, var_value in exec_state.items():
    #                 num_updates_per_variable[var]  = num_updates_per_variable[var] if var_value == var_initial_values[var] else (num_updates_per_variable[var] + 1)
    #                 var_initial_values[var] = var_value
    #             #rof
    #         #rof
    #         dt_metrics["num-assignments"] = num_updates_per_variable
    #         dataset_metrics[dt["src-filename"]] = dt_metrics
    #     #
    #     return dataset_metrics


    # #fed


    def filter_dataset_top_n_complex_programs(self, dataset_file: Path, dataset_complexity_metrics: Path, top_n: int, metrics: list)->dict:
        dataset_metrics: dict = read_from_json_file(str(dataset_complexity_metrics))
        dataset = su.io.load(dataset_file)
        filtered_dataset: list = []
        filtered_metrics: dict = {}
        dataset_metric_values: list = []
        for key, value in dataset_metrics.items():
            metric_values: list = [key]
            for metric in metrics:
                if metric == 'num-assignments':
                    metric_values.append(sum(value['num-assignments'].values()))
                else:
                    if metric in value:
                        metric_values.append(value[metric])
                    else:
                        for key2, value2 in value.items():
                            if isinstance(value2, dict):
                                if metric in value2:
                                    metric_values.append(value2[metric])
                                #fi
                            #fi
                        #fi
                    #fi
                #fi
            #rof
            dataset_metric_values.append(metric_values)
        #rof
        df_columns: list = ['program'] + metrics
        df = pd.DataFrame(dataset_metric_values,columns=df_columns)
        complex_programs_set: set = set()
        for metric in metrics:
            df_sorted = df.sort_values(by=metric, ascending=False)
            complex_programs_set = complex_programs_set | set(df_sorted.head(top_n)['program'])
        #rof
        for data in dataset:
            if data['src-filename'] in complex_programs_set:
                filtered_dataset.append(data)
            #fi
        #rof
        filtered_metrics: dict =  {program: dataset_metrics[program] for program in complex_programs_set}
        print(len(filtered_dataset))
        return filtered_dataset, filtered_metrics
    #fed


def _get_language():
    global _LANGUAGE
    if _LANGUAGE is None:
        # create once per process
        _LANGUAGE = IMP("IMP1_Unmutated", Language.SEMANTICS_TYPE.SOS)
    return _LANGUAGE
#fed

def _compute_one(dt):
    lang = _get_language()

    program     = dt["program"]
    exec_trace  = dt["exec-trace"]
    final_state = dt["final-state"]

    # static metrics
    halstead   = lang.get_halstead_metrics(program)
    cyclo      = lang.get_extended_cyclomatic_metrics(program)
    dep_deg    = lang.get_depdegree_metrics(program)
    loc        = lang.get_loc_metrics(program)

    # dynamic metrics (assign updates per variable)
    num_updates = defaultdict(int)
    last_val    = defaultdict(lambda: None)
    for elem in exec_trace:
        exec_state = elem["state"]
        for var, val in exec_state.items():
            if last_val[var] is None:
                last_val[var] = val
            elif val != last_val[var]:
                num_updates[var] += 1
                last_val[var] = val

    dt_metrics = {
        "program": program,
        "halstead-metrics": halstead,
        "cyclomatic-complexity": cyclo,
        "dep-degree": dep_deg,
        "loc": loc,
        "max-taken-loop-depth": dt['max-loop-depth'] if dt['max-loop-depth'] >= 0 else 0,
        "max-taken-if-depth": dt['max-if-depth'] if dt['max-if-depth'] >= 0 else 0,
        "trace-len": len(exec_trace),
        "num-assignments": dict(num_updates),  # make JSON/pickle friendly
    }
    return (dt["src-filename"], dt_metrics)

def collect_dataset_code_complexity_metrics(dataset_file):
    """
    Collect the code complexity metrics of the dataset:
    # Halstead Metrics
    # Cyclomatic Complexity
    # DepDegree
    # DepDegree per variable
    # Number of updates per variable
    # Number of nested loops
    # Number of nested if statements

    Always collect the metrics for the unmutated program.
    The dataset file should be a jsonl file and should contain a "program" field and
    an "exec-trace" field and a "final-state" field.
    """

    dataset = su.io.load(dataset_file)

    results = process_map(
        _compute_one,
        dataset,
        max_workers=os.cpu_count(),   # or a smaller number if memory is tight
        chunksize=8,                  # bigger chunks = fewer IPC hops
        desc="Computing metrics",
    )
    return {k: v for (k, v) in results}
#fed
