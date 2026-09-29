import os
import sys
import seutil as su
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from llm_interpreter.macros import Macros
from llm_interpreter.utils import subcommand, Category
from llm_interpreter.paper_latex.table import Table
from llm_interpreter.paper_latex.figure import (
    gen_pcp_radar_chart,
    load_json_data,
    make_bars,
    gen_srp_radar_chart,
    gen_etp_radar_chart,
    plot_code_complexity_metric,
    plot_pcp_error_confusion_matrix,
    plot_op_regression_heatmap,
    plot_op_violin_plots,
    plot_dataset_violin_plots,
    plot_op_beta_sd_dendrogram,
    plot_metric_against_metric,
    plot_pcp_dendrogram,
    plot_notation_comprehension_rule_distribution,
    plot_notation_comprehension_rule_distribution_all,
    plot_notation_comprehension_confusion_matrix,
)

from llm_interpreter.paper_latex.intp_functions import *  # noqa: F403
from llm_interpreter.results.results_analysis import ResultsAnalyzer
from llm_interpreter.language import Language, IMP


@subcommand(category=Category.LATEX)
def gen_imp_k_rules_table(paper: str = "lmpl26"):
    """Generate tables/imp_k_rules_table.tex"""
    Table(paper=paper).make_imp_k_rules_table()


# fed

@subcommand(category=Category.LATEX)
def gen_imp_dataset_table():
    table = Table()
    table.make_dataset_numbers(dataset="imp-dataset")
    table.make_dataset_table(ds_name="imp-dataset")


@subcommand(category=Category.LATEX)
def gen_mk_imp_sos_dataset_table():
    table = Table()
    table.make_dataset_numbers(dataset="mk-IMP-SOS")
    table.make_dataset_table(ds_name="mk-IMP-SOS")


@subcommand(category=Category.LATEX)
def gen_op_imp_sos_results_table():
    table = Table()
    table.make_numbers(task="op", exp_name="IMP-SOS")
    table.make_result_table(task="op", exp_name="IMP-SOS")


@subcommand(category=Category.LATEX)
def gen_pcp_imp_sos_results_table(paper: str = "arxiv"):
    table = Table(paper=paper)
    table.make_numbers(task="pcp", exp_name="IMP-SOS")
    table.make_result_table(task="pcp", exp_name="IMP-SOS")


@subcommand(category=Category.LATEX)
def gen_srp_imp_sos_results_table():
    table = Table()
    table.make_numbers(task="srp", exp_name="IMP-SOS")
    table.make_result_table(task="srp", exp_name="IMP-SOS")


@subcommand(category=Category.LATEX)
def gen_etp_imp_sos_results_table():
    table = Table()
    table.make_numbers(task="etp", exp_name="IMP-SOS")
    table.make_result_table(task="etp", exp_name="IMP-SOS")


@subcommand(category=Category.LATEX)
def gen_pcp_op_imp_sos_results_table(paper: str = "arxiv"):
    table = Table(paper=paper)
    table.make_numbers(task="pcp", exp_name="IMP-SOS")
    table.make_numbers(task="op", exp_name="IMP-SOS")
    table.make_two_results_table(exp_name="IMP-SOS", tasks=["pcp", "op"])


@subcommand(category=Category.LATEX)
def gen_pcp_imp_sos_k_results_table():
    table = Table()
    table.make_numbers(task="pcp", exp_name="IMP-SOS")
    table.make_numbers(task="pcp", exp_name="IMP-K")
    table.make_two_results_table_same_task(task="pcp", exp_names=["IMP-K", "IMP-SOS"], metrics='accuracy', headers=['Kos-semantics', 'Sos'])


@subcommand(category=Category.LATEX)
def gen_pcp_qwen_coder_results_table(paper: str = "lmpl26"):
    """
    PCP accuracy table for Qwen2.5-Coder (3B/7B/14B/32B) and the random
    guesser baseline. UK: single accuracy column; MK: KeywordSwap + KeywordObf.
    """
    from llm_interpreter.experiments.prompt_maker import PROMPT_STRATEGY
    from llm_interpreter.experiments.runners.random_guesser_experiment import (
        RandomGuesserRunner,
    )

    random_guesser_name = RandomGuesserRunner.MODEL_NAME
    random_guesser_row = f"{random_guesser_name}-da"
    for setup_name, exp_name in [
        ("uk", "IMP-K"),
        ("uk", "IMP-SOS"),
        ("mk", "IMP-K"),
        ("mk", "IMP-SOS"),
    ]:
        ResultsAnalyzer(
            task="pcp",
            setup_name=setup_name,
            exp_name=exp_name,
            model_list=[random_guesser_name],
        ).analyze_results(
            prompt_strategies=[PROMPT_STRATEGY.DA],
            merge_existing=True,
        )

    table = Table(paper=paper)
    qwen_coder_models = [
        "Qwen-Qwen2.5-Coder-3B-Instruct-da",
        "Qwen-Qwen2.5-Coder-7B-Instruct-da",
        "Qwen-Qwen2.5-Coder-14B-Instruct-da",
        "Qwen-Qwen2.5-Coder-32B-Instruct-da",
        random_guesser_row,
        Table.ROW_SEP,
        "Qwen-Qwen2.5-Coder-3B-Instruct-cot",
        "Qwen-Qwen2.5-Coder-7B-Instruct-cot",
        "Qwen-Qwen2.5-Coder-14B-Instruct-cot",
        "Qwen-Qwen2.5-Coder-32B-Instruct-cot",
    ]
    mk_pcp_accuracy_metrics = ["KeywordSwap-accuracy", "KeywordObf-accuracy"]
    table.make_numbers(task="pcp", exp_name="IMP-SOS")
    table.make_numbers(task="pcp", exp_name="IMP-K")
    table.make_two_results_table_same_task(
        task="pcp",
        exp_names=["IMP-K", "IMP-SOS"],
        metrics="accuracy",
        headers=["Kos-semantics", "Sos"],
        model_list=qwen_coder_models,
        table_suffix="qwen-coder",
        exp_metric_columns={
            "pcp-uk-IMP-SOS": ["accuracy"],
            "pcp-mk-IMP-SOS": mk_pcp_accuracy_metrics,
            "pcp-uk-IMP-K": ["accuracy"],
            "pcp-mk-IMP-K": mk_pcp_accuracy_metrics,
        },
        row_footnote_marks={random_guesser_row: "*"},
        table_notes=[
            (
                "*",
                "The random guesser uses a weighted random selection based on the distribution of the dataset.",
            )
        ],
    )


@subcommand(category=Category.LATEX)
def gen_op_imp_sos_k_results_table():
    table = Table()
    table.make_numbers(task="op", exp_name="IMP-SOS")
    table.make_numbers(task="op", exp_name="IMP-K")
    table.make_two_results_table_same_task(task="op", exp_names=["IMP-K", "IMP-SOS"], metrics='acc', headers=['KTool', 'Sos'])


@subcommand(category=Category.LATEX)
def gen_op_imp_sos_k_var_correct_percentage_results_table():
    table = Table()
    table.make_numbers(task="op", exp_name="IMP-SOS")
    table.make_numbers(task="op", exp_name="IMP-K")
    table.make_two_results_table_same_task(task="op", exp_names=["IMP-K", "IMP-SOS"], metrics='var-acc', headers=['KTool', 'Sos'])


@subcommand(category=Category.LATEX)
def gen_op_imp_sos_k_synthetic_cpp_results_table():
    table = Table()
    table.make_numbers(task="op", exp_name="IMP-SOS", dataset_name="synthetic_cpp")
    table.make_numbers(task="op", exp_name="IMP-K", dataset_name="synthetic_cpp")
    table.make_two_results_table_same_task_reasoning_only(task="op", exp_names=["IMP-K", "IMP-SOS"], metrics='acc', headers=['KTool', 'Sos'], dataset_name="synthetic_cpp")


@subcommand(category=Category.LATEX)
def gen_op_imp_sos_k_var_correct_percentage_synthetic_cpp_results_table():
    table = Table()
    table.make_numbers(task="op", exp_name="IMP-SOS", dataset_name="synthetic_cpp")
    table.make_numbers(task="op", exp_name="IMP-K", dataset_name="synthetic_cpp")
    table.make_two_results_table_same_task_reasoning_only(task="op", exp_names=["IMP-K", "IMP-SOS"], metrics='var-acc', headers=['KTool', 'Sos'], dataset_name="synthetic_cpp")

    
@subcommand(category=Category.LATEX)
def gen_op_imp_sos_k_fuzzer_generated_results_table():
    table = Table()
    table.make_numbers(task="op", exp_name="IMP-SOS", dataset_name="fuzzer_generated")
    table.make_numbers(task="op", exp_name="IMP-K", dataset_name="fuzzer_generated")
    table.make_two_results_table_same_task_reasoning_only(task="op", exp_names=["IMP-K", "IMP-SOS"], metrics='acc', headers=['KTool', 'Sos'], dataset_name="fuzzer_generated")


@subcommand(category=Category.LATEX)
def gen_op_imp_sos_k_var_correct_percentage_fuzzer_generated_results_table():
    table = Table()
    table.make_numbers(task="op", exp_name="IMP-SOS", dataset_name="fuzzer_generated")
    table.make_numbers(task="op", exp_name="IMP-K", dataset_name="fuzzer_generated")
    table.make_two_results_table_same_task_reasoning_only(task="op", exp_names=["IMP-K", "IMP-SOS"], metrics='var-acc', headers=['KTool', 'Sos'], dataset_name="fuzzer_generated")


@subcommand(category=Category.LATEX)
def gen_op_imp_sos_k_all_datasets_results_table():
    table = Table()
    table.make_numbers(task="op", exp_name="IMP-SOS")
    table.make_numbers(task="op", exp_name="IMP-K")
    table.make_numbers(task="op", exp_name="IMP-SOS", dataset_name="synthetic_cpp")
    table.make_numbers(task="op", exp_name="IMP-K", dataset_name="synthetic_cpp")
    table.make_numbers(task="op", exp_name="IMP-SOS", dataset_name="fuzzer_generated")
    table.make_numbers(task="op", exp_name="IMP-K", dataset_name="fuzzer_generated")
    table.make_two_results_table_same_task_op(exp_names=["IMP-K", "IMP-SOS"], metrics='acc', headers=['KTool', 'Sos'], dataset_names=["human_written","synthetic_cpp","fuzzer_generated"])


@subcommand(category=Category.LATEX)
def gen_op_imp_sos_k_var_correct_percentage_all_datasets_results_table():
    table = Table()
    table.make_numbers(task="op", exp_name="IMP-SOS")
    table.make_numbers(task="op", exp_name="IMP-K")
    table.make_numbers(task="op", exp_name="IMP-SOS", dataset_name="synthetic_cpp")
    table.make_numbers(task="op", exp_name="IMP-K", dataset_name="synthetic_cpp")
    table.make_numbers(task="op", exp_name="IMP-SOS", dataset_name="fuzzer_generated")
    table.make_numbers(task="op", exp_name="IMP-K", dataset_name="fuzzer_generated")
    table.make_two_results_table_same_task_op(exp_names=["IMP-K", "IMP-SOS"], metrics='var-acc', headers=['KTool', 'Sos'], dataset_names=["human_written","synthetic_cpp","fuzzer_generated"])


@subcommand(category=Category.LATEX)
def gen_srp_imp_sos_k_results_table():
    table = Table()
    table.make_numbers(task="srp", exp_name="IMP-SOS")
    table.make_numbers(task="srp", exp_name="IMP-K")
    table.make_two_results_table_same_task(task="srp", exp_names=["IMP-K", "IMP-SOS"], metrics='xmatch-accuracy', headers=['KTool', 'Sos'])


@subcommand(category=Category.LATEX)
def gen_etp_imp_sos_k_results_table():
    table = Table()
    table.make_numbers(task="etp", exp_name="IMP-SOS")
    table.make_numbers(task="etp", exp_name="IMP-K")
    table.make_two_results_table_same_task(task="etp", exp_names=["IMP-K", "IMP-SOS"], metrics='xmatch-accuracy')


@subcommand(category=Category.LATEX)
def gen_etp_imp_sos_k_approx_match_results_table():
    table = Table()
    table.make_numbers(task="etp", exp_name="IMP-SOS")
    table.make_numbers(task="etp", exp_name="IMP-K")
    table.make_two_results_table_same_task(task="etp", exp_names=["IMP-K", "IMP-SOS"], metrics='approx-match-accuracy')


@subcommand(category=Category.LATEX)
def gen_etp_imp_sos_k_final_state_match_results_table():
    table = Table()
    table.make_numbers(task="etp", exp_name="IMP-SOS")
    table.make_numbers(task="etp", exp_name="IMP-K")
    table.make_two_results_table_same_task(task="etp", exp_names=["IMP-K", "IMP-SOS"], metrics='final-state-match')


@subcommand(category=Category.LATEX)
def gen_pcp_op_imp_k_results_table():
    table = Table()
    table.make_numbers(task="pcp", exp_name="IMP-K")
    table.make_numbers(task="op", exp_name="IMP-K")
    table.make_two_results_table(exp_name="IMP-K", tasks=["pcp", "op"])


@subcommand(category=Category.LATEX)
def gen_srp_etp_imp_sos_results_table():
    table = Table()
    table.make_numbers(task="srp", exp_name="IMP-SOS")
    table.make_numbers(task="etp", exp_name="IMP-SOS")
    table.make_two_results_table(exp_name="IMP-SOS", tasks=["srp", "etp"])

@subcommand(category=Category.LATEX)
def gen_etp_percentage_match_imp_sos_results_table(mutation: str):
    # Pick mutation from ['uk', 'addSub_mulDiv_negateRelation', 'unseen']
    table = Table()
    table.make_etp_percentage_match_table(mutation, exp_name="IMP-SOS")

@subcommand(category=Category.LATEX)
def gen_etp_percentage_match_imp_k_results_table(mutation: str):
    # Pick mutation from ['uk', 'addSub_mulDiv_negateRelation', 'unseen']
    table = Table()
    table.make_etp_percentage_match_table(mutation, exp_name="IMP-K")


@subcommand(category=Category.LATEX)
def gen_benchmark_dataset_table():
    table = Table()
    table.make_dataset_numbers(dataset="benchmark")
    table.make_benchmark_dataset_table(ds_name="benchmark")


@subcommand(category=Category.LATEX)
def gen_program_executability_stats_table(paper: str = "lmpl26"):
    table = Table(paper=paper)
    table.make_program_executability_stats_table()


@subcommand(category=Category.LATEX)
def gen_imp_split_loc_stats_table(paper: str = "lmpl26"):
    table = Table(paper=paper)
    table.make_imp_split_loc_stats_table()


@subcommand(category=Category.LATEX)
def gen_op_uk_odds_per_iqr_table(dataset_file: str = str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric.jsonl"), filter_while_count: int = - 1, filter_if_count: int = -1):
    table = Table()
    model_list = [
        "meta-llama-Llama-3.3-70B-Instruct:SOS:da",
        "meta-llama-Llama-3.3-70B-Instruct:SOS:cot",
        "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:da",
        "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:cot",
        "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:da",
        "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:cot",
        "gpt-4o-mini:SOS:da",
        "gpt-4o-mini:SOS:cot",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:SOS:da",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B:SOS:da",
        "meta-llama-Llama-3.3-70B-Instruct:K:da",
        "meta-llama-Llama-3.3-70B-Instruct:K:cot",
        "Qwen-Qwen2.5-Coder-14B-Instruct:K:da",
        "Qwen-Qwen2.5-Coder-14B-Instruct:K:cot",
        "Qwen-Qwen2.5-Coder-32B-Instruct:K:da",
        "Qwen-Qwen2.5-Coder-32B-Instruct:K:cot",
        "gpt-4o-mini:K:da",
        "gpt-4o-mini:K:cot",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:K:da",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B:K:da"
    ]
    models: list = []
    semantics_types: list = []
    strategies: list = []
    op_traces: list = []

    for model in model_list:
        models.append(model.split(":")[0])
        semantics_types.append(model.split(":")[1])
        strategies.append(model.split(":")[2])
    #rof

    for (model, semantics_type, strategy) in zip(models, semantics_types, strategies):
        analyzer = ResultsAnalyzer(
            task="op",
            setup_name='uk',
            exp_name=f"IMP-{semantics_type}",
            model_list=[],
        )
        op_traces.append(analyzer.raw_results_trace(model, strategy))
    #rof
    metrics: list[str] = ['LOC', 'Volume', 'Vocabulary', 'CC', 'MaxNestedIf', 'MaxNestedLoop', 'DepDegree', 'NumAssignments', 'TraceLength']
    headers: list[str] = ['CC', r'$\texttt{max}_{\texttt{if}}$', r'$\texttt{max}_{\texttt{loop}}$', 'DepDeg.', '\#Assign.', 'LOC', 'Vol.', 'Vocab.', r'$\left\vert\texttt{Trace}\right\vert$']
    header_cat: dict = {'Size': ['LOC', 'Volume', 'Vocabulary', 'TraceLength'], 'Control-flow': ['CC', 'MaxNestedIf', 'MaxNestedLoop'], 'Data-flow': ['DepDegree', 'NumAssignments']}
    table.make_op_odds_per_iqr_table(models, semantics_types, strategies, op_traces, dataset_file, metrics, headers, header_cat, filter_while_count, filter_if_count)
#fed


@subcommand(category=Category.LATEX)
def gen_op_odds_per_iqr_table_datasets_combined(dataset_files: list[str] = [str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric.jsonl"),str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric-synthetic_cpp.jsonl"),str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric-fuzzer_generated.jsonl")], dataset_names: list[str] = ['human_written', 'synthetic_cpp', 'fuzzer_generated'], mutation: str = 'uk', filter_while_count: int = - 1, filter_if_count: int = -1):
    table = Table()
    model_lists = {
        "human_written": [
            "meta-llama-Llama-3.3-70B-Instruct:SOS:da",
            "meta-llama-Llama-3.3-70B-Instruct:SOS:cot",
            "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:da",
            "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:da",
            "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:cot",
            "gpt-4o-mini:SOS:da",
            "gpt-4o-mini:SOS:cot",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:SOS:da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B:SOS:da",
            "meta-llama-Llama-3.3-70B-Instruct:K:da",
            "meta-llama-Llama-3.3-70B-Instruct:K:cot",
            "Qwen-Qwen2.5-Coder-14B-Instruct:K:da",
            "Qwen-Qwen2.5-Coder-14B-Instruct:K:cot",
            "Qwen-Qwen2.5-Coder-32B-Instruct:K:da",
            "Qwen-Qwen2.5-Coder-32B-Instruct:K:cot",
            "gpt-4o-mini:K:da",
            "gpt-4o-mini:K:cot",
            "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:K:da",
            "deepseek-ai-DeepSeek-R1-Distill-Llama-70B:K:da"
        ],
        "synthetic_cpp": [
            "Qwen-QwQ-32B:SOS:da",
            "Qwen-QwQ-32B:K:da",
        ],
        "fuzzer_generated": [
            "Qwen-QwQ-32B:SOS:da",
            "gpt-5-mini:SOS:da",
            "gemini-2.5-pro:SOS:da",
            "Qwen-QwQ-32B:K:da",
            "gpt-5-mini:K:da",
            "gemini-2.5-pro:K:da"
        ]
    }

    def get_odds_inputs(model_list: list, dataset_name: str):
        models: list = []
        semantics_types: list = []
        strategies: list = []
        op_traces: list = []
        for model in model_list:
            if mutation == "nk" and model.split(":")[1] != "SOS":
                continue
            #fi
            models.append(model.split(":")[0])
            semantics_types.append(model.split(":")[1])
            strategies.append(model.split(":")[2])
        #rof

        for (model, semantics_type, strategy) in zip(models, semantics_types, strategies):
            analyzer = ResultsAnalyzer(
                task="op",
                setup_name=mutation,
                exp_name=f"IMP-{semantics_type}",
                dataset_name=dataset_name,
                filter_dataset = Macros.data_dir / "dataset" / "dataset-etp-uk-IMP-K-synthetic_cpp.jsonl" if dataset_name == "synthetic_cpp" else None,
                model_list=[],
            )
            op_traces.append(analyzer.raw_results_trace(model, strategy))
        #rof
        return (models, semantics_types, strategies, op_traces)
    #fed

    models = []
    semantics_types = []
    strategies = []
    op_traces = []

    for dataset_name in dataset_names:
        [model, semantic_type, strategy, op_trace] = get_odds_inputs(model_lists[dataset_name], dataset_name)
        models.append(model)
        semantics_types.append(semantic_type)
        strategies.append(strategy)
        op_traces.append(op_trace)
    #rof
    
    metrics: list[str] = ['LOC', 'Volume', 'Vocabulary', 'CC', 'MaxTakenIf', 'MaxTakenLoop', 'DepDegree', 'NumAssignments', 'TraceLength']
    headers: list[str] = [r'\metricCCBold', r'\metricMaxtakenifBold', r'\metricMaxtakenwhileBold', r'\metricDepdegBold', r'\metricNumAssignBold', r'\metricLocBold', r'\metricVolBold', r'\metricVocabBold', r'\metricTracelenBold']
    header_cat: dict = {'Size': ['LOC', 'Volume', 'Vocabulary', 'TraceLength'], 'Control-flow': ['CC', 'MaxTakenIf', 'MaxTakenLoop'], 'Data-flow': ['DepDegree', 'NumAssignments']}
    table.make_op_odds_per_iqr_table_combined(models, semantics_types, strategies, op_traces, dataset_files, dataset_names, mutation, metrics, headers, header_cat, filter_while_count, filter_if_count, {'K': 'K', 'SOS': '\Sos'} if mutation == 'uk' else {'SOS': '\Sos'})
#fed


@subcommand(category=Category.LATEX)
def gen_op_uk_odds_per_iqr_table_fuzzer_generated(dataset_file: str = str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric-fuzzer_generated.jsonl"), filter_while_count: int = - 1, filter_if_count: int = -1):
    table = Table()
    model_list = [
        "Qwen-QwQ-32B:K:da",
        "Qwen-QwQ-32B:SOS:da",
        "gemini-2.5-pro:K:da",
        "gemini-2.5-pro:SOS:da",
        "gpt-5-mini:K:da",
        "gpt-5-mini:SOS:da",
    ]
    models: list = []
    semantics_types: list = []
    strategies: list = []
    op_traces: list = []

    for model in model_list:
        models.append(model.split(":")[0])
        semantics_types.append(model.split(":")[1])
        strategies.append(model.split(":")[2])
    #rof

    for (model, semantics_type, strategy) in zip(models, semantics_types, strategies):
        analyzer = ResultsAnalyzer(
            task="op",
            setup_name='uk',
            exp_name=f"IMP-{semantics_type}",
            model_list=[],
        )
        op_traces.append(analyzer.raw_results_trace(model, strategy))
    #rof
    metrics: list[str] = ['LOC', 'Volume', 'Vocabulary', 'CC', 'MaxNestedIf', 'MaxNestedLoop', 'DepDegree', 'NumAssignments', 'TraceLength']
    headers: list[str] = ['CC', r'$\texttt{max}_{\texttt{if}}$', r'$\texttt{max}_{\texttt{loop}}$', 'DepDeg.', '\#Assign.', 'LOC', 'Vol.', 'Vocab.', r'$\left\vert\texttt{Trace}\right\vert$']
    header_cat: dict = {'Size': ['LOC', 'Volume', 'Vocabulary', 'TraceLength'], 'Control-flow': ['CC', 'MaxNestedIf', 'MaxNestedLoop'], 'Data-flow': ['DepDegree', 'NumAssignments']}
    table.make_op_odds_per_iqr_table(models, semantics_types, strategies, op_traces, dataset_file, metrics, headers, header_cat, filter_while_count, filter_if_count)
#fed


@subcommand(category=Category.LATEX)
def gen_op_nk_odds_per_iqr_table(dataset_file: str = str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric.jsonl"), filter_while_count: int = - 1, filter_if_count: int = -1):
    table = Table()
    model_list = [
        "meta-llama-Llama-3.3-70B-Instruct:da",
        "meta-llama-Llama-3.3-70B-Instruct:cot",
        "Qwen-Qwen2.5-Coder-14B-Instruct:da",
        "Qwen-Qwen2.5-Coder-14B-Instruct:cot",
        "Qwen-Qwen2.5-Coder-32B-Instruct:da",
        "Qwen-Qwen2.5-Coder-32B-Instruct:cot",
        "gpt-4o-mini:da",
        "gpt-4o-mini:cot",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:da",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B:da",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B:da",
    ]
    models: list = []
    semantics_types: list = []
    strategies: list = []
    op_traces: list = []

    for model in model_list:
        models.append(model.split(":")[0])
        semantics_types.append("SOS")
        strategies.append(model.split(":")[1])
    #rof

    for (model, strategy) in zip(models, strategies):
        analyzer = ResultsAnalyzer(
            task="op",
            setup_name='nk',
            exp_name=f"IMP-SOS",
            model_list=[],
        )
        op_traces.append(analyzer.raw_results_trace(model, strategy))
    #rof
    metrics: list[str] = ['LOC', 'Volume', 'Vocabulary', 'CC', 'MaxNestedIf', 'MaxNestedLoop', 'DepDegree', 'NumAssignments', 'TraceLength']
    headers: list[str] = ['CC', r'$\texttt{max}_{\texttt{if}}$', r'$\texttt{max}_{\texttt{loop}}$', 'DepDeg.', '\#Assign.', 'LOC', 'Vol.', 'Vocab.', r'$\left\vert\texttt{Trace}\right\vert$']
    header_cat: dict = {'Size': ['LOC', 'Volume', 'Vocabulary', 'TraceLength'], 'Control-flow': ['CC', 'MaxNestedIf', 'MaxNestedLoop'], 'Data-flow': ['DepDegree', 'NumAssignments']}
    table.make_op_nk_odds_per_iqr_table(models, semantics_types, strategies, op_traces, dataset_file, metrics, headers, header_cat, filter_while_count, filter_if_count)
#fed


@subcommand(category=Category.LATEX)
def gen_dataset_stats_table(dataset_files: list[str] = [str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric.jsonl"), str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric-synthetic_cpp.jsonl"), str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric-fuzzer_generated.jsonl")], dataset_names: list[str] = ['human_written','synthetic_cpp','fuzzer_generated']):
    table = Table()
    headers: list[str] = [r'\metricCCBold', r'\metricMaxnestifBold', r'\metricMaxnestwhileBold', r'\metricMaxtakenifBold', r'\metricMaxtakenwhileBold', r'\metricDepdegBold', r'\metricNumAssignBold', r'\metricLocBold', r'\metricVolBold', r'\metricVocabBold', r'\metricTracelenBold']
    header_cat: dict = {'Size': ['LOC', 'Volume', 'Vocabulary', 'TraceLength'], 'Control-flow': ['CC', 'MaxNestedIf', 'MaxNestedLoop', 'MaxTakenIf', 'MaxTakenLoop'], 'Data-flow': ['DepDegree', 'NumAssignments']}
    table.make_dataset_stats_table(dataset_files, dataset_names, headers, header_cat)
#fed



# figures


@subcommand(category=Category.LATEX)
def gen_pcp_radar_charts():
    gen_pcp_uk_radar_chart()
    gen_pcp_mk_replace_radar_chart()
    gen_pcp_mk_unseen_radar_chart()


# fed


@subcommand(category=Category.LATEX)
def gen_pcp_uk_radar_chart(semantics_type: str, legend: bool = False):
    rules: dict = {}
    match semantics_type:
        case "SOS":
            rules = {
                "None": "No error",
                "Rule 2": "Use before declare",
                "Rule 19": "Divide by 0",
                "Rule 23": "Modulo by 0",
                "Rule 76": "Continue outside loop",
                "Rule 73": "Break outside loop",
            }
        case "K":
            rules = {
                "None": "No error",
                "Rule 2": "Use before declare",
                "Rule 7": "Divide by 0",
                "Rule 9": "Modulo by 0",
                "Rule 31": "Continue outside loop",
                "Rule 34": "Break outside loop",
            }            
    gen_pcp_radar_chart("uk", semantics_type, rules, f"pcp-uk-{semantics_type}-radar-chart", legend)


# fed


@subcommand(category=Category.LATEX)
def gen_pcp_mk_replace_radar_chart(semantics_type: str, legend: bool = False):
    rules: dict = {}
    match semantics_type:
        case "SOS":
            rules = {
                "addSub_mulDiv_negateRelation-None": "No error",
                "addSub_mulDiv_negateRelation-Rule 2": "Use before declare",
                "addSub_mulDiv_negateRelation-Rule 19": "Divide by 0",
                "addSub_mulDiv_negateRelation-Rule 23": "Modulo by 0",
                "addSub_mulDiv_negateRelation-Rule 76": "Continue outside loop",
                "addSub_mulDiv_negateRelation-Rule 73": "Break outside loop",
            }
        case "K":
            rules = {
                "addSub_mulDiv_negateRelation-None": "No error",
                "addSub_mulDiv_negateRelation-Rule 2": "Use before declare",
                "addSub_mulDiv_negateRelation-Rule 7": "Divide by 0",
                "addSub_mulDiv_negateRelation-Rule 9": "Modulo by 0",
                "addSub_mulDiv_negateRelation-Rule 31": "Continue outside loop",
                "addSub_mulDiv_negateRelation-Rule 34": "Break outside loop",
            }
    gen_pcp_radar_chart("mk", semantics_type, rules, f"pcp-mk-replace-{semantics_type}-radar-chart", legend)


# fed


@subcommand(category=Category.LATEX)
def gen_pcp_mk_unseen_radar_chart(semantics_type:str, legend: bool = False):
    rules: dict = {}
    match semantics_type:
        case "SOS":
            rules = {
                "unseen-None": "No error",
                "unseen-Rule 2": "Use before declare",
                "unseen-Rule 19": "Divide by 0",
                "unseen-Rule 23": "Modulo by 0",
                "unseen-Rule 76": "Continue outside loop",
                "unseen-Rule 73": "Break outside loop",
            }
        case "K":
            rules = {
                "unseen-None": "No error",
                "unseen-Rule 2": "Use before declare",
                "unseen-Rule 7": "Divide by 0",
                "unseen-Rule 9": "Modulo by 0",
                "unseen-Rule 31": "Continue outside loop",
                "unseen-Rule 34": "Break outside loop",
            }
    gen_pcp_radar_chart("mk", semantics_type, rules, f"pcp-mk-unseen-{semantics_type}-radar-chart", legend)


# fed


# generate the barplot for the pcp task
@subcommand(category=Category.LATEX)
def gen_pcp_barplot(legend: bool = True):
    semantic_types = ["mk"]

    # Load data for all types
    data_dict = {}
    for semantic in semantic_types:
        data_dict[semantic] = load_json_data(
            f"{Macros.results_dir}/metrics-pcp-{semantic}-IMP-SOS.json"
        )
    ax = make_bars(data_dict["mk"], metric="accuracy", color="#B39DDB")
    ax.set_ylabel("Accuracy", fontsize=14)
    # ax.set_title("PCP Task Performance by Model", fontsize=16)
    plt.tight_layout()
    plt.savefig("pcp-barplot.svg", format="svg", bbox_inches="tight")
    plt.close()


@subcommand(category=Category.LATEX)
def gen_pop_barplot(legend: bool = True):
    semantic_types = ["mk"]

    # Load data for all types
    data_dict = {}
    for semantic in semantic_types:
        data_dict[semantic] = load_json_data(
            f"{Macros.results_dir}/metrics-op-{semantic}-IMP-SOS.json"
        )
    ax = make_bars(data_dict["mk"], metric="acc", color="#F6C6BD")
    ax.set_ylabel("Accuracy", fontsize=14)
    # ax.set_title("PCP Task Performance by Model", fontsize=16)
    plt.tight_layout()
    plt.savefig("pop-barplot.svg", format="svg", bbox_inches="tight")
    plt.close()


@subcommand(category=Category.LATEX)
def gen_srp_barplot(legend: bool = True):
    semantic_types = ["mk"]

    # Load data for all types
    data_dict = {}
    for semantic in semantic_types:
        data_dict[semantic] = load_json_data(
            f"{Macros.results_dir}/metrics-srp-{semantic}-IMP-SOS.json"
        )
    ax = make_bars(data_dict["mk"], metric="xmatch-accuracy", color="#A0D6B4")
    ax.set_ylabel("xMatch", fontsize=14)
    # ax.set_title("PCP Task Performance by Model", fontsize=16)
    plt.tight_layout()
    plt.savefig("srp-barplot.svg", format="svg", bbox_inches="tight")
    plt.close()


@subcommand(category=Category.LATEX)
def gen_etp_barplot(legend: bool = True):
    semantic_types = ["mk"]

    # Load data for all types
    data_dict = {}
    for semantic in semantic_types:
        data_dict[semantic] = load_json_data(
            f"{Macros.results_dir}/metrics-srp-{semantic}-IMP-SOS.json"
        )
    ax = make_bars(data_dict["mk"], metric="xmatch-accuracy", color="#FBE7A1")
    ax.set_ylabel("xMatch", fontsize=14)
    # ax.set_title("PCP Task Performance by Model", fontsize=16)
    plt.tight_layout()
    plt.savefig("etp-barplot.svg", format="svg", bbox_inches="tight")
    plt.close()


@subcommand(category=Category.LATEX)
def gen_srp_radar_charts(expr_name: str):
    
    gen_srp_uk_first_mismatch_rule_radar_chart(expr_name)
    gen_srp_mk_replace_first_mismatch_rule_radar_chart(expr_name)
    gen_srp_mk_unseen_first_mismatch_rule_radar_chart(expr_name)
    gen_srp_uk_most_correct_rules_radar_chart(expr_name)
    gen_srp_mk_replace_most_correct_rules_radar_chart(expr_name)
    gen_srp_mk_unseen_most_correct_rules_radar_chart(expr_name)

# fed


@subcommand(category=Category.LATEX)
def gen_srp_uk_first_mismatch_rule_radar_chart(expr_name: str, legend: bool = False):
    gen_srp_radar_chart(
        prefix="",
        expr_name=expr_name,
        data_type="first-mismatch-rule",
        legend=legend,
    )

# fed


@subcommand(category=Category.LATEX)
def gen_srp_mk_replace_first_mismatch_rule_radar_chart(expr_name: str, legend: bool = False):
    gen_srp_radar_chart(
        prefix="addSub_mulDiv_negateRelation-",
        expr_name=expr_name,
        data_type="first-mismatch-rule",
        legend=legend,
    )


# fed


@subcommand(category=Category.LATEX)
def gen_srp_mk_unseen_first_mismatch_rule_radar_chart(expr_name: str, legend: bool = False):
    gen_srp_radar_chart(
        prefix="unseen-",
        expr_name=expr_name,
        data_type="first-mismatch-rule",
        legend=legend,
    )


# fed


@subcommand(category=Category.LATEX)
def gen_srp_uk_most_correct_rules_radar_chart(expr_name: str, legend: bool = False):
    gen_srp_radar_chart(
        prefix="",
        expr_name=expr_name,
        data_type="most-correct-rules",
        legend=legend,
    )


# fed


@subcommand(category=Category.LATEX)
def gen_srp_mk_replace_most_correct_rules_radar_chart(expr_name: str, legend: bool = False):
    gen_srp_radar_chart(
        prefix="addSub_mulDiv_negateRelation-",
        expr_name=expr_name,
        data_type="most-correct-rules",
        legend=legend,
    )


# fed


@subcommand(category=Category.LATEX)
def gen_srp_mk_unseen_most_correct_rules_radar_chart(expr_name: str, legend: bool = False):
    gen_srp_radar_chart(
        prefix="unseen-",
        expr_name=expr_name,
        data_type="most-correct-rules",
        legend=legend,
    )


# fed


@subcommand(category=Category.LATEX)
def gen_etp_radar_charts():
    gen_etp_uk_radar_chart()
    gen_etp_mk_replace_radar_chart()
    gen_etp_mk_unseen_radar_chart()


# fed


@subcommand(category=Category.LATEX)
def gen_etp_uk_radar_chart(legend: bool = False):
    gen_etp_radar_chart("", "uk", "none", legend)


# fed


@subcommand(category=Category.LATEX)
def gen_etp_mk_replace_radar_chart(legend: bool = False):
    gen_etp_radar_chart("addSub_mulDiv_negateRelation-", "mk", "replace", legend)


# fed


@subcommand(category=Category.LATEX)
def gen_etp_mk_unseen_radar_chart(legend: bool = False):
    gen_etp_radar_chart("unseen-", "mk", "unseen", legend)


@subcommand(category=Category.LATEX)
def gen_caucasian_albanian_font_images():
    start = 0x10530
    end = 0x1056F

    FONT_PATH = f"{Macros.ext_dir}/NotoSansCaucasianAlbanian-Regular.ttf"
    FONT_SIZE = 64
    OUTPUT_DIR = "caucasian_albanian_chars"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception as e:
        print("Failed to load font:", e)
        exit()

    for codepoint in range(start, end + 1):
        char = chr(codepoint)
        hexcode = f"{codepoint:04X}"

        img = Image.new("RGBA", (128, 128), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # Get the bounding box for the text
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]  # width and height of the text

        position = ((128 - w) // 2, (128 - h) // 2)

        draw.text(position, char, font=font, fill="black")

        output_path = os.path.join(OUTPUT_DIR, f"U+{hexcode}.png")
        img.save(output_path)

        print(f"Saved: {output_path}")


# fed


@subcommand(category=Category.LATEX)
def gen_code_complexity_plot_compare(data_paths: str, metric_name: str, data_names: str):
   data_paths = data_paths.split(",")
   data_names = data_names.split(",")
   data_lists: list = []
   for data_path in data_paths:
       data_lists.append(su.io.load(data_path))
   #rof
   plot_code_complexity_metric(data_lists, metric_name, data_names)
   
#fed


@subcommand(category=Category.LATEX)
def gen_pcp_model_confusion_matrix(model_name: str, semantics_type: str, mutation_type: str, strategy: str):
    analyzer = ResultsAnalyzer(
        task="pcp",
        setup_name='uk' if mutation_type == 'standard' else 'mk',
        exp_name=f"IMP-{semantics_type}",
        model_list=[],
    )
    rules: dict = {}
    match semantics_type:
        case "SOS":
            rules = {
                "None": "No error",
                "Rule 2": "Use before declare",
                "Rule 6": "Use before declare",
                "Rule 19": "Divide by 0",
                "Rule 23": "Modulo by 0",
                "Rule 76": "Continue outside loop",
                "Rule 73": "Break outside loop",
                "_"      : "Unknown"
            }
        case "K":
            rules = {
                "None": "No error",
                "Rule 2": "Use before declare",
                "Rule 7": "Divide by 0",
                "Rule 9": "Modulo by 0",
                "Rule 31": "Continue outside loop",
                "Rule 34": "Break outside loop",
                "_"      : "Unknown"
            }

    model_pcp_result = analyzer.raw_results_trace(model_name, strategy)
    pcp_trace = (
        model_pcp_result
        if mutation_type == 'standard'
        else [trace[mutation_type] for trace in model_pcp_result]
    )

    plot_pcp_error_confusion_matrix(
        [model_name],
        [semantics_type],
        [mutation_type],
        [strategy],
        [pcp_trace],
        [rules],
    )
#fed


@subcommand(category=Category.LATEX)
def gen_pcp_model_confusion_matrices(model_list: list[str]):
    models: list = []
    semantics_types: list = []
    mutation_types: list = []
    strategies: list = []
    pcp_traces: list = []
    rules: list = []

    for model in model_list:
        models.append(model.split(":")[0])
        semantics_types.append(model.split(":")[1])
        mutation_types.append(model.split(":")[2])
        strategies.append(model.split(":")[3])
    #rof

    for (model, semantics_type, mutation_type, strategy) in zip(models, semantics_types, mutation_types, strategies):
        analyzer = ResultsAnalyzer(
            task="pcp",
            setup_name='uk' if mutation_type == 'standard' else 'mk',
            exp_name=f"IMP-{semantics_type}",
            model_list=[],
        )
        model_pcp_result = analyzer.raw_results_trace(model, strategy)
        pcp_traces.append(model_pcp_result if mutation_type == 'standard' else [pcp_trace[mutation_type] for pcp_trace in model_pcp_result])
        match semantics_type:
            case "SOS":
                rules.append({
                    "None": "No error",
                    "Rule 2": "Use before declare",
                    "Rule 6": "Use before declare",
                    "Rule 19": "Divide by 0",
                    "Rule 23": "Modulo by 0",
                    "Rule 76": "Continue outside loop",
                    "Rule 73": "Break outside loop",
                    "_"      : "Unknown"
                })
            case "K":
                rules.append({
                    "None": "No error",
                    "Rule 2": "Use before declare",
                    "Rule 7": "Divide by 0",
                    "Rule 9": "Modulo by 0",
                    "Rule 31": "Continue outside loop",
                    "Rule 34": "Break outside loop",
                    "_"      : "Unknown"
                })
    #rof

    plot_pcp_error_confusion_matrix(models, semantics_types, mutation_types, strategies, pcp_traces, rules)
#fed


@subcommand(category=Category.LATEX)
def gen_pcp_dendrogram():
    model_list = [
        # "meta-llama-Llama-3.3-70B-Instruct:SOS:da:uk",
        # "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:da:uk",
        # "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:cot:uk",
        # "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:da:uk",
        # "gpt-4o-mini:SOS:da:uk",
        # "Qwen-Qwen2.5-Coder-14B-Instruct:K:da:uk",
        # "Qwen-Qwen2.5-Coder-32B-Instruct:K:da:uk",
        # "gpt-4o-mini:K:da:uk",
        # "gpt-4o-mini:K:cot:uk",
        
        # "meta-llama-Llama-3.3-70B-Instruct:SOS:da:addSub_mulDiv_negateRelation",
        # "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:da:addSub_mulDiv_negateRelation",
        # "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:cot:addSub_mulDiv_negateRelation",
        # "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:da:addSub_mulDiv_negateRelation",
        # "gpt-4o-mini:SOS:da:addSub_mulDiv_negateRelation",
        # "meta-llama-Llama-3.3-70B-Instruct:K:da:addSub_mulDiv_negateRelation",
        # "Qwen-Qwen2.5-Coder-14B-Instruct:K:da:addSub_mulDiv_negateRelation",
        # "Qwen-Qwen2.5-Coder-32B-Instruct:K:da:addSub_mulDiv_negateRelation",
        # "gpt-4o-mini:K:da:addSub_mulDiv_negateRelation",


        "meta-llama-Llama-3.3-70B-Instruct:SOS:da:unseen",
        "meta-llama-Llama-3.3-70B-Instruct:SOS:cot:unseen",
        "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:da:unseen",
        "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:cot:unseen",
        "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:da:unseen",
        "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:cot:unseen",
        "gpt-4o-mini:SOS:da:unseen",
        "gpt-4o-mini:SOS:cot:unseen",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:SOS:da:unseen",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B:SOS:da:unseen",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B:SOS:da:unseen",
        "meta-llama-Llama-3.3-70B-Instruct:K:da:unseen",
        "Qwen-Qwen2.5-Coder-14B-Instruct:K:da:unseen",
        "Qwen-Qwen2.5-Coder-14B-Instruct:K:cot:unseen",
        "Qwen-Qwen2.5-Coder-32B-Instruct:K:da:unseen",
        "Qwen-Qwen2.5-Coder-32B-Instruct:K:cot:unseen",
        "gpt-4o-mini:K:da:unseen",
        "gpt-4o-mini:K:cot:unseen",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:K:da:unseen",

    ]
    models: list = []
    semantics_types: list = []
    mutation_types: list = []
    strategies: list = []
    pcp_traces: list = []
    rules: list = []

    for model in model_list:
        models.append(model.split(":")[0])
        semantics_types.append(model.split(":")[1])
        strategies.append(model.split(":")[2])
        mutation_types.append(model.split(":")[3])
    #rof

    for (model, semantics_type, mutation_type, strategy) in zip(models, semantics_types, mutation_types, strategies):
        analyzer = ResultsAnalyzer(
            task="pcp",
            setup_name='uk' if mutation_type == 'uk' else 'mk',
            exp_name=f"IMP-{semantics_type}",
            model_list=[],
        )
        model_pcp_result = analyzer.raw_results_trace(model, strategy)
        pcp_traces.append(model_pcp_result if mutation_type == 'uk' else [pcp_trace[mutation_type] for pcp_trace in model_pcp_result])
        match semantics_type:
            case "SOS":
                rules.append({
                    "None": "No error",
                    "Rule 2": "Use before declare",
                    "Rule 6": "Use before declare",
                    "Rule 19": "Divide by 0",
                    "Rule 23": "Modulo by 0",
                    "Rule 76": "Continue outside loop",
                    "Rule 73": "Break outside loop",
                    "_"      : "Unknown"
                })
            case "K":
                rules.append({
                    "None": "No error",
                    "Rule 2": "Use before declare",
                    "Rule 7": "Divide by 0",
                    "Rule 9": "Modulo by 0",
                    "Rule 31": "Continue outside loop",
                    "Rule 34": "Break outside loop",
                    "_"      : "Unknown"
                })
    #rof

    plot_pcp_dendrogram(models, semantics_types, mutation_types, strategies, pcp_traces, rules)
#fed


@subcommand(category=Category.LATEX)
def gen_op_model_metric_scatter_plot(model: str, metric_name: str, semantics_type: str = "SOS", strategy: str = "da"):
    analyzer = ResultsAnalyzer(
        task="op",
        setup_name='uk',
        exp_name=f"IMP-{semantics_type}",
        model_list=[],
    )
    op_results = analyzer.raw_results_trace(model, strategy)
    language = IMP("IMP1", Language.SEMANTICS_TYPE.SOS)
    plot_op_correct_percentage_against_metric(language, op_results, metric_name)
#fed

@subcommand(category=Category.LATEX)
def gen_metric_against_metric(dataset_file: str, metric1: str, metric2: str):
    plot_metric_against_metric(dataset_file, metric1, metric2)
#fed

# @subcommand(category=Category.LATEX)
# def gen_op_vif_table():
#     analyzer = ResultsAnalyzer(
#         task="op",
#         setup_name='uk',
#         exp_name=f"IMP-SOS",
#         model_list=[],
#     )
#     op_results = analyzer.raw_results_trace("o3-mini", "da")
#     language = IMP("IMP1", Language.SEMANTICS_TYPE.SOS)
#     table = Table()
#     table.make_op_predictor_vif_table(language, op_results)
# #fed


@subcommand(category=Category.LATEX)
def gen_op_regression_heatmap(model_list: list[str], dataset_file: str = str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric.jsonl"), dataset_name: str = "human_written", filter_while_count: int = 9, filter_if_count: int = -1):
    models: list = []
    semantics_types: list = []
    strategies: list = []
    op_traces: list = []

    for model in model_list:
        models.append(model.split(":")[0])
        semantics_types.append(model.split(":")[1])
        strategies.append(model.split(":")[2])
    #rof

    for (model, semantics_type, strategy) in zip(models, semantics_types, strategies):
        analyzer = ResultsAnalyzer(
            task="op",
            setup_name='nk',
            exp_name=f"IMP-{semantics_type}",
            dataset_name=dataset_name,
            model_list=[],
        )
        op_traces.append(analyzer.raw_results_trace(model, strategy))
    #rof
    metrics: list[str] = ['CC', 'LOC', 'Volume', 'Vocabulary', 'DepDegree', 'MaxTakenIf', 'MaxTakenLoop', 'NumAssignments', 'TraceLength']
    plot_op_regression_heatmap(models, semantics_types, strategies, op_traces, dataset_file, metrics, filter_while_count, filter_if_count)
#fed


@subcommand(category=Category.LATEX)
def gen_op_violin_plots(model_list: list[str], dataset_file: str = str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric.jsonl"), dataset_name: str = "human_written", filter_while_count: int = -1, filter_if_count: int = -1):
    models: list = []
    semantics_types: list = []
    strategies: list = []
    op_traces: list = []

    for model in model_list:
        models.append(model.split(":")[0])
        semantics_types.append(model.split(":")[1])
        strategies.append(model.split(":")[2])
    #rof

    for (model, semantics_type, strategy) in zip(models, semantics_types, strategies):
        analyzer = ResultsAnalyzer(
            task="op",
            setup_name='uk',
            exp_name=f"IMP-{semantics_type}",
            dataset_name=dataset_name,
            model_list=[],
        )
        op_traces.append(analyzer.raw_results_trace(model, strategy))
    #rof
    metrics: list[str] = ['CC', 'LOC', 'Volume', 'Vocabulary', 'DepDegree', 'MaxNestedIf', 'MaxTakenLoop', 'NumAssignments']
    plot_op_violin_plots(models, semantics_types, strategies, op_traces, dataset_file, metrics, filter_while_count, filter_if_count)
#fed


@subcommand(category=Category.LATEX)
def gen_dataset_violin_plots(dataset_files: list[str] = [str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric.jsonl"), str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric-synthetic_cpp.jsonl"), str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric-fuzzer_generated.jsonl")], dataset_names: list[str] = ['Human-Written','LLM-Translated','Fuzzer-Generated'], num_row: int = 4, num_col: int = 3):
    metrics: list[str] = ['CC','MaxNestedIf', 'MaxNestedLoop', 'MaxTakenIf', 'MaxTakenLoop','DepDegree', 'NumAssignments', 'LOC', 'Volume', 'Vocabulary', 'TraceLength']
    plot_dataset_violin_plots(dataset_files, metrics, dataset_names, num_row, num_col)
#fed


@subcommand(category=Category.LATEX)
def gen_op_dendrogram(dataset_file: str = str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric.jsonl"), filter_while_count: int = - 1, filter_if_count: int = -1):
    model_list = [
        "meta-llama-Llama-3.3-70B-Instruct:SOS:da:uk",
        "meta-llama-Llama-3.3-70B-Instruct:SOS:cot:uk",
        "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:da:uk",
        "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:cot:uk",
        "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:da:uk",
        "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:cot:uk",
        "gpt-4o-mini:SOS:da:uk",
        "gpt-4o-mini:SOS:cot:uk",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:SOS:da:uk",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B:SOS:da:uk",
        "meta-llama-Llama-3.3-70B-Instruct:K:da:uk",
        "meta-llama-Llama-3.3-70B-Instruct:K:cot:uk",
        "Qwen-Qwen2.5-Coder-14B-Instruct:K:da:uk",
        "Qwen-Qwen2.5-Coder-14B-Instruct:K:cot:uk",
        "Qwen-Qwen2.5-Coder-32B-Instruct:K:da:uk",
        "Qwen-Qwen2.5-Coder-32B-Instruct:K:cot:uk",
        "gpt-4o-mini:K:da:uk",
        "gpt-4o-mini:K:cot:uk",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:K:da:uk",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B:K:da:uk",
        "meta-llama-Llama-3.3-70B-Instruct:SOS:da:nk",
        "meta-llama-Llama-3.3-70B-Instruct:SOS:cot:nk",
        "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:da:nk",
        "Qwen-Qwen2.5-Coder-14B-Instruct:SOS:cot:nk",
        "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:da:nk",
        "Qwen-Qwen2.5-Coder-32B-Instruct:SOS:cot:nk",
        "gpt-4o-mini:SOS:da:nk",
        "gpt-4o-mini:SOS:cot:nk",
        "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B:SOS:da:nk",
        "deepseek-ai-DeepSeek-R1-Distill-Llama-70B:SOS:da:nk",
    ]
    models: list = []
    semantics_types: list = []
    strategies: list = []
    mutations: list = []
    op_traces: list = []

    for model in model_list:
        models.append(model.split(":")[0])
        semantics_types.append(model.split(":")[1])
        strategies.append(model.split(":")[2])
        mutations.append(model.split(":")[3])
    #rof

    for (model, semantics_type, strategy, mutation) in zip(models, semantics_types, strategies, mutations):
        analyzer = ResultsAnalyzer(
            task="op",
            setup_name=mutation,
            exp_name=f"IMP-{semantics_type}",
            model_list=[],
        )
        op_traces.append(analyzer.raw_results_trace(model, strategy))
    #rof
    metrics: list[str] = ['LOC', 'Volume', 'Vocabulary', 'CC', 'MaxTakenIf', 'MaxTakenLoop', 'DepDegree', 'NumAssignments', 'TraceLength']
    plot_op_beta_sd_dendrogram(models, semantics_types, strategies, op_traces, dataset_file, metrics, mutations, filter_while_count, filter_if_count)
#fed


@subcommand(category=Category.LATEX)
def gen_op_dendrogram_fuzzer_generated(dataset_file: str = str(Macros.data_dir / "dataset/dataset-IMP-code-complexity-metric-fuzzer_generated.jsonl"), filter_while_count: int = - 1, filter_if_count: int = -1):
    model_list = [
        #"Qwen-QwQ-32B:K:da:uk",
        #"Qwen-QwQ-32B:SOS:da:uk",
        "gemini-2.5-pro:K:da:uk",
        "gemini-2.5-pro:SOS:da:uk",
        #"gpt-5-mini:K:da:uk",
        #"gpt-5-mini:SOS:da:uk",
        #"Qwen-QwQ-32B:SOS:da:nk",
        "gemini-2.5-pro:SOS:da:nk",
        #"gpt-5-mini:SOS:da:nk",
    ]
    models: list = []
    semantics_types: list = []
    strategies: list = []
    mutations: list = []
    op_traces: list = []

    for model in model_list:
        models.append(model.split(":")[0])
        semantics_types.append(model.split(":")[1])
        strategies.append(model.split(":")[2])
        mutations.append(model.split(":")[3])
    #rof

    for (model, semantics_type, strategy, mutation) in zip(models, semantics_types, strategies, mutations):
        analyzer = ResultsAnalyzer(
            task="op",
            setup_name=mutation,
            exp_name=f"IMP-{semantics_type}",
            model_list=[],
            dataset_name="fuzzer_generated"
        )
        op_traces.append(analyzer.raw_results_trace(model, strategy))
    #rof
    metrics: list[str] = ['LOC', 'Volume', 'Vocabulary', 'CC', 'MaxNestedIf', 'MaxNestedLoop', 'DepDegree', 'NumAssignments', 'TraceLength']
    plot_op_beta_sd_dendrogram(models, semantics_types, strategies, op_traces, dataset_file, metrics, mutations, filter_while_count, filter_if_count)
#fed

@subcommand(category=Category.LATEX)
def gen_notation_comprehension_rule_distribution(
    hf_repo_id: str = "LambdaadbmaL/PLSemanticsBench", 
    hf_config: str = "nl2rule", 
    hf_split: str = "NonStandard_NumRule5_RandomSampleFalse"):
    plot_notation_comprehension_rule_distribution(hf_repo_id, hf_config, hf_split)
#fed

@subcommand(category=Category.LATEX)
def gen_notation_comprehension_rule_distribution_all(
    hf_repo_id: str = "LambdaadbmaL/PLSemanticsBench",
):
    plot_notation_comprehension_rule_distribution_all(hf_repo_id)
#fed


@subcommand(category=Category.LATEX)
def gen_notation_comprehension_confusion_matrix(
    model_name: str = "gpt-4o-mini", 
    task: str = "nl2rule", 
    semantics_type: str = "SOS", 
    setup_name: str = "uk",
    mutation_type: str = "keywordswap"
):
    analyzer = ResultsAnalyzer(
        task=task,
        setup_name=setup_name,
        exp_name=f"IMP-{semantics_type}",
        model_list=[model_name],
    )
    model_results = analyzer.raw_results_trace(model_name, "da")
    plot_notation_comprehension_confusion_matrix(model_results, setup_name, mutation_type)
#fed