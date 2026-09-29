from llm_interpreter.utils import subcommand, Category
from llm_interpreter.paper_latex.table import Table


@subcommand(category=Category.LATEX)
def gen_ig_uk_imp_sos_ebnf_results_table():
    table = Table(paper="iclr26")
    table.make_ig_numbers(task="ig", exp_name="uk-IMP-SOS-EBNF")
    table.make_ig_result_table(task="ig", exp_name="uk-IMP-SOS-EBNF")


@subcommand(category=Category.LATEX)
def gen_ig_uk_imp_sos_antlr_results_table():
    table = Table(paper="iclr26")
    table.make_ig_numbers(task="ig", exp_name="uk-IMP-SOS-ANTLR")
    table.make_ig_result_table(task="ig", exp_name="uk-IMP-SOS-ANTLR")


@subcommand(category=Category.LATEX)
def gen_translate_mk_imp_sos_results_table():
    table = Table(paper="iclr26")
    table.make_numbers(task="translate", exp_name="mk-IMP-SOS")
    table.make_result_table(task="translate", exp_name="mk-IMP-SOS")


@subcommand(category=Category.LATEX)
def gen_translate_uk_imp_sos_results_table():
    table = Table(paper="iclr26")
    table.make_numbers(task="translate", exp_name="uk-IMP-SOS")
    table.make_result_table(task="translate", exp_name="uk-IMP-SOS")

def gen_iga_uk_imp_sos_results_table():
    table = Table(paper="iclr26")
    table.make_ig_numbers(task="iga", exp_name="uk-IMP-SOS")
    table.make_ig_result_table(task="iga", exp_name="uk-IMP-SOS")


@subcommand(category=Category.LATEX)
def gen_iga_mk_imp_sos_results_table():
    table = Table(paper="iclr26")
    table.make_ig_numbers(task="iga", exp_name="mk-ks-IMP-SOS")
    table.make_ig_result_table(task="iga", exp_name="mk-ks-IMP-SOS")
    table.make_ig_numbers(task="iga", exp_name="mk-ko-IMP-SOS")
    table.make_ig_result_table(task="iga", exp_name="mk-ko-IMP-SOS")


@subcommand(category=Category.LATEX)
def gen_igaf_uk_imp_sos_results_table():
    table = Table(paper="iclr26")
    table.make_ig_numbers(task="igaf", exp_name="uk-IMP-SOS")
    table.make_ig_result_table(task="igaf", exp_name="uk-IMP-SOS")


@subcommand(category=Category.LATEX)
def gen_igaf_mk_imp_sos_results_table():
    table = Table(paper="iclr26")
    table.make_ig_numbers(task="igaf", exp_name="mk-ks-IMP-SOS")
    table.make_ig_result_table(task="igaf", exp_name="mk-ks-IMP-SOS")
    table.make_ig_numbers(task="igaf", exp_name="mk-ko-IMP-SOS")
    table.make_ig_result_table(task="igaf", exp_name="mk-ko-IMP-SOS")

