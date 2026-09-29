import os
from pathlib import Path


class Macros:
    papers: list[str] = ["neurips25", "iclr26", "arxiv", "lmpl26"]
    main_dir = Path(os.path.realpath(__file__)).parent.resolve()
    test_oracles_dir = main_dir / "test_oracles"
    pytest_dir = main_dir / "../../tests"
    tmp_dir = main_dir / ".tmp"
    batch_dir = main_dir / ".batch"
    model_config_dir = main_dir / "../model_configs"
    papers_dir = main_dir / "../../papers"
    log_file = main_dir / "experiment.log"
    data_dir = (main_dir / "../../data").resolve()
    ext_dir = main_dir / "../../third_party"
    results_dir = main_dir / "../../results"
    pytest_oracles_dir = pytest_dir / "oracles"
    downloads_dir = main_dir / "../../_downloads"
    doc_dir = main_dir / "../../docs"

    # ANTLR related directories
    antlr_dir = main_dir / "antlr4_parsers"
    antlr_template_dir = antlr_dir / "imp" / "templates"
    antlr_imp_python_uk_dir = antlr_dir / "imp" / "python-uk"
    antlr_imp_java_uk_dir = antlr_dir / "imp" / "java-uk"
    antlr_imp_python_ks_dir = antlr_dir / "imp" / "python-ks"
    antlr_imp_java_ks_dir = antlr_dir / "imp" / "java-ks"
    antlr_imp_python_ko_dir = antlr_dir / "imp" / "python-ko"
    antlr_imp_java_ko_dir = antlr_dir / "imp" / "java-ko"
    antlr_jar_path = main_dir / "../../third_party" / "antlr-4.13.2-complete.jar"

    # IMP programs directories
    invalid_imp_uk_dir = data_dir / "imp" / "invalid_imp_programs"
    valid_imp_uk_dir = data_dir / "imp" / "valid_imp_programs" / "human_written"
    invalid_imp_mk_ks_dir = data_dir / "imp" / "mk_ks" / "invalid"
    valid_imp_mk_ks_dir = data_dir / "imp" / "mk_ks" / "valid"
    invalid_imp_mk_ko_dir = data_dir / "imp" / "mk_ko" / "invalid"
    valid_imp_mk_ko_dir = data_dir / "imp" / "mk_ko" / "valid"

    valid_imp_ans_dir = data_dir / "imp" / "valid_imp_programs" / "human_written_ans"

    # Feedback exmaples
    valid_examples = ["addition.imp"]
    invalid_examples = ["addition_modulo_zero.imp"]
    invalid_outputs = {
        "modulo_zero": "Semantic Error: Modulo by zero.",
    }


# ssalc
