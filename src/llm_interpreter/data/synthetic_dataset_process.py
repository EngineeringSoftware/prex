import seutil as su
from tqdm import tqdm

from llm_interpreter.data.dataset_process import DatasetProcessor
from llm_interpreter.compiler_runners import KResult
from llm_interpreter.macros import Macros
from llm_interpreter.language import Language, IMP
from llm_interpreter.utils import write_to_tmp

logger = su.log.get_logger(__name__, su.log.INFO)


class SyntheticDataProcess(DatasetProcessor):
    def __init__(self, dataset):
        super().__init__(dataset)

    def collect_imp_programs(self):
        # Setup k framework
        execution_k = IMP(
            name="IMP_UNMUTATED",
            semantics_type=Language.SEMANTICS_TYPE.K,
        )
        self._setup_k_framework(program_lang=execution_k)

        # Collecting programs
        synthetic_programs = []
        raw_data = su.io.load(
            Macros.results_dir
            / "results-data-gen-uk-IMP-SOS-Qwen-Qwen2.5-Coder-32B-Instruct-da.jsonl",
            iter_line=True,
        )
        for item in raw_data:
            if "```imp" in item.get("model-prediction", ""):
                imp_code = extract_imp_code(item["model-prediction"])
                if imp_code:
                    synthetic_programs.append(imp_code)

        logger.info(f"Collected {len(synthetic_programs)} synthetic IMP programs.")

        # Execute programs with K semantics
        dataset = []
        for program in tqdm(
            synthetic_programs,
            desc="Processing synthetic IMP programs",
            total=len(synthetic_programs),
        ):
            dt = {"program": program, "language": "IMP"}
            k_result = self.run_k_on_imp(program, execution_k=execution_k)
            if k_result:
                dt["K-evaluatable"] = True
                dt["exec-trace"] = k_result.get_execution_trace()
                dt["final-state"] = k_result.get_final_state()
                dt["ans"] = k_result.get_final_state().get("ans")
                dataset.append(dt)

        # Save dataset to file
        su.io.dump(Macros.data_dir / "dataset" / "synthetic-imp-dataset.jsonl", dataset)

    def transform_imp_programs(
        self, out_dir=Macros.data_dir / "imp" / "valid_imp_programs" / "synthetic_cpp"
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_data = su.io.load(
            Macros.data_dir / "dataset" / "synthetic-imp-dataset.jsonl",
            iter_line=True,
        )

        for i, item in enumerate(raw_data):
            program = item["program"].splitlines()
            if program[-1].startswith("halt"):
                program = program[:-1]

            su.io.dump(
                out_dir / f"pgm_{i+1}.imp", "\n".join(program), fmt=su.io.Fmt.txt
            )

    def run_k_on_imp(self, program: str, execution_k: Language) -> KResult:
        write_to_tmp(program, "program.imp")
        try:
            k_result: KResult = self._get_runtime_result(
                f"{Macros.tmp_dir}/program.imp",
                program_language=execution_k,
                semantics_type=Language.SEMANTICS_TYPE.SOS,
            )
        except Exception as e:
            logger.error(f"Error executing IMP program: {e}")
            k_result = None

        return k_result

    def collect_csmith_programs(self):
        raw_dir = Macros.data_dir / "imp" / "synthetic" / "csmith"
        data = []

        for file in sorted(raw_dir.glob("*.c")):
            with open(file, "r") as f:
                code = f.read()
                filename = file.stem
                data.append({"program": code, "filename": filename})

        logger.info(f"Collected {len(data)} Csmith programs.")
        # Save dataset to file
        su.io.dump(Macros.data_dir / "dataset" / "dataset-dgc-uk-IMP-SOS.jsonl", data)

    def collect_translated_csmith_programs(self):
        model_name = "Qwen-Qwen2.5-Coder-14B-Instruct-da"
        out_dir = Macros.data_dir / "imp" / "synthetic" / "csmith_translated"
        out_dir.mkdir(parents=True, exist_ok=True)

        res_file = Macros.results_dir / "dgc/uk/IMP-SOS" / model_name / "results-42.jsonl"
        raw_data = su.io.load(res_file)

        print(f"Loaded {len(raw_data)} results from {res_file}")

        for item in raw_data:
            filename = item["filename"]
            lines = item["model-prediction"].splitlines()
            lines = [line for line in lines if "```" not in line]
            pgm = "\n".join(lines)

            su.io.dump(
                out_dir / f"{filename}.imp", pgm, fmt=su.io.Fmt.txt
            )

### Helper functions


def extract_imp_code(llm_output: str) -> str:
    """Extract the IMP code from the LLM output."""
    if "```imp" in llm_output:
        return llm_output.split("```imp")[1].split("```")[0]
    else:
        return ""
