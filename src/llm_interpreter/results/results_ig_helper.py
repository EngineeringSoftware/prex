from collections import defaultdict
import os
import json
import seutil as su
import shutil
from datetime import datetime
from llm_interpreter.macros import Macros
from typing import Tuple
from pathlib import Path

logger = su.log.get_logger(__name__, su.log.INFO)


class IGResultsHelper:
    """
    Helper class for handling results for ig tasks.
    """

    def __init__(self, res_dir: str, task: str, setup_name: str, exp_name: str):
        self.res_dir = Macros.tmp_dir / res_dir
        self.task = task
        self.setup_name = setup_name
        self.exp_name = exp_name
        self.suf_map = {
            "Python": "py",
            "Java": "java",
            "Rust": "rs",
            "Go": "go",
        }
        self.intp_output_dir = (
            self.res_dir
            / f"ig_intp_output_{datetime.now().strftime('%Y-%m-%d--%H:%M:%S.%f')}"
        )
        self.exec_output_dir = (
            self.res_dir
            / f"ig_exec_output_{datetime.now().strftime('%Y-%m-%d--%H:%M:%S.%f')}"
        )
        os.makedirs(self.intp_output_dir, exist_ok=True)
        os.makedirs(self.exec_output_dir, exist_ok=True)

    def extract_program(
        self, result: dict, program: str, idx1=0, idx2=0, mutation_pattern="uk"
    ) -> str:
        """
        Extract the program from the result.
        """
        program = program.split("\n")
        program = [line for line in program if not line.startswith("```")]
        # if program[0].startswith("```") and program[-1].startswith("```"):
        #     # Remove the code block markers
        #     program = program[1:-1]
        program = "\n".join(program)

        # Save the program into tmp dir
        syntax_type = result.get("syntax-type", "antlr")
        impl_lang = result["impl-language"]
        if impl_lang != "Java" or self.task != "ig":
            out_filename = f"{impl_lang}_{syntax_type}_{mutation_pattern}_{idx2}.{self.suf_map[impl_lang]}"
        else:
            out_filename = "Interpreter.java"
        output_file_path = self.intp_output_dir / f"{idx1}"
        os.makedirs(output_file_path, exist_ok=True)
        output_file_path = output_file_path / out_filename
        with open(output_file_path, "w") as f:
            f.write(program)

        return output_file_path

    def copy_to_dest(self, from_path: str, to_path: str) -> None:
        """
        Copy the file from one path to another.
        Overwrite the destination file if it exists.
        """
        if not os.path.exists(from_path):
            raise FileNotFoundError(f"Source file does not exist: {from_path}")
        shutil.copy(from_path, to_path)

    def _get_build_cmd(self, intp_path: str) -> str:
        """
        Build the command to compile the interpreter.
        """
        intp_dir = intp_path.parent
        (inpt_prefix, inpt_suffix) = intp_path.name.split(".")
        if self.task == "ig":
            if inpt_suffix == "java":
                cmd = f"javac {intp_path}"
            elif inpt_suffix == "rs":
                cmd = f"rustc {intp_path} -o {intp_dir / inpt_prefix}"
            elif inpt_suffix == "go":
                cmd = f"go build -o {intp_dir / inpt_prefix} {intp_path}"
            else:
                raise ValueError(f"Unsupported impl lang suffix: {inpt_suffix}")
        elif self.task == "iga" or self.task == "igaf":
            if inpt_suffix == "java":
                cmd = f"javac -cp {Macros.antlr_jar_path} {intp_dir / '*.java'}"
            else:
                raise ValueError(f"Unsupported impl lang suffix: {inpt_suffix}")
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        return cmd

    def _get_run_cmd(self, intp_path: str, program_path: str) -> str:
        """
        Build the command to run the interpreter with the given program.
        """
        intp_dir = intp_path.parent
        (inpt_prefix, inpt_suffix) = intp_path.name.split(".")
        if self.task == "ig":
            if inpt_suffix == "py":
                cmd = f"python3 {intp_path} {program_path}"
            elif inpt_suffix == "java":
                cmd = f"cd {intp_dir} && java {inpt_prefix} {program_path}"
            elif inpt_suffix == "go":
                cmd = f"{intp_dir / inpt_prefix} {program_path}"
            elif inpt_suffix == "rs":
                cmd = f"{intp_dir / inpt_prefix} {program_path}"
            else:
                raise ValueError(f"Unsupported impl lang suffix: {inpt_suffix}")
        elif self.task == "iga" or self.task == "igaf":
            if inpt_suffix == "py":
                cmd = f"python3 {intp_dir / 'main.py'} {program_path}"
            elif inpt_suffix == "java":
                cmd = f"java -cp {intp_dir}:{Macros.antlr_jar_path} Main {program_path}"
            else:
                raise ValueError(f"Unsupported impl lang suffix: {inpt_suffix}")
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        return cmd

    def collect_feedback(
        self, intp_path: Path, idx, mutation_pattern, vp_states, ivp_states
    ) -> None:
        suffix = intp_path.suffix
        if suffix == ".py":
            impl_lang = "Python"
        elif suffix == ".java":
            impl_lang = "Java"
        else:
            raise ValueError(f"Unsupported impl lang suffix: {suffix}")

        feedback = {
            "idx": idx,
            "mutation_pattern": mutation_pattern,
            "impl_lang": impl_lang,
            "intp_path": str(intp_path),
            "intp_program": su.io.load(intp_path, fmt=su.io.Fmt.txt),
        }

        for i, ve in enumerate(Macros.valid_examples):
            for vp_state in vp_states:
                if ve == vp_state["program_name"]:
                    feedback[f"valid_example_{i}"] = vp_state
                    break
        for i, ive in enumerate(Macros.invalid_examples):
            for ivp_state in ivp_states:
                if ive == ivp_state["program_name"]:
                    feedback[f"invalid_example_{i}"] = ivp_state
                    break

        return feedback

    def interpret_program(
        self, intp_path: Path, idx1=0, idx2=0, mutation_pattern="uk"
    ) -> Tuple[dict, dict]:
        logger.info(f"Executing interpreter at {intp_path} ...")
        vp_states = []
        ivp_states = []

        def _collect_results(programs, result, is_build_failed, is_timeout):
            return [
                {
                    "program_name": prog.name,
                    "stdout": result.stdout if result else "",
                    "stderr": result.stderr if result else "",
                    "exit_code": result.returncode if result else -1,
                    "is_build_failed": is_build_failed,
                    "is_timeout": is_timeout,
                }
                for prog in programs
            ]

        def _run_programs(programs, results_list):
            for prog in programs:
                run_cmd = self._get_run_cmd(intp_path, prog)
                logger.info(f"Running command: {run_cmd}")
                try:
                    run_res = su.bash.run(run_cmd, timeout=5)
                    logger.info(
                        f"Program: {prog.name}, Exit code: {run_res.returncode}"
                    )
                    results_list.append(
                        {
                            "program_name": prog.name,
                            "stdout": run_res.stdout,
                            "stderr": run_res.stderr,
                            "exit_code": run_res.returncode,
                            "is_build_failed": False,
                            "is_timeout": False,
                        }
                    )
                except su.bash.TimeoutExpired:
                    logger.warning(f"Run command timed out: {run_cmd}")
                    results_list.append(
                        {
                            "program_name": prog.name,
                            "stdout": "",
                            "stderr": "",
                            "exit_code": -1,
                            "is_build_failed": False,
                            "is_timeout": True,
                        }
                    )

        COMPILE_PLS = [".java", ".rs", ".go"]
        if mutation_pattern == "ks":
            valid_imp_dir = Macros.valid_imp_mk_ks_dir
            invalid_imp_dir = Macros.invalid_imp_mk_ks_dir
        elif mutation_pattern == "ko":
            valid_imp_dir = Macros.valid_imp_mk_ko_dir
            invalid_imp_dir = Macros.invalid_imp_mk_ko_dir
        elif mutation_pattern == "uk":
            valid_imp_dir = Macros.valid_imp_uk_dir
            invalid_imp_dir = Macros.invalid_imp_uk_dir
        else:
            raise ValueError(f"Unsupported mutation pattern: {mutation_pattern}")
        # If language needs building
        if intp_path.suffix in COMPILE_PLS:
            build_cmd = self._get_build_cmd(intp_path)
            try:
                logger.info(f"Building interpreter with command: {build_cmd}")
                build_res = su.bash.run(build_cmd, timeout=5)
                if build_res.returncode != 0:
                    logger.warning(
                        f"Build failed for {intp_path.name} with exit code {build_res.returncode}"
                    )
                    vp_states = _collect_results(
                        valid_imp_dir.iterdir(), build_res, True, False
                    )
                    ivp_states = _collect_results(
                        invalid_imp_dir.iterdir(), build_res, True, False
                    )
                    feedback = self.collect_feedback(
                        intp_path, idx1, mutation_pattern, vp_states, ivp_states
                    )
                    return (vp_states, ivp_states, feedback)
            except su.bash.TimeoutExpired:
                logger.warning(f"Build command timed out: {build_cmd}")
                vp_states = _collect_results(valid_imp_dir.iterdir(), None, True, True)
                ivp_states = _collect_results(
                    invalid_imp_dir.iterdir(), None, True, True
                )
                feedback = self.collect_feedback(
                    intp_path, idx1, mutation_pattern, vp_states, ivp_states
                )
                return (vp_states, ivp_states, feedback)

        # Otherwise, run the interpreter
        _run_programs(valid_imp_dir.iterdir(), vp_states)
        _run_programs(invalid_imp_dir.iterdir(), ivp_states)

        exec_output_file = (
            self.exec_output_dir
            / f"results-{intp_path.suffix[1:]}-{mutation_pattern}-{idx1}-{idx2}.json"
        )
        su.io.dump(exec_output_file, {"valid": vp_states, "invalid": ivp_states})
        feedback = self.collect_feedback(
            intp_path, idx1, mutation_pattern, vp_states, ivp_states
        )
        return (vp_states, ivp_states, feedback)

    @staticmethod
    def _compute_loc(pgm_path: str) -> int:
        cloc_cmd = f"cloc {pgm_path} --quiet --csv | awk -F',' 'NR==2{{print $5}}'"
        try:
            loc_output = su.bash.run(cloc_cmd)
            loc = int(loc_output.stdout.strip())
        except Exception:
            raise RuntimeError("Error running cloc command")
        return loc

    @staticmethod
    def _determine_valid_state(valid_state: dict) -> str:
        assert not valid_state["is_build_failed"], (
            "Build should not fail if reach this point."
        )
        exit_code = valid_state["exit_code"]
        stdout = valid_state["stdout"]
        stderr = valid_state["stderr"]
        ans_path = Macros.valid_imp_ans_dir / valid_state["program_name"].replace(
            ".imp", ".txt"
        )

        if valid_state["is_timeout"]:
            return "valid_timeout"
        elif exit_code == 0:
            assert ans_path.exists(), f"Ans file should exist: {ans_path}"
            with open(ans_path, "r") as ans_file:
                expected_output = ans_file.read().strip()
            try:
                actual_output = json.loads(stdout)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from stdout: {stdout}")
                return "valid_fail_semantic_w"
            if actual_output == json.loads(expected_output):
                return "valid_pass"
            else:
                return "valid_fail_semantic_w"
        elif "Syntax Error" in stdout or "Syntax Error" in stderr:
            return "valid_fail_syntax"
        elif "Semantic Error" in stdout or "Semantic Error" in stderr:
            return "valid_fail_semantic_r"
        else:
            return "valid_fail_other"

    @staticmethod
    def _determine_invalid_state(invalid_state: dict) -> str:
        assert not invalid_state["is_build_failed"], (
            "Build should not fail if reach this point."
        )

        exit_code = invalid_state["exit_code"]
        stdout = invalid_state["stdout"]
        stderr = invalid_state["stderr"]

        if invalid_state["is_timeout"]:
            return "invalid_timeout"
        elif exit_code == 0:
            return "invalid_fail_semantic"
        elif "Semantic Error" in stdout or "Semantic Error" in stderr:
            return "invalid_pass"
        elif "Syntax Error" in stdout or "Syntax Error" in stderr:
            return "invalid_fail_syntax"
        else:
            return "invalid_fail_other"

    @staticmethod
    def _compute_metrics_inner(
        pgm: str, valid_states: list, invalid_states: list
    ) -> dict:
        res = {
            "loc": IGResultsHelper._compute_loc(pgm),
            "compilation_pass": True,
            "valid_pass": 0,
            "valid_fail_syntax": 0,
            "valid_fail_semantic_w": 0,
            "valid_fail_semantic_r": 0,
            "valid_fail_other": 0,
            "valid_timeout": 0,
            "invalid_pass": 0,
            "invalid_fail_semantic": 0,
            "invalid_fail_syntax": 0,
            "invalid_fail_other": 0,
            "invalid_timeout": 0,
        }
        if valid_states[0]["is_build_failed"]:
            assert invalid_states[0]["is_build_failed"], "Build should fail for both."
            res["compilation_pass"] = False
            return res
        for valid_state, invalid_state in zip(valid_states, invalid_states):
            valid_res = IGResultsHelper._determine_valid_state(valid_state)
            invalid_res = IGResultsHelper._determine_invalid_state(invalid_state)
            res[valid_res] += 1
            res[invalid_res] += 1

        return res

    @staticmethod
    def compute_metrics(analysis):
        metrics = defaultdict(list)
        scores = {}

        for impl_lang in analysis.pgms.keys():
            pgms = analysis.pgms[impl_lang]
            vs_lists = analysis.valid_pgm_run_states[impl_lang]
            ivs_lists = analysis.invalid_pgm_run_states[impl_lang]

            for pgm, vp_states, ivp_states in zip(pgms, vs_lists, ivs_lists):
                metrics[impl_lang].append(
                    IGResultsHelper._compute_metrics_inner(pgm, vp_states, ivp_states)
                )

        return metrics, scores
