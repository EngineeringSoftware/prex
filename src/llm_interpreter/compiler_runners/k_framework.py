import re
import copy
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List
from llm_interpreter.language import Language


class KFrameworkError(Exception):
    def __init__(self, error_msg: str):
        super().__init__(error_msg)

    # fed


# ssalc


@dataclass
class ExecutionState:
    line_number: int
    rule: List[str]
    state: Dict[str, Any]
    stmt_rule: List[str]
    stmt: List[str]
    control_stack: List[str]


class KResult:
    _execution_trace: List[ExecutionState] = []
    _final_state: dict = {}
    _k_output: str = None
    _program: str = None
    _language: str = None
    _k_framework: str = None

    def __init__(
        self,
        program: str,
        k_output: str,
        language: Language,
        k_framework,
    ):
        self._execution_trace: list = []
        self._final_state: dict = {}
        self._k_output = k_output
        self._program = program
        self._language = language
        self._k_framework = k_framework

    # fed

    def get_final_state(self) -> dict:
        return self._final_state

    # fed

    def get_execution_trace(self) -> list:
        return self._execution_trace

    # fed

    def check_k_output(
        self,
    ):
        # check if the program completed execution
        exec_status: str = (self._k_output.split("<k>")[1]).split("</k>")[0]
        if not exec_status.strip() == ".K":
            raise KFrameworkError(
                f" Program did not execute completely. Execution stopped at {exec_status}"
            )
        # fi

    # fed

    def parse_final_state(
        self,
    ) -> dict:
        var_values: list = []
        var_map: dict = {}
        # Get final state
        var_values = (self._k_output.split("<state>")[1]).split("</state>")[0]
        var_values = var_values.strip().splitlines()
        for var in var_values:
            var_line: list = var.strip().split(" |-> ")
            var_map[var_line[0]] = int(var_line[1])
        # rof
        return var_map

    # fed

    def get_node_as_list(self, node: str) -> list:
        content_list: list = []
        content = (self._k_output.split(f"<{node}>")[1]).split(f"</{node}>")[0]
        content = content.lstrip()
        blocks = re.findall(r"ListItem\s*\((.*?)\)\n", content, re.DOTALL)
        for block in blocks:
            d = {}
            lines = block.strip().splitlines()
            for line in lines:
                # for states
                if "|->" in line:
                    key, value = map(str.strip, line.split("|->"))
                    d[key] = int(value)
                # fi
            # rof
            if node == "states":
                content_list.append(d)
            else:
                content_list.append("".join(lines))
            # fi
        # rof
        return content_list

    # fed

    def get_rules_as_list(
        self,
        states: List[Dict],
        lines: List[str],
        statements: List[str],
        semantics_type: Language.SEMANTICS_TYPE,
    ) -> List[List[str]]:
        """
        This function should convert the statements: List[str] to a list of rules: List[List[str]].

        Args:
            statements: get_node_as_list("rules") List[str]

        Returns:
            List[List[str]]: A list of rules, where each rule list corresponds to a statement.
        """
        rules: list = []
        stmts: list = []
        states_before_exec: list = [{}] + states[0 : len(states) - 1]
        control_stacks: list = []
        n: int = len(statements)
        for i in range(n):
            program_lines = self._program.split("\n")
            program_lines[int(lines[i]) - 1] = f"@ {program_lines[int(lines[i]) - 1]}"
            marked_program = "\n".join(program_lines)
            [cleaned_statement, control_stack] = self._language.get_cleaned_stmt(
                marked_program
            )
            if semantics_type == Language.SEMANTICS_TYPE.K:
                cleaned_k_statement = cleaned_statement.replace("@", "")
                initial_state: dict = copy.deepcopy(states_before_exec[i])
                initial_state["ble"] = 0
                k_output: str = None
                try:
                    k_output: str = self._k_framework.run_program_stmts(
                        cleaned_k_statement, initial_state
                    )
                except Exception:
                    rules.append([])
                    stmts.append(None)
                    control_stacks.append(None)
                    continue
                # yrt
                stmt_rules: list = self._k_framework.get_k_result(
                    "", k_output, self._language
                ).get_node_as_list("rules")
                stmt_rules = [
                    stmt_rule.replace('"', "")
                    for stmt_rule in stmt_rules
                    if stmt_rule != '"BREAK"'
                ]
                rules.append(stmt_rules)
            else:
                rules.append(
                    self._language.get_rule(
                        cleaned_statement, states_before_exec[i], semantics_type
                    )[0]
                )
            # fi
            stmts.append(cleaned_statement)
            control_stacks.append(control_stack)
        # rof
        return rules, stmts, control_stacks

    # fed

    def parse_execution_trace(
        self,
        semantics_type: Language.SEMANTICS_TYPE,
        gen_rules_for_srp: bool = False,
    ):
        try:
            self.check_k_output()
            self._final_state = self.parse_final_state()

            # Get rules: List[str] from statements
            lines: list = self.get_node_as_list("lines")
            statements: list = self.get_node_as_list("stmts")
            states: list = self.get_node_as_list("states")[1:] + [self._final_state]
            if semantics_type == Language.SEMANTICS_TYPE.K:
                unprocessed_rules: list = self.get_node_as_list("rules")
                unprocessed_rules.pop(0)
                rules: list = []
                stmt_rules: list = []
                for rule in unprocessed_rules:
                    if rule == '"BREAK"':
                        rules.append(stmt_rules)
                        stmt_rules = []
                    else:
                        stmt_rules.append(rule.replace('"', ""))
                    # fi
                # rof
                rules.append(stmt_rules)
            else:
                rules = [""] * len(lines)
            # fi

            if gen_rules_for_srp:
                [srp_stmt_rules, stmts, control_stack] = self.get_rules_as_list(
                    states, lines, statements, semantics_type
                )
            else:
                srp_stmt_rules = [""] * len(lines)
                stmts = [""] * len(lines)
                control_stack = [""] * len(lines)
            # fi

            if not (len(lines) == len(statements) and len(lines) == len(states)):
                raise KFrameworkError(
                    f"Mismatch in length of lines, statements and states in {self._k_output}"
                )
            # fi
            n: int = len(lines)
            for i in range(0, n):
                self._execution_trace.append(
                    ExecutionState(
                        line_number=int(lines[i]),
                        rule=rules[i],
                        stmt_rule=srp_stmt_rules[i],
                        state=states[i],
                        stmt=stmts[i],
                        control_stack=control_stack[i],
                    )
                )
            # rof
        except Exception as e:
            raise KFrameworkError(f"{e}")
        # yrt

    # fed


# ssalc


class KFramework:
    language_file: str = None

    def compile_k_specification(self, k_file: str, output_dir: str) -> None:
        result = subprocess.run(
            ["kompile", "--gen-glr-bison-parser", "-o", output_dir, k_file],
            stderr=subprocess.PIPE,
        )
        self.language_file = output_dir
        if result.returncode != 0:
            raise KFrameworkError(result.stderr.decode())
        # fi

    # fed

    def run_program_stmts(
        self, program_stmts: str, initial_state: dict = {}, timeout: int = 20
    ) -> str:
        state: str = ".Map"
        if len(initial_state) > 0:
            for key, value in initial_state.items():
                state = f"{key} |-> {value} {state}"
            # rof
        # fi
        try:
            result = subprocess.run(
                [
                    "krun",
                    "-d",
                    self.language_file,
                    f"-cPGM={program_stmts}",
                    f"-cSTATE={state}",
                    "-o",
                    "pretty",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise KFrameworkError(result.stderr.decode())
            else:
                return result.stdout.decode()
            # fi
        except subprocess.TimeoutExpired as err:
            raise KFrameworkError(f"{err}")
        # yrt

    # fed

    def run_program(
        self, program_file: str, initial_state: dict = {}, timeout: int = 20
    ) -> str:
        state: str = ".Map"
        if len(initial_state) > 0:
            for key, value in initial_state.items():
                state = f"{key} |-> {value} {state}"
            # rof
        # fi
        try:
            result = subprocess.run(
                [
                    "krun",
                    "-d",
                    self.language_file,
                    f"-cSTATE={state}",
                    "-o",
                    "pretty",
                    program_file,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise KFrameworkError(result.stderr.decode())
            else:
                return result.stdout.decode()
            # fi
        except subprocess.TimeoutExpired as err:
            raise KFrameworkError(f"{err}")
        # yrt

    # fed

    def parse_k_framework_output(
        self,
        program: str,
        k_output: str,
        language: Language,
        semantics_type: Language.SEMANTICS_TYPE,
        gen_rules_for_srp: bool = False,
    ) -> dict:
        k_result = KResult(program, k_output, language, self)
        k_result.parse_execution_trace(semantics_type, gen_rules_for_srp)
        return k_result

    # fed

    def get_k_result(
        self,
        program: str,
        k_output: str,
        language: Language,
    ) -> dict:
        return KResult(program, k_output, language, self)

    # fed


# ssalc
