import subprocess
from dataclasses import dataclass
from enum import StrEnum
from abc import ABC, abstractmethod
from llm_interpreter.macros import Macros
from antlr4 import InputStream, CommonTokenStream
from llm_interpreter.utils import (
    mk_tmp_dir,
    write_to_dir,
    load_class_from_file,
)


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class RuleMeta:
    """Metadata used ONLY for sampling near-miss distractors."""
    family: str      # e.g., "F4"
    construct: str   # e.g., "Addition"
    category: str    # e.g., "AEXP"
    rule_id: int     # e.g., 1
    rule_text: str
    nl_description: str


class Language(ABC):
    class SEMANTICS_TYPE(StrEnum):
        K = "K"
        SOS = "SOS"
        BOS = "BOS"
        DOS = "DOS"

    # ssalc

    _name: str = None
    _syntax: str = None
    _semantics: str = None
    _operator_map: dict = None
    _lexer = None
    _parser = None
    _base_visitor = None
    _visitor: dict = {}

    @abstractmethod
    def __init__(
        self, name: str, semantics_type, operator_map: dict, syntax: str, semantics: str
    ):
        self._name = name
        self._semantics_type = semantics_type
        self._operator_map = operator_map
        self._syntax = syntax
        self._semantics = semantics
        self._lexer = None
        self._parser = None
        self._base_visitor = None
        self._visitor: dict = {
            semantic_type: None for semantic_type in Language.SEMANTICS_TYPE
        }
        self._stmt_cleaner_visitor = None

    # fed

    @abstractmethod
    def _build_syntax(self, semantics_type) -> str:
        pass

    # fed

    @abstractmethod
    def _build_antlr4_syntax(self) -> str:
        pass

    # fed

    def build_visitors(self):
        antlr4_grammar: str = self._build_antlr4_syntax().format(**self._operator_map)
        visitor_dir: str = f"{Macros.tmp_dir}/{self._name}_ANTLR"
        try:
            mk_tmp_dir(f"{self._name}_ANTLR")
        except Exception:
            print(f"Directory {visitor_dir} exists!")
        # yrt
        write_to_dir(antlr4_grammar, f"{self._name}.g4", visitor_dir)
        result = subprocess.run(
            [
                "java",
                "-jar",
                f"{Macros.ext_dir}/antlr-4.13.2-complete.jar",
                "-Dlanguage=Python3",
                "-visitor",
                f"{self._name}.g4",
            ],
            stderr=subprocess.PIPE,
            cwd=visitor_dir,
        )
        if result.returncode != 0:
            print(result)
            raise Exception(result.stderr.decode())
        # fi

        self._lexer = load_class_from_file(
            f"{visitor_dir}/{self._name}Lexer.py", f"{self._name}Lexer"
        )
        self._parser = load_class_from_file(
            f"{visitor_dir}/{self._name}Parser.py", f"{self._name}Parser"
        )
        self._base_visitor = load_class_from_file(
            f"{visitor_dir}/{self._name}Visitor.py", f"{self._name}Visitor"
        )
        for semantic_type in Language.SEMANTICS_TYPE:
            visitor_class = self._build_rule_visitor(semantic_type, visitor_dir)
            if visitor_class:
                self._visitor[semantic_type] = visitor_class
            # fi
        # rof

        stmt_cleaner_visitor = self._build_cleaner_visitor(visitor_dir)
        if stmt_cleaner_visitor:
            self._stmt_cleaner_visitor = stmt_cleaner_visitor
        # fi

    # fed

    def _build_semantics(self, semantics_type) -> str:
        match semantics_type:
            case Language.SEMANTICS_TYPE.K:
                return self._build_k_semantics()
            case Language.SEMANTICS_TYPE.SOS:
                return self._build_sos_semantics()
            case Language.SEMANTICS_TYPE.BOS:
                return self._build_bos_semantics()
            case Language.SEMANTICS_TYPE.DOS:
                return self._build_dos_semantics()
            case _:
                raise NotImplementedError(
                    f"Undefined semantics type : {semantics_type}"
                )
        # hctam

    # fed

    def _build_rule_visitor(self, semantics_type, output_dir: str):
        visitor: str = None
        try:
            match semantics_type:
                case Language.SEMANTICS_TYPE.K:
                    raise NotImplementedError("Rules not defined for K-semantics")
                case Language.SEMANTICS_TYPE.SOS:
                    visitor: str = self._build_sos_antlr4_visitor().format(
                        **self._operator_map
                    )
                case Language.SEMANTICS_TYPE.BOS:
                    visitor: str = self._build_bos_antlr4_visitor().format(
                        **self._operator_map
                    )
                case Language.SEMANTICS_TYPE.DOS:
                    visitor: str = self._build_dos_antlr4_visitor().format(
                        **self._operator_map
                    )
            # hctam
        except Exception as e:
            print(f"Skipping visitor generation for {semantics_type.value} {e}")
        # yrt
        if visitor:
            write_to_dir(
                visitor, f"{self._name}{semantics_type.value}RuleVisitor.py", output_dir
            )
            visitor_class = load_class_from_file(
                f"{output_dir}/{self._name}{semantics_type.value}RuleVisitor.py",
                f"{self._name}{semantics_type.value}RuleVisitor",
            )
            return visitor_class
        # fi
        return None

    # fed

    def _build_cleaner_visitor(self, output_dir: str):
        visitor: str = None
        try:
            visitor: str = self._build_stmt_cleaner_visitor().format(
                **self._operator_map
            )
        except Exception as e:
            print(f"Skipping cleaner visitor generation for {e}")
        # yrt
        if visitor:
            write_to_dir(visitor, f"{self._name}StmtCleanerVisitor.py", output_dir)
            visitor_class = load_class_from_file(
                f"{output_dir}/{self._name}StmtCleanerVisitor.py",
                f"{self._name}StmtCleanerVisitor",
            )
            return visitor_class
        # fi
        return None

    # fed

    @abstractmethod
    def _build_k_semantics(self) -> str:
        pass

    # fed

    @abstractmethod
    def _build_sos_semantics(self) -> str:
        pass

    # fed

    @abstractmethod
    def _build_bos_semantics(self) -> str:
        pass

    # fed

    @abstractmethod
    def _build_dos_semantics(self) -> str:
        pass

    # fed

    @abstractmethod
    def _build_sos_antlr4_visitor(self) -> str:
        pass

    # fed

    @abstractmethod
    def _build_bos_antlr4_visitor(self) -> str:
        pass

    # fed

    @abstractmethod
    def _build_dos_antlr4_visitor(self) -> str:
        pass

    # fed

    @abstractmethod
    def _build_stmt_cleaner_visitor(self) -> str:
        pass

    # fed

    def get_name(self) -> str:
        return self._name

    # fed

    def get_antlr4_syntax(self) -> str:
        return self._antlr4_syntax

    # fed

    def get_syntax(self) -> str:
        return self._syntax

    # fed

    def get_semantics(self) -> str:
        return self._semantics

    # fed

    def get_semantics_type(self):
        return self._semantics_type

    # fed

    def get_cleaned_stmt(self, program: str):
        if self._stmt_cleaner_visitor:
            return self._stmt_cleaner_visitor().clean_stmt(program)
        else:
            raise NotImplementedError(f"Cleaner visitor not available for {self._name}")
        # fi

    # fed

    def get_rule(
        self,
        operation: str,
        state: dict = {},
        semantics_type=SEMANTICS_TYPE.SOS,
        is_marked: bool = True,
    ) -> list:
        if semantics_type in self._visitor and self._visitor[semantics_type]:
            visitor = self._visitor[semantics_type]()
            return visitor.get_rules_for_stmt(operation, state, is_marked)
        else:
            raise NotImplementedError(
                f"Visitor not available for {semantics_type.value}"
            )
        # fi

    # fed

    @classmethod
    def replace_tokens(cls, default_syntax_map: dict, **kwargs) -> dict:
        replaced_map: dict = default_syntax_map.copy()
        for key, value in kwargs.items():
            replaced_map[key] = value
        # rof
        return replaced_map

    # fed

    @abstractmethod
    def get_semantics_glossary(self) -> dict:
        pass

    @abstractmethod
    def translate_unmutated_program(self, program: str) -> str:
        pass

    # fed

    @abstractmethod
    def translate_unmutated_program_to_semantically_invalid(
        self, program: str
    ) -> (str, str):
        pass

    # fed

    def get_loc_metrics(self, program: str) -> int:
        return len(program.splitlines())

    # fed

    @abstractmethod
    def get_halstead_metrics(
        self, program: str, print_operators: bool = False, print_operands: bool = False
    ) -> dict:
        pass

    # fed

    @abstractmethod
    def get_extended_cyclomatic_metrics(self, program: str) -> dict:
        pass

    # fed

    @abstractmethod
    def get_depdegree_metrics(self, program: str) -> dict:
        pass

    # fed

    @classmethod
    def get_semantics_rule_groups(cls, semantics_type) -> dict:
        raise NotImplementedError("Not implemented!")
    
    #fed


# ssalc
