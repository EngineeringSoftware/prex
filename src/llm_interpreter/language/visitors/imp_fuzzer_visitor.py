import string
import random
from enum import StrEnum, auto
from typing import List, Optional, Set, Union
from llm_interpreter.antlr4_parsers import IMPBASEParser, IMPBASEVisitor


def pick_var_use_injection_var_name(
    declared: Union[Set[str], List[str]],
    *,
    seed: Optional[int] = None,
) -> str:
    """
    Pick a random lowercase name (e.g. xyzabc) that is not already declared.
    """
    declared_set = set(declared)
    rng = random.Random(seed) if seed is not None else random
    for _ in range(128):
        name = "".join(rng.choice(string.ascii_lowercase) for _ in range(6))
        if name not in declared_set:
            return name

    for idx in range(10_000):
        name = f"xyz{idx:04d}"
        if name not in declared_set:
            return name

    raise RuntimeError("Could not generate a use-before-declare injection variable name.")


class IMPFuzzerVisitor(IMPBASEVisitor):
    class FUZZING_TYPES(StrEnum):
        DIVIDE_BY_ZERO = auto()
        MODULO_ZERO = auto()
        BREAK_OUTSIDE_LOOP = auto()
        CONTINUE_OUTSIDE_LOOP = auto()
        VAR_USE_BEFORE_DECLARE = auto()

    # ssalc

    _indent: int = -4
    _declared_vars: list = []
    _fuzzing_types: set = None
    _fuzz_on: bool = False
    _zeroed_var: str = None
    _num_stmts_to_fuzz: int = 1

    def __init__(
        self,
        num_stmts_to_fuzz: int = 1,
        fuzzing_types: list = [f_type for f_type in FUZZING_TYPES],
        target_stmt_index: Optional[int] = None,
        target_assign_site_id: Optional[int] = None,
        target_assign_site_ids: Optional[List[int]] = None,
        exact_target_stmt: bool = False,
        literal_zero_divisor_probability: float = 0.3,
    ):
        self._indent = -4
        self._declared_vars = []
        self._fuzzing_types = fuzzing_types
        self._selected_fuzzing_type = None
        self._num_stmts_to_fuzz = num_stmts_to_fuzz
        self._target_stmt_index = target_stmt_index
        if target_assign_site_ids is not None:
            self._target_assign_site_ids = target_assign_site_ids
        elif target_assign_site_id is not None:
            self._target_assign_site_ids = [target_assign_site_id]
        else:
            self._target_assign_site_ids = None
        self._exact_target_stmt = exact_target_stmt
        self._literal_zero_divisor_probability = literal_zero_divisor_probability
        self._current_site_use_literal = False
        self._zeroed_var = None
        self._zero_prepended_sites: set[int] = set()
        self._assign_site_counter = 0
        self._fuzz_on = False
        self._inside_loop = False

    # fed

    def get_selected_fuzzing_type(self):
        return self._selected_fuzzing_type

    # fed

    def _pick_zeroed_var(self, assign_lhs: Optional[str] = None) -> Optional[str]:
        if not self._declared_vars:
            return None
        candidates = [var for var in self._declared_vars if var != assign_lhs]
        if not candidates:
            candidates = self._declared_vars
        return random.choice(candidates)

    # fed

    def _divisor_for_zero_error(self) -> str:
        if self._exact_target_stmt:
            if self._current_site_use_literal or self._zeroed_var is None:
                return "0"
            return self._zeroed_var
        if self._zeroed_var is not None:
            return random.choice([self._zeroed_var, "0"])
        return "0"

    # fed

    def _prepare_zero_error_for_assign_site(self, assign_lhs: str) -> None:
        site_id = self._assign_site_counter
        self._current_site_use_literal = (
            random.random() < self._literal_zero_divisor_probability
        )
        self._zeroed_var = None
        if self._current_site_use_literal or site_id in self._zero_prepended_sites:
            return
        self._zeroed_var = self._pick_zeroed_var(assign_lhs)

    # fed

    def visitProgram(self, ctx: IMPBASEParser.ProgramContext):
        self._declared_vars = []
        self._assign_site_counter = 0
        self._zero_prepended_sites = set()
        self._selected_fuzzing_type = random.choice(self._fuzzing_types)
        if ctx.stmt_list():
            return self.visit(ctx.stmt_list())
        return ""

    # fed

    def visitDeclStmt(self, ctx: IMPBASEParser.DeclStmtContext):
        declarations: list = []
        if ctx.ids():
            ids: list = self.visit(ctx.ids())
            declarations = [f"int {id};" for id in ids]
        # fi
        return "\n".join(declarations)

    # fed

    def visitIds(self, ctx: IMPBASEParser.IdsContext):
        ids = [id.getText() for id in ctx.ID()]
        self._declared_vars.extend(ids)
        return ids

    # fed

    def visitAssignStmt(self, ctx: IMPBASEParser.AssignStmtContext):
        return f"{ctx.ID().getText()} = {self.visit(ctx.aexp())};"

    # fed

    def visitIfElseStmt(self, ctx: IMPBASEParser.IfElseStmtContext):
        condition: str = self.visit(ctx.bexp())
        condition = f"({condition})" if condition[0] != "(" else condition
        if_body: str = self.visit(ctx.stmt_list(0))
        else_body: str = self.visit(ctx.stmt_list(1))
        return f"if{condition}\n{if_body}\n{' ' * self._indent}else\n{else_body};"

    # fed

    def visitIfStmt(self, ctx: IMPBASEParser.IfStmtContext):
        condition: str = self.visit(ctx.bexp())
        condition = f"({condition})" if condition[0] != "(" else condition
        body: str = self.visit(ctx.stmt_list())
        return f"if{condition}\n{body}\n{' ' * self._indent}else\n{' ' * self._indent}{{\n\n{' ' * self._indent}}};"

    # fed

    def visitWhileStmt(self, ctx: IMPBASEParser.WhileStmtContext):
        condition: str = self.visit(ctx.bexp())
        condition = f"({condition})" if condition[0] != "(" else condition
        self._inside_loop = True
        body: str = self.visit(ctx.stmt_list())
        self._inside_loop = False
        return f"while{condition}\n{body};"

    # fed

    def visitHaltStmt(self, ctx: IMPBASEParser.HaltStmtContext):
        return "halt;"

    # fed

    def visitContinueStmt(self, ctx: IMPBASEParser.ContinueStmtContext):
        return "continue;"

    # fed

    def visitBreakStmt(self, ctx: IMPBASEParser.BreakStmtContext):
        return "break;"

    # fed

    def visitStmt_list(self, ctx: IMPBASEParser.Stmt_listContext):
        self._indent += 4
        body: list = []
        for id, stmt in enumerate(ctx.stmt()):
            # Introduce randomness in selection of stmts for fuzzing but fuzz all stmts
            # if the number of remaining stmts is less than or equal to the number of
            # stmts to be fuzzed.
            if self._exact_target_stmt:
                if self._target_assign_site_ids is not None:
                    if isinstance(stmt, IMPBASEParser.AssignStmtContext):
                        self._fuzz_on = (
                            self._assign_site_counter in self._target_assign_site_ids
                        )
                    else:
                        self._fuzz_on = False
                elif self._target_stmt_index is not None:
                    self._fuzz_on = id == self._target_stmt_index
            elif not self._fuzz_on:
                if self._num_stmts_to_fuzz > 0:
                    self._fuzz_on = random.choice([True, False])
                # fi
                if id <= self._num_stmts_to_fuzz:
                    self._fuzz_on = True
                # fi
            # fi

            if self._fuzz_on:
                match self._selected_fuzzing_type:
                    case IMPFuzzerVisitor.FUZZING_TYPES.BREAK_OUTSIDE_LOOP:
                        if not self._inside_loop:
                            body.append(" " * self._indent + "break;")
                            self._fuzz_on = False
                            self._num_stmts_to_fuzz -= 1
                        # fi
                    case IMPFuzzerVisitor.FUZZING_TYPES.CONTINUE_OUTSIDE_LOOP:
                        if not self._inside_loop:
                            body.append(" " * self._indent + "continue;")
                            self._fuzz_on = False
                            self._num_stmts_to_fuzz -= 1
                        # fi
                    case (
                        IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO
                        | IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO
                    ):
                        if isinstance(stmt, IMPBASEParser.AssignStmtContext):
                            site_id = self._assign_site_counter
                            if self._exact_target_stmt:
                                assign_lhs = stmt.ID().getText()
                                self._prepare_zero_error_for_assign_site(assign_lhs)
                                if (
                                    self._zeroed_var is not None
                                    and site_id not in self._zero_prepended_sites
                                ):
                                    body.append(
                                        " " * self._indent
                                        + f"{self._zeroed_var} = 0;"
                                    )
                                    self._zero_prepended_sites.add(site_id)
                            elif site_id not in self._zero_prepended_sites:
                                self._zeroed_var = self._pick_zeroed_var()
                                if self._zeroed_var is not None:
                                    body.append(
                                        " " * self._indent + f"{self._zeroed_var} = 0;"
                                    )
                                    self._zero_prepended_sites.add(site_id)
                # hctam
            # fi
            body.append(" " * self._indent + self.visit(stmt))
            if isinstance(stmt, IMPBASEParser.AssignStmtContext):
                self._assign_site_counter += 1
        # rof

        add_parenthesis = False if self._indent == 0 else True
        self._indent -= 4
        body = "\n".join(body)
        if add_parenthesis:
            return f"{' ' * self._indent}{{\n{body}\n{' ' * self._indent}}}"
        else:
            return body
        # fi

    # fed

    def visitAddsubexp(self, ctx: IMPBASEParser.AddsubexpContext):
        lhs_term: str = self.visit(ctx.muldivexp(0))
        n: int = len(ctx.ADDSUBOP())
        for i in range(n):
            rhs_term: str = self.visit(ctx.muldivexp(i + 1))
            op: str = ctx.ADDSUBOP(i).getText()
            lhs_term = f"({lhs_term} {op} {rhs_term})"
        # rof
        return lhs_term

    # fed

    def visitMuldivexp(self, ctx: IMPBASEParser.MuldivexpContext):
        lhs_term: str = self.visit(ctx.unaryexp(0))
        n: int = len(ctx.MULDIVOP())
        for i in range(n):
            rhs_term: str = self.visit(ctx.unaryexp(i + 1))
            op: str = ctx.MULDIVOP(i).getText()
            lhs_term = f"({lhs_term} {op} {rhs_term})"
        # rof
        if self._fuzz_on:
            match self._selected_fuzzing_type:
                # We increase diversity of the divide/modulo 0 errors by either dividing the expression
                # explicitly with literal 0 or with a declared variable we have zeroed earlier
                case IMPFuzzerVisitor.FUZZING_TYPES.DIVIDE_BY_ZERO:
                    self._fuzz_on = False
                    self._num_stmts_to_fuzz -= 1
                    divisor = self._divisor_for_zero_error()
                    return f"({lhs_term} / {divisor})"
                case IMPFuzzerVisitor.FUZZING_TYPES.MODULO_ZERO:
                    self._fuzz_on = False
                    self._num_stmts_to_fuzz -= 1
                    divisor = self._divisor_for_zero_error()
                    return f"({lhs_term} % {divisor})"
            # hctam
        # fi
        return lhs_term

    # fed

    def visitUnaryexp(self, ctx: IMPBASEParser.UnaryexpContext):
        lhs_term: str = self.visit(ctx.atomexp())
        if ctx.ADDSUBOP():
            op: str = ctx.ADDSUBOP().getText()
            return f"({op}{lhs_term})"
        else:
            return lhs_term
        # fi

    # fed

    def visitLogicalexp(self, ctx: IMPBASEParser.LogicalexpContext):
        lhs_term: str = self.visit(ctx.notexp(0))
        n: int = len(ctx.LOGICALOP())
        for i in range(n):
            rhs_term: str = self.visit(ctx.notexp(i + 1))
            op: str = ctx.LOGICALOP(i).getText()
            lhs_term = f"({lhs_term} {op} {rhs_term})"
        # rof
        return lhs_term

    # fed

    def visitNotexp(self, ctx: IMPBASEParser.NotexpContext):
        lhs_term: str = self.visit(ctx.relexp())
        if ctx.LOGNOT():
            op: str = ctx.LOGNOT().getText()
            return f"({op}{lhs_term})"
        else:
            return lhs_term
        # fi

    # fed

    def visitRelexp(self, ctx: IMPBASEParser.RelexpContext):
        lhs_term: str = self.visit(ctx.boolatomexp(0))
        n: int = len(ctx.RELOP())
        for i in range(n):
            rhs_term: str = self.visit(ctx.boolatomexp(i + 1))
            op: str = ctx.RELOP(i).getText()
            lhs_term = f"({lhs_term} {op} {rhs_term})"
        # rof
        return lhs_term

    # fed

    def _get_random_var(self) -> str:
        return pick_var_use_injection_var_name(self._declared_vars)

    # rof

    def visitAtomexp(self, ctx: IMPBASEParser.AtomexpContext):
        if ctx.LITERAL():
            if (
                self._fuzz_on
                and self._selected_fuzzing_type
                == IMPFuzzerVisitor.FUZZING_TYPES.VAR_USE_BEFORE_DECLARE
            ):
                self._fuzz_on = False
                self._num_stmts_to_fuzz -= 1
                return self._get_random_var()
            return ctx.LITERAL().getText()
        elif ctx.ID():
            if (
                self._fuzz_on
                and self._selected_fuzzing_type
                == IMPFuzzerVisitor.FUZZING_TYPES.VAR_USE_BEFORE_DECLARE
            ):
                self._fuzz_on = False
                self._num_stmts_to_fuzz -= 1
                return self._get_random_var()
            else:
                return ctx.ID().getText()
            # fi
        else:
            out: str = self.visit(ctx.aexp())
            return f"({out})" if out[0] != "(" else out
        # fi

    # fed

    def visitBoolatomexp(self, ctx: IMPBASEParser.BoolatomexpContext):
        if ctx.BOOL():
            return ctx.BOOL().getText()
        elif ctx.aexp():
            return f"{self.visit(ctx.aexp())}"
        elif ctx.bexp():
            return f"({self.visit(ctx.bexp())})"
        # fi

    # fed


# ssalc
