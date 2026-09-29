import sys
import random
import string
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


class ImpFuzzerGenerator:
    def __init__(self):
        self.LOOP_BREAKER_VAR_PREFIX: str = "ble"
        self.LOOP_BREAKER_VAR_COUNT: int = 0
        self.LOOP_BREAKER_VAR_RANGE: dict = defaultdict(int)
        self.MIN_NUM_VARS: int = 5
        self.MAX_NUM_VARS: int = 10
        self.MIN_NUM_STMTS: int = 1
        self.MAX_NUM_STMTS: int = 3
        self.INSIDE_LOOP: int = 0
        self.MIN_BLOCK_DEPTH: int = 3
        self.MAX_BLOCK_DEPTH: int = 5
        self.BLOCK_INDEX: int = 0
        self.INDENT: int = -4
        self.HALT_PROB: float = 0.0
        self.CONTINUE_PROB: float = 0.0
        # Local operator map copied from IMP._DEFAULT_SYNTAX_MAP for expression generation
        self._operator_map: dict = {
            "PLUS_OP": "+",
            "MINUS_OP": "-",
            "MUL_OP": "*",
            "DIV_OP": "/",
            "MOD_OP": "%",
            "LT_OP": "<",
            "LTEQ_OP": "<=",
            "GT_OP": ">",
            "GTEQ_OP": ">=",
            "EQ_OP": "==",
            "NEQ_OP": "!=",
            "AND_OP": "&&",
            "OR_OP": "||",
            "NOT_OP": "!",
        }

        # base mix (assign > while ~ if > continue > break > halt)
        self._BASE_WEIGHTS = {
            "_visit_assign_stmt": 0.4,
            "_visit_while_stmt": 0.3,
            "_visit_if_else_stmt": 0.2,
            "_visit_break_stmt": 0.09,
            "_visit_continue_stmt": 0.005,
            "_visit_halt_stmt": 0.005,
        }

    def _visit_program(self) -> str:
        # Generate sample expressions locally (avoid importing modules using Python 3.10 features)
        arith_expr, _arith_state, _arith_counts = self._generate_arithmetic_expr(
            num_terms=3
        )
        bool_expr, _bool_result, _bool_state, _bool_counts = (
            self._generate_boolean_expr(num_terms=2)
        )

        self.stmt_types: list = [
            self._visit_assign_stmt,
            self._visit_if_else_stmt,
            self._visit_while_stmt,
            self._visit_break_stmt,
            self._visit_continue_stmt,
            self._visit_halt_stmt,
        ]
        self._name_by_func = {f: f.__name__ for f in self.stmt_types}

        # Use simple positive random weights; random.choices supports weights in Python 3.9
        self.stmt_weights: list[float] = [
            random.random() + 0.01 for _ in self.stmt_types
        ]
        num_vars: int = random.randint(self.MIN_NUM_VARS, self.MAX_NUM_VARS)
        self.var_list: list[str] = self._visit_var_list(num_vars)
        decl_stmts: str = "\n".join([f"int {var};" for var in self.var_list])
        assign_stmts: str = "\n".join(
            [self._visit_assign_stmt() for var in self.var_list]
        )
        stmt_list: str = "\n".join(self._visit_stmt_list())
        loop_breaker_decl_stmts: str = "\n".join(
            [f"int {var};" for var in self.LOOP_BREAKER_VAR_RANGE.keys()]
        )
        loop_breaker_assign_stmts: str = "\n".join(
            [f"{var} = {value};" for var, value in self.LOOP_BREAKER_VAR_RANGE.items()]
        )
        return f"{decl_stmts}\n{loop_breaker_decl_stmts}\n{assign_stmts}\n{loop_breaker_assign_stmts}\n{stmt_list}"

    def _visit_var_list(self, num_vars: int) -> list[str]:
        var_set: set[str] = set()
        while len(var_set) < num_vars:
            var_set.add(random.choice(string.ascii_letters))
        return list(var_set)

    def _visit_stmt_list(self) -> list[str]:
        self.BLOCK_INDEX += 1
        self.INDENT += 4
        stmt_list: list[str] = []
        num_stmts: int = random.randint(self.MIN_NUM_STMTS, self.MAX_NUM_STMTS)
        for _ in range(num_stmts):
            stmt_list.append(f"{' ' * self.INDENT}{self._visit_stmt()}")
        self.BLOCK_INDEX -= 1
        self.INDENT -= 4
        return stmt_list

    def _pick_stmt_func(self):
        funcs = self.stmt_types
        names = self._name_by_func

        # 1) start from base weights (assign highest, break lowest)
        w = np.array([self._BASE_WEIGHTS[names[f]] for f in funcs], dtype=float)

        def taper_cosine(d: int) -> float:
            d0, d1 = self.MIN_BLOCK_DEPTH, self.MAX_BLOCK_DEPTH
            if d <= d0:
                return 1.0
            if d >= d1:
                return 0.0
            x = (d - d0) / (d1 - d0)
            # smooth 1 → 0 (no flat spots)
            return float(0.5 * (1.0 + np.cos(np.pi * x)))

        # fed

        # 2) depth taper for control flow once past MIN_BLOCK_DEPTH
        d = int(self.BLOCK_INDEX)
        scale = taper_cosine(d)
        for i, f in enumerate(funcs):
            if f in (self._visit_if_else_stmt, self._visit_while_stmt):
                w[i] *= scale

        # 3) hard masks
        if self.INSIDE_LOOP == 0:
            # break/continue illegal outside loops
            for i, f in enumerate(funcs):
                if f in (self._visit_break_stmt, self._visit_continue_stmt):
                    w[i] = 0.0

        # absolutely forbid new blocks at/after MAX
        if d >= self.MAX_BLOCK_DEPTH:
            for i, f in enumerate(funcs):
                if f in (self._visit_if_else_stmt, self._visit_while_stmt):
                    w[i] = 0.0

        # 4) normalize and sample
        s = w.sum()
        if s <= 0:
            return self._visit_assign_stmt
        p = w / s
        idx = np.random.default_rng().choice(len(funcs), p=p)
        return funcs[idx]

    def _visit_stmt(self) -> str:
        stmt: str = None
        while stmt is None:
            stmt_func = self._pick_stmt_func()
            stmt = stmt_func()
        return stmt

    def _visit_assign_stmt(self) -> str:
        var: str = random.choice(self.var_list)
        aexp: str = self._visit_aexp(
            num_terms=random.randint(3, 6), num_var_terms=random.randint(0, 3)
        )
        return f"{var} = {aexp};"

    def _visit_if_else_stmt(self) -> str:
        if self.BLOCK_INDEX > self.MAX_BLOCK_DEPTH:
            return None
        cond: str = self._visit_bexp()
        then_branch: str = "\n".join(self._visit_stmt_list())
        else_branch: str = "\n".join(self._visit_stmt_list())
        return (
            f"if ({cond}) \n{{ \n{then_branch} \n}} \n  else \n{{ \n{else_branch} \n}};"
        )

    def _make_loop_breaker_var(self):
        init_val: int = -1
        loop_break_stmt: str = ""
        loop_cond_stmt: str = ""
        loop_var: str = f"{self.LOOP_BREAKER_VAR_PREFIX}{self.LOOP_BREAKER_VAR_COUNT}"
        # Choose incrementing/decrementing with 50% probability
        if random.random() < 0.5:
            init_value = random.randint(-10, 10)
            max_value = random.randint(init_value, 20)
            loop_break_stmt = f"{loop_var} = {loop_var} + {random.randint(1, max(round(max_value / 3), 2))};"
            loop_cond_stmt = (
                f"{loop_var} < {max_value}"
                if random.random() < 0.5
                else f"{loop_var} <= {max_value}"
            )
        else:
            # decrementing type
            init_value = random.randint(-10, 10)
            min_value = random.randint(-20, init_value)
            loop_break_stmt = f"{loop_var} = {loop_var} + {random.randint(min(round(min_value / 3), -2), -1)};"
            loop_cond_stmt = (
                f"{loop_var} > {min_value}"
                if random.random() < 0.5
                else f"{loop_var} >= {min_value}"
            )
        # fi
        self.LOOP_BREAKER_VAR_RANGE[loop_var] = init_val
        self.LOOP_BREAKER_VAR_COUNT += 1
        return loop_break_stmt, loop_cond_stmt

    # fed

    def _visit_while_stmt(self) -> str:
        if self.BLOCK_INDEX > self.MAX_BLOCK_DEPTH:
            return None
        [loop_break_stmt, loop_cond_stmt] = self._make_loop_breaker_var()
        cond: str = f"{self._visit_bexp()} && {loop_cond_stmt}"
        self.INSIDE_LOOP += 1
        body: str = "\n".join(self._visit_stmt_list())
        self.INSIDE_LOOP -= 1
        return f"while ({cond}) \n{{ \n{body}\n{loop_break_stmt} \n}};"

    def _visit_break_stmt(self) -> str:
        if self.INSIDE_LOOP > 0:
            return "break;"
        else:
            return None

    def _visit_continue_stmt(self) -> str:
        if self.INSIDE_LOOP > 0 and random.random() < getattr(
            self, "CONTINUE_PROB", 0.0
        ):
            return "continue;"
        else:
            return None

    def _visit_halt_stmt(self) -> str:
        if random.random() < getattr(self, "HALT_PROB", 0.0):
            return "halt;"
        else:
            return None

    # ---------- Expression Generators ----------
    def _generate_arithmetic_expr(
        self,
        num_terms: int = 3,
        num_var_terms: int = 2,
        zero_init: bool = True,
        var_pool: list[str] = None,
    ):
        if var_pool and len(var_pool) > 0:
            var_set = list(set(var_pool))
        else:
            var_set: set = set()
            while len(var_set) < max(1, num_var_terms):
                var_set.add((random.choice(string.ascii_letters)).lower())
            var_set = list(var_set)
        state: dict = {
            var: 1 if zero_init else random.randint(-10, 10) for var in var_set
        }
        # Prefer simpler ops
        ops: list[str] = [
            "PLUS_OP",
            "MINUS_OP",
            "MUL_OP",
            "DIV_OP",
            "MOD_OP",
        ]
        op_weights = {
            "PLUS_OP": 4,
            "MINUS_OP": 3,
            "MUL_OP": 2,
            "DIV_OP": 1,
            "MOD_OP": 1,
        }
        op_count: dict = {op: 0 for op in ops}
        expr_parts: list[str] = []
        var_terms_left: int = min(num_var_terms, len(var_set))
        chosen_op = None
        for i in range(num_terms):
            # choose variable or small literal
            use_var = var_terms_left > 0 and (
                random.random() < 0.6 or (num_terms - i) <= var_terms_left
            )
            if use_var and (
                True
                if chosen_op is None
                else (False if chosen_op in ["DIV_OP", "MOD_OP"] else True)
            ):
                term = random.choice(var_set)
                var_terms_left -= 1
            else:
                term = str(
                    random.randint(
                        0
                        if chosen_op is None
                        else (1 if chosen_op in ["DIV_OP", "MOD_OP"] else 0),
                        9,
                    )
                )
            # small chance of unary negation
            if random.random() < 0.15:
                term = f"(- {term})"
            expr_parts.append(term)
            if i < num_terms - 1:
                # weighted op pick
                population = ops
                weights = [op_weights[o] for o in population]
                chosen_op = random.choices(population, weights=weights, k=1)[0]
                op_count[chosen_op] += 1
                expr_parts.append(self._operator_map[chosen_op])
        expr_str = " ".join(expr_parts)
        if num_terms > 2:
            expr_str = f"({expr_str})"
        return (expr_str, state, op_count)

    def _make_bool_term(self, var_pool: list[str] = None):
        rel_ops: list[str] = [
            "LT_OP",
            "LTEQ_OP",
            "GT_OP",
            "GTEQ_OP",
            "EQ_OP",
            "NEQ_OP",
        ]
        # simpler arithmetic inside comparisons
        [arith_expr1, arith_state1, arith_rule_count1] = self._generate_arithmetic_expr(
            num_terms=random.choice([2, 3]), var_pool=var_pool
        )
        [arith_expr2, arith_state2, arith_rule_count2] = self._generate_arithmetic_expr(
            num_terms=random.choice([2, 3]), var_pool=var_pool
        )
        state = arith_state1 | arith_state2
        rel_op: str = random.choice(rel_ops)
        expr = f"({arith_expr1} {self._operator_map[rel_op]} {arith_expr2})"
        # best-effort result for internal use (unused by callers)
        expr2 = expr.replace(self._operator_map["DIV_OP"], "//")
        try:
            result: bool = bool(eval(expr2, {}, state))
        except Exception:
            result = False
        return expr, result, state, {}

    def _generate_boolean_expr(
        self, num_terms: int = 2, ensure_true: bool = False, var_pool: list[str] = None
    ):
        log_ops: list[str] = ["AND_OP", "OR_OP"]
        terms: list[str] = []
        state: dict = {}
        expr_result: bool = None

        for i in range(num_terms):
            term_str, term_res, term_state, _ = self._make_bool_term(var_pool=var_pool)
            state = state | term_state
            # optional NOT
            if random.random() < 0.15:
                term_str = f"{self._operator_map['NOT_OP']}({term_str})"
                term_res = not term_res
            if i == 0:
                terms.append(term_str)
                expr_result = term_res
            else:
                log_op = random.choice(log_ops)
                terms.append(self._operator_map[log_op])
                terms.append(term_str)
                if log_op == "AND_OP":
                    expr_result = bool(expr_result and term_res)
                else:
                    expr_result = bool(expr_result or term_res)
        expr = " ".join(terms)
        return [expr, expr_result, state, {}]

    def _visit_aexp(self, num_terms: int = 3, num_var_terms: int = 2) -> str:
        expr, _state, _counts = self._generate_arithmetic_expr(
            num_terms=num_terms,
            num_var_terms=num_var_terms,
            var_pool=getattr(self, "var_list", None),
        )
        return expr

    def _visit_bexp(self, num_terms: int = 2) -> str:
        expr, _result, _state, _counts = self._generate_boolean_expr(
            num_terms=num_terms, var_pool=getattr(self, "var_list", None)
        )
        return expr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMP random program generator")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible program generation",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    fuzzer: ImpFuzzerGenerator = ImpFuzzerGenerator()
    print(fuzzer._visit_program())
