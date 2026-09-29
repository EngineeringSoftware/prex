import time
import string
import random
from llm_interpreter.macros import Macros
from antlr4 import InputStream, CommonTokenStream
from llm_interpreter.utils import read_from_txt_file
from llm_interpreter.language.language import Language
from llm_interpreter.language.visitors import IMPMutationVisitor, IMPFuzzerVisitor
from llm_interpreter.language.language import RuleMeta
from llm_interpreter.antlr4_parsers import IMPBASELexer, IMPBASEParser
from llm_interpreter.language.metrics import (
    IMPHalsteadVisitor,
    IMPCyclomaticVisitor,
    IMPDepDegreeVisitor,
)


class IMP(Language):
    _DEFAULT_SYNTAX_MAP: dict = {
        "IF": "if",
        "ELSE": "else",
        "WHILE": "while",
        "HALT": "halt",
        "CONTINUE": "continue",
        "BREAK": "break",
        "ASSIGN_OP": "=",
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
        "LOOP": "LOOP",
        "LE": "LE",
        "ERROR": "ERROR",
    }

    def __init__(self, name, semantics_type: Language.SEMANTICS_TYPE, **kwargs):
        replaced_map: dict = self.replace_tokens(IMP._DEFAULT_SYNTAX_MAP, **kwargs)
        replaced_map["name"] = name.upper()
        replaced_map["semantic_type"] = semantics_type.value
        self._replaced_map = replaced_map
        super().__init__(
            name.upper(),
            semantics_type,
            replaced_map,
            self._build_syntax(semantics_type).format(**replaced_map),
            self._build_semantics(semantics_type).format(**replaced_map),
        )

    # fed

    def get_ebnf_syntax(self) -> str:
        replaced_map = self._replaced_map.copy()
        replaced_map["name"] = "MyLang"
        return read_from_txt_file(f"{Macros.main_dir}/language/imp/syntax.txt").format(
            **replaced_map
        )

    def get_antlr4_syntax(self) -> str:
        replaced_map = self._replaced_map.copy()
        replaced_map["name"] = "MyLang"
        return read_from_txt_file(
            f"{Macros.main_dir}/language/imp/antlr4-new.txt"
        ).format(**replaced_map)

    def _build_syntax(self, semantics_type: Language.SEMANTICS_TYPE) -> str:
        if semantics_type == Language.SEMANTICS_TYPE.K:
            return read_from_txt_file(f"{Macros.main_dir}/language/imp/k_syntax.txt")
        else:
            return read_from_txt_file(f"{Macros.main_dir}/language/imp/syntax.txt")
        fi

    # fed

    def _build_antlr4_syntax(self) -> str:
        return read_from_txt_file(f"{Macros.main_dir}/language/imp/antlr4.txt")

    # fed

    def _build_sos_antlr4_visitor(self) -> str:
        return read_from_txt_file(f"{Macros.main_dir}/language/imp/sos_visitor.txt")

    # fed

    def _build_bos_antlr4_visitor(self) -> str:
        raise NotImplementedError("Semantics not defined")

    # fed

    def _build_dos_antlr4_visitor(self) -> str:
        raise NotImplementedError("Semantics not defined")

    # fed

    def _build_stmt_cleaner_visitor(self) -> str:
        return read_from_txt_file(f"{Macros.main_dir}/language/imp/cleaner_visitor.txt")

    # fed

    def _build_k_semantics(self) -> str:
        return read_from_txt_file(f"{Macros.main_dir}/language/imp/k.txt")

    # fed

    def _build_sos_semantics(self) -> str:
        return read_from_txt_file(f"{Macros.main_dir}/language/imp/sos.txt")

    # fed

    def _build_bos_semantics(self) -> str:
        raise NotImplementedError("Semantics not defined")

    # fed

    def _build_dos_semantics(self) -> str:
        raise NotImplementedError("Semantics not defined")

    # fed

    def translate_unmutated_program(self, program: str) -> str:
        input_stream = InputStream(program)
        lexer = IMPBASELexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = IMPBASEParser(token_stream)
        tree = parser.program()
        mutation_visitor = IMPMutationVisitor(self._operator_map)
        return mutation_visitor.visit(tree)

    # fed

    def translate_unmutated_program_to_semantically_invalid(
        self, program: str
    ) -> (str, str):
        input_stream = InputStream(program)
        lexer = IMPBASELexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = IMPBASEParser(token_stream)
        tree = parser.program()
        fuzzer_visitor = IMPFuzzerVisitor()
        return (fuzzer_visitor.visit(tree), fuzzer_visitor.get_selected_fuzzing_type())

    # fed

    def generate_arithmetic_expr(
        self, num_terms: int = 8, num_var_terms: int = 4, zero_init: bool = True
    ):
        random.seed(time.time())
        var_set: set = set()
        while len(var_set) < num_var_terms:
            var_set.add((random.choice(string.ascii_letters)).lower())
        # rof
        var_set = list(var_set)
        state: dict = {
            var: 1 if zero_init else random.randint(-50, 50) for var in var_set
        }
        ops: list[str] = ["PLUS_OP", "MINUS_OP", "MUL_OP", "DIV_OP", "MOD_OP"]
        op_count: dict = {op: 0 for op in ops}
        op_count["Rule 1"] = 0
        op_count["Rule 2"] = 0
        op_count["Rule 48"] = 0
        expr: list[str] = []
        var_terms_left: int = 4
        terms_left: int = num_terms
        while terms_left > 0:
            term: str = None
            # if terms left in the expr is less than or equal to
            # num of var terms to be used then all remaining terms will be var terms
            # else, pick var and literal terms randomly (50/50).
            if terms_left <= var_terms_left:
                term = random.choice(var_set)
                op_count["Rule 2"] = op_count["Rule 2"] + 1
                var_terms_left -= 1
            else:
                if random.randint(0, 10) % 2 == 0:
                    term = random.choice(var_set)
                    op_count["Rule 2"] = op_count["Rule 2"] + 1
                    var_terms_left -= 1
                else:
                    term = str(random.randint(1, 10))
                    op_count["Rule 1"] = op_count["Rule 1"] + 1
                # fi
            # fi
            # Choose unary negation with 20% probability
            if random.randint(0, 10) % 4 == 0:
                term = f"(- {term})"
                op_count["Rule 48"] = op_count["Rule 48"] + 1
            # fi
            if terms_left != num_terms:
                term = term + ")"
            else:
                term = ("(" * (num_terms - 1)) + term
            # fi
            expr.append(term)
            if terms_left == 1:
                break
            else:
                chosen_op: str = random.choice(ops)
                op_count[chosen_op] = op_count[chosen_op] + 1
                expr.append(self._operator_map[chosen_op])
            # fi
            terms_left -= 1
        # elihw
        op_count["Rule 5"] = op_count.pop("PLUS_OP")
        op_count["Rule 6"] = op_count.pop("MINUS_OP")
        op_count["Rule 7"] = op_count.pop("MUL_OP")
        op_count["Rule 8"] = op_count.pop("DIV_OP")
        op_count["Rule 9"] = op_count.pop("MOD_OP")
        return (" ".join(expr), state, op_count)

    # fed

    def _make_bool_term(self):
        rel_ops: list[str] = ["LT_OP", "LTEQ_OP", "GT_OP", "GTEQ_OP", "EQ_OP", "NEQ_OP"]
        rule_map: dict = {
            "LT_OP_True": "10",
            "LT_OP_False": "11",
            "LTEQ_OP_True": "12",
            "LTEQ_OP_False": "13",
            "GT_OP_True": "14",
            "GT_OP_False": "15",
            "GTEQ_OP_True": "16",
            "GTEQ_OP_False": "17",
            "EQ_OP_True": "18",
            "EQ_OP_False": "19",
            "NEQ_OP_True": "20",
            "NEQ_OP_False": "21",
        }
        [arith_expr1, arith_state1, arith_rule_count1] = self.generate_arithmetic_expr()
        [arith_expr2, arith_state2, arith_rule_count2] = self.generate_arithmetic_expr()
        state = arith_state1 | arith_state2
        rule_count = arith_rule_count1
        for key, value in arith_rule_count2.items():
            rule_count[key] += value
        # rof
        rel_op: str = random.choice(rel_ops)
        expr = f"({arith_expr1} {self._operator_map[rel_op]} {arith_expr2})"
        expr2 = expr.replace(self._operator_map["DIV_OP"], "//")
        result: bool = eval(expr2, {}, state)
        map_key = f"{rel_op}_{str(result)}"
        rule_count[f"Rule {rule_map[map_key]}"] = 1
        return expr, result, state, rule_count

    # fed

    def generate_boolean_expr(self, num_terms: int = 8, enure_true: bool = False):
        num_terms_left: int = num_terms
        log_ops: list[str] = ["AND_OP", "OR_OP"]
        rule_map: dict = {
            "AND_OP_True": "22",
            "AND_OP_False": "23",
            "OR_OP_True": "24",
            "OR_OP_False": "25",
            "NOT_OP_False": "28",
            "NOT_OP_True": "29",
        }
        expr: list[str] = []
        expr_result: bool = None
        state: dict = {}
        rule_count: dict = {}

        while num_terms_left > 0:
            [bool_expr, result, expr_state, expr_rule_count] = self._make_bool_term()
            term: str = bool_expr
            state = state | expr_state

            if random.randint(0, 10) % 4 == 0:
                term = f"{self._operator_map['NOT_OP']}({term})"
                result = not result
                map_key = f"NOT_OP_{str(result)}"
                expr_rule_count[f"Rule {rule_map[map_key]}"] = 1
            # fi
            if num_terms_left == num_terms:
                term = ("(" * (num_terms - 1)) + term
                expr_result = result
                rule_count = expr_rule_count
            else:
                log_op_symbol: str = expr[-1]
                log_op: str = None
                for key, value in self._operator_map.items():
                    if value == log_op_symbol:
                        log_op = key
                        break
                    # fi
                # rof
                match log_op:
                    case "AND_OP":
                        expr_result = expr_result and result
                        map_key = f"AND_OP_{str(expr_result)}"
                        expr_rule_count[f"Rule {rule_map[map_key]}"] = 1
                    case "OR_OP":
                        expr_result = expr_result or result
                        map_key = f"OR_OP_{str(expr_result)}"
                        expr_rule_count[f"Rule {rule_map[map_key]}"] = 1
                # hctam
                for key, value in expr_rule_count.items():
                    if key in rule_count:
                        rule_count[key] += value
                    else:
                        rule_count[key] = value
                    # fi
                # rof
                term = term + ")"
            # fi
            expr.append(term)
            if num_terms_left == 1:
                break
            else:
                chosen_log_op: str = random.choice(log_ops)
                expr.append(self._operator_map[chosen_log_op])
            # fi
            num_terms_left -= 1
        # elihw
        expr = " ".join(expr)
        return [expr, expr_result, state, rule_count]

    # fed

    def get_halstead_metrics(
        self, program: str, print_operators: bool = False, print_operands: bool = False
    ) -> dict:
        return IMPHalsteadVisitor().get_halstead_metrics(
            program, print_operators, print_operands
        )

    # fed

    def get_extended_cyclomatic_metrics(self, program: str) -> dict:
        return IMPCyclomaticVisitor().get_extended_cyclomatic_metrics(program)

    # fed

    def get_depdegree_metrics(self, program: str) -> dict:
        return IMPDepDegreeVisitor().get_depdegree_metrics(program)

    # fed

    def get_semantics_glossary(self, semantics_type: Language.SEMANTICS_TYPE) -> dict:
        if semantics_type == Language.SEMANTICS_TYPE.K:
            return self.SEMANTICS_GLOSSARY_K.format(**self._replaced_map)
        else:
            return self.SEMANTICS_GLOSSARY.format(**self._replaced_map)
        # fi
        
    # fed

    @classmethod
    def get_semantics_rule_groups(cls, semantics_type) -> dict:
        match semantics_type:
            case Language.SEMANTICS_TYPE.K:
                return {
                    "Arithmetic": [[3, 11]],
                    "Assignment": [[21]],
                    "Relational": [[12, 17]],
                    "Logical": [[18,20]],
                    "Declaration": [[36]],
                    "Loop": [[24,25]],
                    "Halt": [[26]],
                    "Id": [[1,2]],
                    "Conditional": [[22,23]],
                    "Break & Continue": [[27, 35]],
                }
            case Language.SEMANTICS_TYPE.SOS:
                return {
                    "Arithmetic": [[7, 27]],
                    "Assignment": [[4, 6]],
                    "Relational": [[28, 51]],
                    "Logical": [[52, 62]],
                    "Declaration": [[3]],
                    "Loop": [[67, 70], [77]],
                    "Halt": [[78]],
                    "Id": [[1, 2]],
                    "Conditional": [[64, 66]],
                    "Break & Continue": [[71, 76]],
                }
            case _:
                raise NotImplementedError("Semantics not implemented for IMP!")
        #hctam
    
    #fed




    RULE_DESCRIPTIONS_NOTATION_COMPREHENSION = {

    1: "If a variable is associated with a value in the current program state, then evaluating it yields that value.",

    2: "If a variable has no associated value in the current program state, then attempting to evaluate it results in an error and the program execution stops immediately.",

    3: "A declaration of the form `int x` introduces `x` into the program state and assigns it the value `0` before execution continues.",

    4: "If the expression on the right-hand side of `{ASSIGN_OP}` can still be evaluated further, then evaluation proceeds by first evaluating that expression.",

    5: "If a variable on the left-hand side of `{ASSIGN_OP}` already exists in the program state and the right-hand side of `{ASSIGN_OP}` has been fully evaluated to a value, then that variable is updated with that value in the program state.",

    6: "If a variable on the left-hand side of `{ASSIGN_OP}` does not exist in the program state and the right-hand side of `{ASSIGN_OP}` has been fully evaluated to a value, then an error occurs and the program execution stops immediately.",

    7: "If the left operand of a binary expression using `{PLUS_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    8: "If the left operand of a binary expression using `{PLUS_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    9: "When both operands of a `{PLUS_OP}` expression are values, the result is obtained by adding those two values.",

    10: "If the left operand of a binary expression using `{MINUS_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    11: "If the left operand of a binary expression using `{MINUS_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    12: "When both operands of a `{MINUS_OP}` expression are values, the result is obtained by subtracting the right operand value from the left operand value.",

    13: "If the left operand of a binary expression using `{MUL_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    14: "If the left operand of a binary expression using `{MUL_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    15: "When both operands of a `{MUL_OP}` expression are values, the result is obtained by multiplying those two values.",

    16: "If the left operand of a binary expression using `{DIV_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    17: "If the left operand of a binary expression using `{DIV_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    18: "When both operands of a `{DIV_OP}` expression are values and the right operand value is not zero, the result is obtained by integer division of the left operand value by the right operand value.",

    19: "When both operands of a `{DIV_OP}` expression are values and the right operand value is zero, then an error occurs and the program execution stops immediately.",

    20: "If the left operand of a binary expression using `{MOD_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    21: "If the left operand of a binary expression using `{MOD_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    22: "When both operands of a `{MOD_OP}` expression are values and the right operand value is not zero, then the remainder of the integer division of the left operand value by the right operand value is the result.",

    23: "When both operands of a `{MOD_OP}` expression are values and the right operand value is zero, then an error occurs and the program execution stops immediately.",

    24: "If the operand of a unary expression using `{MINUS_OP}` can still be evaluated, then evaluation proceeds within that operand.",

    25: "When the operand of a unary `{MINUS_OP}` expression is a value, the result is obtained by negating the value.",

    26: "If the operand of a unary expression using `{PLUS_OP}` can still be evaluated, then evaluation proceeds within that operand.",

    27: "When the operand of a unary `{PLUS_OP}` expression is a value, the result is that same value.",

    28: "If the left operand of a binary expression using `{LT_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    29: "If the left operand of a binary expression using `{LT_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    30: "When both operands of a `{LT_OP}` expression are values and the left operand value is less than the right operand value, the result is `true`.",

    31: "When both operands of a `{LT_OP}` expression are values and the left operand value is not less than the right operand value, the result is `false`.",

    32: "If the left operand of a binary expression using `{LTEQ_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    33: "If the left operand of a binary expression using `{LTEQ_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    34: "When both operands of a `{LTEQ_OP}` expression are values and the left operand value is less than or equal to the right operand value, the result is `true`.",

    35: "When both operands of a `{LTEQ_OP}` expression are values and the left operand value is greater than the right operand value, the result is `false`.",

    36: "If the left operand of a binary expression using `{GT_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    37: "If the left operand of a binary expression using `{GT_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    38: "When both operands of a `{GT_OP}` expression are values and the left operand value is greater than the right operand value, the result is `true`.",

    39: "When both operands of a `{GT_OP}` expression are values and the left operand value is not greater than the right operand value, the result is `false`.",

    40: "If the left operand of a binary expression using `{GTEQ_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    41: "If the left operand of a binary expression using `{GTEQ_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    42: "When both operands of a `{GTEQ_OP}` expression are values and the left operand value is greater than or equal to the right operand value, the result is `true`.",

    43: "When both operands of a `{GTEQ_OP}` expression are values and the left operand value is less than the right operand value, the result is `false`.",

    44: "If the left operand of a binary expression using `{EQ_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    45: "If the left operand of a binary expression using `{EQ_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    46: "When both operands of a `{EQ_OP}` expression are values and the left operand value is equal to the right operand value, the result is `true`.",

    47: "When both operands of a `{EQ_OP}` expression are values and the left operand value is not equal to the right operand value, the result is `false`.",

    48: "If the left operand of a binary expression using `{NEQ_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    49: "If the left operand of a binary expression using `{NEQ_OP}` is already a value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    50: "When both operands of a `{NEQ_OP}` expression are values and the left operand value is not equal to the right operand value, the result is `true`.",

    51: "When both operands of a `{NEQ_OP}` expression are values and the left operand value is equal to the right operand value, the result is `false`.",

    52: "If the left operand of a binary expression using `{AND_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    53: "If the left operand of a binary expression using `{AND_OP}` is already a boolean value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    54: "When both operands of a `{AND_OP}` expression are boolean values and both the left and right operand values are `true`, the result is `true`.",

    55: "When both operands of a `{AND_OP}` expression are boolean values and either the left operand value or the right operand value is `false`, the result is `false`.",

    56: "If the left operand of a binary expression using `{OR_OP}` can be evaluated further, then evaluation proceeds within that left operand.",

    57: "If the left operand of a binary expression using `{OR_OP}` is already a boolean value and the right operand can still be evaluated, then evaluation proceeds within the right operand.",

    58: "When both operands of a `{OR_OP}` expression are boolean values and either the left operand value or the right operand value is `true`, the result is `true`.",

    59: "When both operands of a `{OR_OP}` expression are boolean values and both the left and right operand values are `false`, the result is `false`.",

    60: "If the operand of a unary expression using `{NOT_OP}` can still be evaluated, then evaluation proceeds within that operand.",

    61: "When the operand of a unary `{NOT_OP}` expression is a boolean value and the operand value is `false`, the result is `true`.",

    62: "When the operand of a unary `{NOT_OP}` expression is a boolean value and the operand value is `true`, the result is `false`.",

    63: "If the head/first statement can make a step, then the whole statement list steps by updating that head statement (and program state may change).",

    64: "If the predicate `b` in the `{IF}(b) {{SL1}} {ELSE} {{SL2}}` statement can still be evaluated, then evaluation proceeds within that predicate.",

    65: "If the predicate `q` in the `{IF}(q) {{SL1}} {ELSE} {{SL2}}` statement is a boolean value and `q = true`, then the execution proceeds with the statements in the statement list `SL1` ({IF}-block).",

    66: "If the predicate `q` in the `{IF}(q) {{SL1}} {ELSE} {{SL2}}` statement is a boolean value and `q = false`, then the execution proceeds with the statements in the statement list `SL2` ({ELSE}-block).",

    67: "A `{WHILE}(b) {{SL1}}` statement is translated to a `{LOOP}(b) {{SL1}}` statement and also pushed onto the control stack.",

    68: "If the predicate `b` in the `{LOOP}(b) {{SL1}}` statement can still be evaluated, then evaluation proceeds within that predicate.",

    69: "If the predicate `q` in the `{LOOP}(q) {{SL1}}` statement is a boolean value and `q = false` and if `SL` is the list of statements following `{LOOP}(q) {{SL1}}`, then the execution proceeds with the statements in `SL`.",

    70: "If the predicate `q` in the `{LOOP}(q) {{SL1}}` statement is a boolean value and `q = true`, then the statement `{LE}` is inserted after the statements in statement list `SL1` ({LOOP}-block) and the execution proceeds with the statements in the statement list `SL1`.",

    71: "When a `{BREAK}` statement is encountered and the statement following it is not `{LE}` and the control stack is not empty, then the statement following it is dropped and the `{BREAK}` remains the head of the statement list.",

    72: "When a `{BREAK}` statement is encountered and the statement following it is `{LE}` and the control stack is not empty, then the control stack is popped and the execution proceeds with the statements following the statements `{BREAK}` and `{LE}`.",

    73: "If a `{BREAK}` statement is encountered and the control stack is empty, then an error occurs and the program execution stops immediately.",

    74: "When a `{CONTINUE}` statement is encountered and the statement following it is not `{LE}` and the control stack is not empty, then the statement following it is dropped and the `{CONTINUE}` remains the head of the statement list.",

    75: "When a `{CONTINUE}` statement is encountered and the statement following it is `{LE}` and the control stack is not empty, then the `{CONTINUE}` and `{LE}` statements are replaced with the statement at the top of the control stack and the control stack is popped.",

    76: "If a `{CONTINUE}` statement is encountered and the control stack is empty, then an error occurs and the program execution stops immediately.",

    77: "When a `{LE}` statement is encountered and the control stack is not empty, then the `{LE}` statement is replaced with the statement at the top of the control stack and the control stack is popped.",

    78: "When a `{HALT}` statement is encountered, the program execution stops immediately."
    }

    RULE_SEMANTIC_DEFINITIONS = {

    1: """

        𝜎(x) = v
        -------------
    〈x,𝜎,χ〉 →  v

    """,

    2: """

                𝜎(x) = ⟂
        --------------------------
    〈x,𝜎,χ〉 →  〈{ERROR},𝜎,χ〉

    """,

    3: """

        ------------------------------------
    〈int x :: SL,𝜎,χ〉 ⇉ 〈SL,𝜎[x ↦ 0],χ〉

    """,

    4: """

                    〈a,𝜎,χ〉 → 〈a',𝜎,χ〉
        -----------------------------------------------------------
    〈x {ASSIGN_OP} a :: SL,𝜎,χ〉 ⇉ 〈x {ASSIGN_OP} a' :: SL,𝜎,χ〉

    """,

    5: """

                        𝜎(x) ≠ ⟂
        ----------------------------------------------
    〈x {ASSIGN_OP} v :: SL,𝜎,χ〉 ⇉ 〈SL,𝜎[x ↦ v],χ〉

    """,

    6: """

                        𝜎(x) = ⟂
        ---------------------------------------------
    〈x {ASSIGN_OP} v :: SL,𝜎,χ〉 ⇉  〈{ERROR},𝜎,χ〉

    """,

    7: """

                〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        -----------------------------------------------
    〈a1 {PLUS_OP} a2,𝜎,χ〉 → 〈a1' {PLUS_OP} a2,𝜎,χ〉

    """,

    8: """

                〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        -----------------------------------------------
    〈v1 {PLUS_OP} a2,𝜎,χ〉 → 〈v1 {PLUS_OP} a2',𝜎,χ〉

    """,

    9: """

                v3 = v1 + v2
        ---------------------------
    〈v1 {PLUS_OP} v2,𝜎,χ〉 → v3

    """,

    10: """

                〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        -------------------------------------------------
    〈a1 {MINUS_OP} a2,𝜎,χ〉 → 〈a1' {MINUS_OP} a2,𝜎,χ〉

    """,

    11: """

                〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        -------------------------------------------------
    〈v1 {MINUS_OP} a2,𝜎,χ〉 → 〈v1 {MINUS_OP} a2',𝜎,χ〉

    """,

    12: """

                v3 = v1 - v2
        ----------------------------
    〈v1 {MINUS_OP} v2,𝜎,χ〉 → v3

    """,

    13: """

                    〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        ---------------------------------------------
    〈a1 {MUL_OP} a2,𝜎,χ〉 → 〈a1' {MUL_OP} a2,𝜎,χ〉

    """,

    14: """

                〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        ---------------------------------------------
    〈v1 {MUL_OP} a2,𝜎,χ〉 → 〈v1 {MUL_OP} a2',𝜎,χ〉

    """,

    15: """

                v3 = v1 * v2
        --------------------------
    〈v1 {MUL_OP} v2,𝜎,χ〉 → v3

    """,

    16: """

                〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        ---------------------------------------------
    〈a1 {DIV_OP} a2,𝜎,χ〉 → 〈a1' {DIV_OP} a2,𝜎,χ〉

    """,

    17: """

                〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        ---------------------------------------------
    〈v1 {DIV_OP} a2,𝜎,χ〉 → 〈v1 {DIV_OP} a2',𝜎,χ〉

    """,

    18: """

        v2 ≠ 0       v3 = v1 / v2
        ----------------------------
    〈v1 {DIV_OP} v2,𝜎,χ〉 → v3

    """,

    19: """

                        v2 = 0
        --------------------------------------
    〈v1 {DIV_OP} v2,𝜎,χ〉 →  〈{ERROR},𝜎,χ〉

    """,

    20: """

                    〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        ---------------------------------------------
    〈a1 {MOD_OP} a2,𝜎,χ〉 → 〈a1' {MOD_OP} a2,𝜎,χ〉

    """,

    21: """

                〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        ---------------------------------------------
    〈v1 {MOD_OP} a2,𝜎,χ〉 → 〈v1 {MOD_OP} a2',𝜎,χ〉

    """,

    22: """

        v2 ≠ 0           v3 = v1 % v2
        --------------------------------
    〈v1 {MOD_OP} v2,𝜎,χ〉 → v3

    """,

    23: """

                        v2 = 0
        --------------------------------------
    〈v1 {MOD_OP} v2,𝜎,χ〉 →  〈{ERROR},𝜎,χ〉

    """,

    24: """

                〈a,𝜎,χ〉 → 〈a',𝜎,χ〉
        -----------------------------------------
    〈{MINUS_OP} a,𝜎,χ〉 → 〈{MINUS_OP} a',𝜎,χ〉

    """,

    25: """

                v2 = -v1
        --------------------------
    〈{MINUS_OP} v1,𝜎,χ〉 →  v2

    """,

    26: """

                〈a,𝜎,χ〉 → 〈a',𝜎,χ〉
        ---------------------------------------
    〈{PLUS_OP} a,𝜎,χ〉 → 〈{PLUS_OP} a',𝜎,χ〉

    """,

    27: """

        -----------------------
    〈{PLUS_OP} v,𝜎,χ〉 →  v

    """,

    28: """

                〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        -------------------------------------------
    〈a1 {LT_OP} a2,𝜎,χ〉 → 〈a1' {LT_OP} a2,𝜎,χ〉

    """,

    29: """

                〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        -------------------------------------------
    〈v1 {LT_OP} a2,𝜎,χ〉 → 〈v1 {LT_OP} a2',𝜎,χ〉

    """,

    30: """

                    v1 < v2
        ---------------------------
    〈v1 {LT_OP} v2,𝜎,χ〉 → true

    """,

    31: """

                    v1 ≥ v2
        ----------------------------
    〈v1 {LT_OP} v2,𝜎,χ〉 → false

    """,

    32: """

                    〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        -----------------------------------------------
    〈a1 {LTEQ_OP} a2,𝜎,χ〉 → 〈a1' {LTEQ_OP} a2,𝜎,χ〉

    """,

    33: """

                    〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        -----------------------------------------------
    〈v1 {LTEQ_OP} a2,𝜎,χ〉 → 〈v1 {LTEQ_OP} a2',𝜎,χ〉

    """,

    34: """

                    v1 ≤ v2
        -----------------------------
    〈v1 {LTEQ_OP} v2,𝜎,χ〉 → true

    """,

    35: """

                    v1 > v2
        ------------------------------
    〈v1 {LTEQ_OP} v2,𝜎,χ〉 → false

    """,

    36: """

                〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        -------------------------------------------
    〈a1 {GT_OP} a2,𝜎,χ〉 → 〈a1' {GT_OP} a2,𝜎,χ〉

    """,

    37: """

                〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        -------------------------------------------
    〈v1 {GT_OP} a2,𝜎,χ〉 → 〈v1 {GT_OP} a2',𝜎,χ〉

    """,

    38: """

                v1 > v2
        ---------------------------
    〈v1 {GT_OP} v2,𝜎,χ〉 → true

    """,

    39: """

                v1 ≤ v2
        ----------------------------
    〈v1 {GT_OP} v2,𝜎,χ〉 → false

    """,

    40: """

                    〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        -----------------------------------------------
    〈a1 {GTEQ_OP} a2,𝜎,χ〉 → 〈a1' {GTEQ_OP} a2,𝜎,χ〉

    """,

    41: """

                    〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        -----------------------------------------------
    〈v1 {GTEQ_OP} a2,𝜎,χ〉 → 〈v1 {GTEQ_OP} a2',𝜎,χ〉

    """,

    42: """

                v1 ≥ v2
        -----------------------------
    〈v1 {GTEQ_OP} v2,𝜎,χ〉 → true

    """,

    43: """

                    v1 < v2
        ------------------------------
    〈v1 {GTEQ_OP} v2,𝜎,χ〉 → false

    """,

    44: """

                〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        -------------------------------------------
    〈a1 {EQ_OP} a2,𝜎,χ〉 → 〈a1' {EQ_OP} a2,𝜎,χ〉

    """,

    45: """

                〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        -------------------------------------------
    〈v1 {EQ_OP} a2,𝜎,χ〉 → 〈v1 {EQ_OP} a2',𝜎,χ〉

    """,

    46: """

                v1 = v2
        ---------------------------
    〈v1 {EQ_OP} v2,𝜎,χ〉 → true

    """,

    47: """

                    v1 ≠ v2
        ----------------------------
    〈v1 {EQ_OP} v2,𝜎,χ〉 → false

    """,

    48: """

                〈a1,𝜎,χ〉 → 〈a1',𝜎,χ〉
        ---------------------------------------------
    〈a1 {NEQ_OP} a2,𝜎,χ〉 → 〈a1' {NEQ_OP} a2,𝜎,χ〉

    """,

    49: """

                〈a2,𝜎,χ〉 → 〈a2',𝜎,χ〉
        ---------------------------------------------
    〈v1 {NEQ_OP} a2,𝜎,χ〉 → 〈v1 {NEQ_OP} a2',𝜎,χ〉

    """,

    50: """

                v1 ≠ v2
        ----------------------------
    〈v1 {NEQ_OP} v2,𝜎,χ〉 → true

    """,

    51: """

                    v1 = v2
        -----------------------------
    〈v1 {NEQ_OP} v2,𝜎,χ〉 → false

    """,

    52: """

                〈b1,𝜎,χ〉 → 〈b1',𝜎,χ〉
        ---------------------------------------------
    〈b1 {AND_OP} b2,𝜎,χ〉 → 〈b1' {AND_OP} b2,𝜎,χ〉

    """,

    53: """

                〈b2,𝜎,χ〉 → 〈b2',𝜎,χ〉
        ---------------------------------------------
    〈q1 {AND_OP} b2,𝜎,χ〉 → 〈q1 {AND_OP} b2',𝜎,χ〉

    """,

    54: """

            q1 = true ⋀ q2 = true
        ----------------------------
    〈q1 {AND_OP} q2,𝜎,χ〉 → true

    """,

    55: """

            q1 = false ⋁ q2 = false
        -----------------------------
    〈q1 {AND_OP} q2,𝜎,χ〉 → false

    """,

    56: """

                〈b1,𝜎,χ〉 → 〈b1',𝜎,χ〉
        -------------------------------------------
    〈b1 {OR_OP} b2,𝜎,χ〉 → 〈b1' {OR_OP} b2,𝜎,χ〉

    """,

    57: """

                〈b2,𝜎,χ〉 → 〈b2',𝜎,χ〉
        -------------------------------------------
    〈q1 {OR_OP} b2,𝜎,χ〉 → 〈q1 {OR_OP} b2',𝜎,χ〉

    """,

    58: """

            q1 = true ⋁ q2 = true
        ---------------------------
    〈q1 {OR_OP} q2,𝜎,χ〉 → true

    """,

    59: """

            q1 = false ⋀ q2 = false
        ----------------------------
    〈q1 {OR_OP} q2,𝜎,χ〉 → false

    """,

    60: """

                〈b,𝜎,χ〉 → 〈b',𝜎,χ〉
        -------------------------------------
    〈{NOT_OP} b,𝜎,χ〉 → 〈{NOT_OP} b',𝜎,χ〉

    """,

    61: """

                q = false
        -------------------------
    〈{NOT_OP} q,𝜎,χ〉 → true

    """,

    62: """

                q = true
        ------------------------
    〈{NOT_OP} q,𝜎,χ〉 → false

    """,

    63: """

        〈s,𝜎,χ〉 ⇉ 〈s',𝜎',χ'〉
        ---------------------------------
    〈s :: SL,𝜎,χ〉 ⇉ 〈s' :: SL,𝜎',χ'〉

    """,

    64: """

                                        〈b,𝜎,χ〉 → 〈b',𝜎,χ〉
        -------------------------------------------------------------------------------------------
    〈{IF}(b) {{SL1}} {ELSE} {{SL2}} :: SL,𝜎,χ〉 ⇉ 〈{IF}(b') {{SL1}} {ELSE} {{SL2}} :: SL,𝜎,χ〉

    """,

    65: """

                                    q = true
        ---------------------------------------------------------------
    〈{IF}(q) {{SL1}} {ELSE} {{SL2}} :: SL,𝜎,χ〉 ⇉ 〈SL1 ++ SL,𝜎,χ〉

    """,

    66: """

                                q = false
        ---------------------------------------------------------------
    〈{IF}(q) {{SL1}} {ELSE} {{SL2}} :: SL,𝜎,χ〉 ⇉ 〈SL2 ++ SL,𝜎,χ〉

    """,

    67: """

        ----------------------------------------------------------------------------------------
    〈{WHILE}(b) {{SL1}} :: SL,𝜎,χ〉 ⇉ 〈{LOOP}(b) {{SL1}} :: SL,𝜎,push({WHILE}(b) {{SL1}}, χ)〉

    """,

    68: """

                            〈b,𝜎,χ〉 → 〈b',𝜎,χ〉
        ---------------------------------------------------------------
    〈{LOOP}(b) {{SL1}} :: SL,𝜎,χ〉 ⇉ 〈{LOOP}(b') {{SL1}} :: SL,𝜎,χ〉

    """,

    69: """

                        q = false
        -----------------------------------------------
    〈{LOOP}(q) {{SL1}} :: SL,𝜎,χ〉 ⇉ 〈SL,𝜎,pop(χ)〉

    """,

    70: """

                            q = true
        ----------------------------------------------------------
    〈{LOOP}(q) {{SL1}} :: SL,𝜎,χ〉 ⇉ 〈SL1 ++ ({LE} :: SL),𝜎,χ〉

    """,

    71: """

                    χ ≠ ε ∧ s ≠ {LE}
        -----------------------------------------------
    〈{BREAK} :: s :: SL,𝜎,χ〉 ⇉ 〈{BREAK} :: SL,𝜎,χ〉

    """,

    72: """

                    χ ≠ ε ∧ s = {LE}
        ------------------------------------------
    〈{BREAK} :: s :: SL,𝜎,χ〉 ⇉ 〈SL,𝜎,pop(χ)〉

    """,

    73: """

                        χ = ε
        -------------------------------------
    〈{BREAK} :: SL,𝜎,χ〉 ⇉  〈{ERROR},𝜎,χ〉

    """,

    74: """

                        χ ≠ ε ∧ s ≠ {LE}
        -----------------------------------------------------
    〈{CONTINUE} :: s :: SL,𝜎,χ〉 ⇉ 〈{CONTINUE} :: SL,𝜎,χ〉

    """,

    75: """

        χ ≠ ε ∧ s = {LE}                s1 = top(χ)
        ----------------------------------------------------
    〈{CONTINUE} :: s :: SL,𝜎,χ〉 ⇉ 〈s1 :: SL,𝜎,pop(χ)〉

    """,

    76: """

                        χ = ε
        ----------------------------------------
    〈{CONTINUE} :: SL,𝜎,χ〉 ⇉  〈{ERROR},𝜎,χ〉

    """,

    77: """

                    s = top(χ)
        --------------------------------------
    〈{LE} :: SL,𝜎,χ〉 ⇉ 〈s :: SL,𝜎,pop(χ)〉

    """,

    78: """

        -----------------------------------
    〈{HALT} :: SL,𝜎,χ〉 ⇉  〈{HALT},𝜎,χ〉

    """
    }

    SEMANTICS_GLOSSARY = """
    Metavariables:
    - x ranges over sort id
    - v ranges over sort literal
    - q ranges over sort bool
    - a ranges over sort aexp
    - b ranges over sort bexp
    - s ranges over sort stmt
    - SL ranges over sort stmt_list and SL ::= ε       // Empty stmt_list
                                            | s :: SL' // head and tail

    Configuration:
    - Store 𝜎 := {{ id ↦ literal }}
    - Control stack χ := ε       // Empty stack
                    | s :: χ'  // s is statement on top of stack
    - configuration := 〈operation,𝜎,χ〉
    - terminal configurations :=〈ε,𝜎,χ〉| 〈{HALT},𝜎,χ〉| 〈{ERROR},𝜎,χ〉

    Metafunctions:
    - push(s, χ) := s :: χ
    - pop(s :: χ) := χ   // χ ≠ ε
    - top(s :: χ) := s
    - top(ε) := ε
    - SL1 ++ SL2 :=
        if SL1 = ε then SL2
        else if SL1 = s :: SL1' then s :: (SL1' ++ SL2)

    Transition/Step:
    - expression step := 〈o,𝜎,χ〉 → 〈o',𝜎,χ〉, expression o steps/reduces to expression o' in exactly one small-step. Store and control stack are not mutated by expression steps
    - statement step  := 〈o,𝜎,χ〉 ⇉ 〈o',𝜎,χ〉, stmt o steps/reduces to stmt o' in exactly one small-step while updating store from 𝜎 to 𝜎' and control stack from χ to χ'

    Every rule is in the form of:
             premises
          --------------
            conclusion
    """

    SEMANTICS_GLOSSARY_K = """
    module {name}-AUXILIARY
    imports {name}-SYNTAX

    syntax KResult ::= Int | Bool
    syntax KItem ::= "breakMarker"
    syntax KItem ::= "{ERROR}"
    syntax Bool ::= isContmark(Stmt) [function]
    rule isContmark(S:ContMark) => true
    rule isContmark(_) => false [owise]
    syntax KItem ::= push(Int)
    syntax KItem ::= "pop"
    syntax KItem ::= pushStmt(Stmt)
    syntax KItem ::= "popStmt"

    configuration <T>
        <k> $PGM:Pgm </k>
        <state> $STATE:Map </state>
        <whileStack> .List </whileStack>
        <whileStmtStack> .List </whileStmtStack>
    </T>

    rule <k> push(X:Int) => . ... </k>
        <whileStack> S:List => ListItem(X) S </whileStack>
    rule <k> pop => . ... </k>
        <whileStack> ListItem(X:Int) S:List => S </whileStack>
    rule <k> pushStmt(S:Stmt) => . ... </k>
        <whileStmtStack> L:List => ListItem(S) L </whileStmtStack>
    rule <k> popStmt => . ... </k>
        <whileStmtStack> ListItem(S:Stmt) L:List => L </whileStmtStack>

    rule {{}} => .
    rule {{S}} => S
    rule S1:Stmt S2:Stmt => S1 ~> S2
    rule <k> parenthesizedA(P:AExp) => P ... </k>
    rule <k> parenthesizedB(P:BExp) => P ... </k>
    rule <k> {ERROR} ~> _ => .</k>
    endmodule
    """

    # K-framework rule descriptions
    RULE_DESCRIPTIONS_NOTATION_COMPREHENSION_K = {
        1: "If a variable is associated with a value in the current program state, then evaluating it yields that value.",
        2: "If a variable has no associated value in the current program state, then attempting to evaluate it results in an error and the program execution stops immediately.",
        3: "When both operands of a `{PLUS_OP}` expression are values, the result is obtained by adding those two values.",
        4: "When both operands of a `{MINUS_OP}` expression are values, the result is obtained by subtracting the right operand value from the left operand value.",
        5: "When both operands of a `{MUL_OP}` expression are values, the result is obtained by multiplying those two values.",
        6: "When both operands of a `{DIV_OP}` expression are values and the right operand value is not zero, the result is obtained by integer division of the left operand value by the right operand value.",
        7: "When both operands of a `{DIV_OP}` expression are values and the right operand value is zero, then an error occurs and the program execution stops immediately.",
        8: "When both operands of a `{MOD_OP}` expression are values and the right operand value is not zero, then the remainder of the integer division of the left operand value by the right operand value is the result.",
        9: "When both operands of a `{MOD_OP}` expression are values and the right operand value is zero, then an error occurs and the program execution stops immediately.",
        10: "When the operand of a unary `{PLUS_OP}` expression is a value, the result is obtained by adding zero to that value.",
        11: "When the operand of a unary `{MINUS_OP}` expression is a value, the result is obtained by negating the value.",
        12: "When both operands of a `{LT_OP}` expression are values, the result is obtained by comparing whether the left operand value is less than the right operand value.",
        13: "When both operands of a `{LTEQ_OP}` expression are values, the result is obtained by comparing whether the left operand value is less than or equal to the right operand value.",
        14: "When both operands of a `{GT_OP}` expression are values, the result is obtained by comparing whether the left operand value is greater than the right operand value.",
        15: "When both operands of a `{GTEQ_OP}` expression are values, the result is obtained by comparing whether the left operand value is greater than or equal to the right operand value.",
        16: "When both operands of an `{EQ_OP}` expression are values, the result is obtained by comparing whether the left operand value is equal to the right operand value.",
        17: "When both operands of a `{NEQ_OP}` expression are values, the result is obtained by comparing whether the left operand value is not equal to the right operand value.",
        18: "When the operand of a unary `{NOT_OP}` expression is a boolean value, the result is obtained by negating that boolean value.",
        19: "When both operands of an `{AND_OP}` expression are boolean values, the result is obtained by applying the logical AND operation to those two boolean values.",
        20: "When both operands of an `{OR_OP}` expression are boolean values, the result is obtained by applying the logical OR operation to those two boolean values.",
        21: "If a variable on the left-hand side of `{ASSIGN_OP}` already exists in the program state and the right-hand side of `{ASSIGN_OP}` has been fully evaluated to a value, then that variable is updated with that value in the program state.",
        22: "If the predicate `q` in the `{IF}(q) SL1 {ELSE} SL2` statement is a boolean value and `q = true`, then the execution proceeds with the statements in the statement list `SL1` ({IF}-block).",
        23: "If the predicate `q` in the `{IF}(q) SL1 {ELSE} SL2` statement is a boolean value and `q = false`, then the execution proceeds with the statements in the statement list `SL2` ({ELSE}-block).",
        24: "When the statement `{WHILE}(B) SL` is executed, the semantics first pushes the entire statement `{WHILE}(B) SL` onto the `<whileStmtStack>`. It then replaces the statement with the internal form `{WHILE}1(B) SL`, inserts a `breakMarker`, and schedules `pop` and `popStmt` to run afterward to restore the `<whileStack>` and `<whileStmtStack>`.",
        25: "If the predicate `q` in the `{WHILE}1(q) S` statement is a boolean value, then the statement is transformed into an `{IF}-{ELSE}` statement that checks the predicate and either executes the body of `{WHILE}1(q) S` statement followed by a continue marker and re-evaluates the `{WHILE}1(q) S` statement, or exits through the `{ELSE}` block.",
        26: "When a `{HALT}` statement is encountered, the program execution stops immediately and all subsequent statements are removed.",
        27: "When a `breakMarker` is encountered, it is removed from the computation and execution continues.",
        28: "When a `continueMarker` is encountered, it is removed from the computation and execution continues.",
        29: "When a `{CONTINUE}` statement is encountered and the statement following it is a `continueMarker`, then both the `{CONTINUE}` and `continueMarker` are removed and execution continues.",
        30: "When a `{CONTINUE}` statement is encountered and the statement following it is not a `continueMarker` and the `<whileStack>` stack is not empty, then the statement following it is dropped and the `{CONTINUE}` remains the head of the computation.",
        31: "If a `{CONTINUE}` statement is encountered and the `<whileStack>` stack is empty, then an error occurs and the program execution stops immediately.",
        32: "When a `{CONTINUE}` statement is encountered and the statement following it is a `breakMarker`, then the statement following it is dropped and the `{CONTINUE}` remains the head of the computation.",
        33: "When a `{BREAK}` statement is encountered and the statement following it is not a `breakMarker` and the `<whileStack>` stack is not empty, then the statement following it is dropped and the `{BREAK}` remains the head of the computation.",
        34: "If a `{BREAK}` statement is encountered and the `<whileStack>` stack is empty, then an error occurs and the program execution stops immediately.",
        35: "When a `{BREAK}` statement is encountered and the statement following it is a `breakMarker`, then both the `{BREAK}` and `breakMarker` are removed and execution continues with the statements following the loop.",
        36: "A declaration of the form `int x` introduces `x` into the program state and assigns it the value `0` before execution continues.",
    }

    # K-framework semantic definitions
    RULE_SEMANTIC_DEFINITIONS_K = {
        1: """
    rule <k> X:Id => I ...</k>
         <state>... X |-> I ...</state>
    """,
        2: """
    rule <k> X:Id => {ERROR} </k>
         <state> Rho:Map </state>
    requires notBool (X in_keys(Rho))
    """,
        3: """
    rule <k> I1 {PLUS_OP} I2 => I1 +Int I2 ... </k>
    """,
        4: """
    rule <k> I1 {MINUS_OP} I2 => I1 -Int I2 ... </k>
    """,
        5: """
    rule <k> I1 {MUL_OP} I2 => I1 *Int I2 ... </k>
    """,
        6: """
    rule <k> I1 {DIV_OP} I2 => I1 /Int I2 ... </k>
    requires I2 =/=Int 0
    """,
        7: """
    rule <k> I1 {DIV_OP} I2 => {ERROR} ... </k>
    requires I2 ==Int 0
    """,
        8: """
    rule <k> I1 {MOD_OP} I2 => I1 %Int I2 ... </k>
    requires I2 =/=Int 0
    """,
        9: """
    rule <k> I1 {MOD_OP} I2 => {ERROR} ... </k>
    requires I2 ==Int 0
    """,
        10: """
    rule <k> {PLUS_OP} I1 => 0 +Int I1 ... </k>
    """,
        11: """
    rule <k> {MINUS_OP} I1 => 0 -Int I1 ... </k>
    """,
        12: """
    rule <k> I1 {LT_OP} I2 => I1 <Int I2 ... </k>
    """,
        13: """
    rule <k> I1 {LTEQ_OP} I2 => I1 <=Int I2 ... </k>
    """,
        14: """
    rule <k> I1 {GT_OP} I2 => I1 >Int I2 ... </k>
    """,
        15: """
    rule <k> I1 {GTEQ_OP} I2 => I1 >=Int I2 ... </k>
    """,
        16: """
    rule <k> I1 {EQ_OP} I2 => I1 ==Int I2 ... </k>
    """,
        17: """
    rule <k> I1 {NEQ_OP} I2 => I1 =/=Int I2 ... </k>
    """,
        18: """
    rule <k> {NOT_OP} B1 => notBool B1 ... </k>
    """,
        19: """
    rule <k> B1 {AND_OP} B2 => B1 andBool B2 ... </k>
    """,
        20: """
    rule <k> B1 {OR_OP} B2 => B1 orBool B2 ... </k>
    """,
        21: """
    rule <k> X {ASSIGN_OP} I:Int; => . ...</k>
         <state>... X |-> (_ => I) ...</state>
    """,
        22: """
    rule <k> {IF} (true)  S {ELSE} _ ; => S ... </k>
    """,
        23: """
    rule <k> {IF} (false)  _ {ELSE} S ; => S ... </k>
    """,
        24: """
    rule <k> ({WHILE} (B) S ;) => pushStmt(({WHILE} (B) S ;)) ~> ({WHILE}1 (B) S) ~> breakMarker ~> pop ~> popStmt ... </k>
    """,
        25: """
    rule <k> {WHILE}1 (B) S => {IF} (B) {{S continueMarker ({WHILE}1 (B) S)}} {ELSE} {{}}; ...</k>
         <whileStack> ListItem(X:Int) S1:List</whileStack>
    """,
        26: """
    rule <k> {HALT}; ~> _ => .</k>
    """,
        27: """
    rule <k> breakMarker => . ... </k>
    """,
        28: """
    rule <k> S:ContMark => . ... </k>
    """,
        29: """
    rule <k> {CONTINUE}; ~> S:ContMark => . ...</k>
    """,
        30: """
    rule <k> {CONTINUE}; ~> S:Stmt => {CONTINUE}; ...</k>
         <whileStack> ListItem(_) REST:List </whileStack>
    requires notBool isContmark(S)
    """,
        31: """
    rule <k> {CONTINUE}; => {ERROR} </k>
         <whileStack> .List </whileStack>
    """,
        32: """
    rule <k> {CONTINUE}; ~> breakMarker => {CONTINUE}; ...</k>
    """,
        33: """
    rule <k> {BREAK}; ~> S:Stmt => {BREAK}; ...</k>
         <whileStack> ListItem(_) REST:List </whileStack>
    """,
        34: """
    rule <k> {BREAK}; => {ERROR} </k>
         <whileStack> .List </whileStack>
    """,
        35: """
    rule <k> {BREAK}; ~> breakMarker => . ... </k>
    """,
        36: """
    rule <k> int X; => . ... </k>
         <state> Rho:Map (.Map => X|->0) </state>
    """,
    }

    # Mapping from rule ID to (family, construct, category) for K-framework semantics
    _K_RULE_META_MAP = {
        1: ("F1", "Variable lookup", "AEXP"),
        2: ("F1", "Variable lookup", "AEXP"),
        3: ("F4", "Addition", "AEXP"),
        4: ("F5", "Subtraction", "AEXP"),
        5: ("F6", "Multiplication", "AEXP"),
        6: ("F7", "Division", "AEXP"),
        7: ("F7", "Division", "AEXP"),
        8: ("F8", "Modulo", "AEXP"),
        9: ("F8", "Modulo", "AEXP"),
        10: ("F10", "Unary plus", "AEXP"),
        11: ("F9", "Unary minus", "AEXP"),
        12: ("F11", "Less-than", "BEXP"),
        13: ("F12", "Less-or-equal", "BEXP"),
        14: ("F13", "Greater-than", "BEXP"),
        15: ("F14", "Greater-or-equal", "BEXP"),
        16: ("F15", "Equality", "BEXP"),
        17: ("F16", "Inequality", "BEXP"),
        18: ("F19", "Boolean NOT", "BEXP"),
        19: ("F17", "Boolean AND", "BEXP"),
        20: ("F18", "Boolean OR", "BEXP"),
        21: ("F3", "Assignment", "SL"),
        22: ("F21", "If-then-else", "CTRL"),
        23: ("F21", "If-then-else", "CTRL"),
        24: ("F22", "While entry", "CTRL"),
        25: ("F23", "Loop execution", "CTRL"),
        26: ("F27", "Halt", "CTRL"),
        27: ("F24", "Break marker", "CTRL"),
        28: ("F24", "Continue marker", "CTRL"),
        29: ("F26", "Continue", "CTRL"),
        30: ("F26", "Continue", "CTRL"),
        31: ("F26", "Continue", "CTRL"),
        32: ("F26", "Continue", "CTRL"),
        33: ("F25", "Break", "CTRL"),
        34: ("F25", "Break", "CTRL"),
        35: ("F25", "Break", "CTRL"),
        36: ("F2", "Declaration", "SL"),
    }

    # Mapping from rule ID to (family, construct, category) for SOS semantics based on Table 13
    _SOS_RULE_META_MAP = {
        1: ("F1", "Variable lookup", "AEXP"),
        2: ("F1", "Variable lookup", "AEXP"),
        3: ("F2", "Declaration", "SL"),
        4: ("F3", "Assignment", "SL"),
        5: ("F3", "Assignment", "SL"),
        6: ("F3", "Assignment", "SL"),
        7: ("F4", "Addition", "AEXP"),
        8: ("F4", "Addition", "AEXP"),
        9: ("F4", "Addition", "AEXP"),
        10: ("F5", "Subtraction", "AEXP"),
        11: ("F5", "Subtraction", "AEXP"),
        12: ("F5", "Subtraction", "AEXP"),
        13: ("F6", "Multiplication", "AEXP"),
        14: ("F6", "Multiplication", "AEXP"),
        15: ("F6", "Multiplication", "AEXP"),
        16: ("F7", "Division", "AEXP"),
        17: ("F7", "Division", "AEXP"),
        18: ("F7", "Division", "AEXP"),
        19: ("F7", "Division", "AEXP"),
        20: ("F8", "Modulo", "AEXP"),
        21: ("F8", "Modulo", "AEXP"),
        22: ("F8", "Modulo", "AEXP"),
        23: ("F8", "Modulo", "AEXP"),
        24: ("F9", "Unary minus", "AEXP"),
        25: ("F9", "Unary minus", "AEXP"),
        26: ("F10", "Unary plus", "AEXP"),
        27: ("F10", "Unary plus", "AEXP"),
        28: ("F11", "Less-than", "BEXP"),
        29: ("F11", "Less-than", "BEXP"),
        30: ("F11", "Less-than", "BEXP"),
        31: ("F11", "Less-than", "BEXP"),
        32: ("F12", "Less-or-equal", "BEXP"),
        33: ("F12", "Less-or-equal", "BEXP"),
        34: ("F12", "Less-or-equal", "BEXP"),
        35: ("F12", "Less-or-equal", "BEXP"),
        36: ("F13", "Greater-than", "BEXP"),
        37: ("F13", "Greater-than", "BEXP"),
        38: ("F13", "Greater-than", "BEXP"),
        39: ("F13", "Greater-than", "BEXP"),
        40: ("F14", "Greater-or-equal", "BEXP"),
        41: ("F14", "Greater-or-equal", "BEXP"),
        42: ("F14", "Greater-or-equal", "BEXP"),
        43: ("F14", "Greater-or-equal", "BEXP"),
        44: ("F15", "Equality", "BEXP"),
        45: ("F15", "Equality", "BEXP"),
        46: ("F15", "Equality", "BEXP"),
        47: ("F15", "Equality", "BEXP"),
        48: ("F16", "Inequality", "BEXP"),
        49: ("F16", "Inequality", "BEXP"),
        50: ("F16", "Inequality", "BEXP"),
        51: ("F16", "Inequality", "BEXP"),
        52: ("F17", "Boolean AND", "BEXP"),
        53: ("F17", "Boolean AND", "BEXP"),
        54: ("F17", "Boolean AND", "BEXP"),
        55: ("F17", "Boolean AND", "BEXP"),
        56: ("F18", "Boolean OR", "BEXP"),
        57: ("F18", "Boolean OR", "BEXP"),
        58: ("F18", "Boolean OR", "BEXP"),
        59: ("F18", "Boolean OR", "BEXP"),
        60: ("F19", "Boolean NOT", "BEXP"),
        61: ("F19", "Boolean NOT", "BEXP"),
        62: ("F19", "Boolean NOT", "BEXP"),
        63: ("F20", "Sequencing", "SL"),
        64: ("F21", "If-then-else", "CTRL"),
        65: ("F21", "If-then-else", "CTRL"),
        66: ("F21", "If-then-else", "CTRL"),
        67: ("F22", "While entry", "CTRL"),
        68: ("F23", "Loop execution", "CTRL"),
        69: ("F23", "Loop execution", "CTRL"),
        70: ("F23", "Loop execution", "CTRL"),
        71: ("F25", "Break", "CTRL"),
        72: ("F25", "Break", "CTRL"),
        73: ("F25", "Break", "CTRL"),
        74: ("F26", "Continue", "CTRL"),
        75: ("F26", "Continue", "CTRL"),
        76: ("F26", "Continue", "CTRL"),
        77: ("F24", "Loop-exit marker", "CTRL"),
        78: ("F27", "Halt", "CTRL"),
    }

    def build_rule_meta_sos(self) -> dict:
        """Build RULE_META_SOS dictionary."""
        result = {}
        
        for rule_id in sorted(set(self.RULE_SEMANTIC_DEFINITIONS.keys()) | set(self.RULE_DESCRIPTIONS_NOTATION_COMPREHENSION.keys())):
            family, construct, category = self._SOS_RULE_META_MAP.get(rule_id, ("F0", "Unknown", "UNKNOWN"))
            
            rule_text = self.RULE_SEMANTIC_DEFINITIONS.get(rule_id, "")        
            rule_text = rule_text.format(**self._replaced_map)
            nl_description = self.RULE_DESCRIPTIONS_NOTATION_COMPREHENSION.get(rule_id, "")
            nl_description = nl_description.format(**self._replaced_map)
            
            result[f"R{rule_id}"] = RuleMeta(
                family=family,
                construct=construct,
                category=category,
                rule_id=rule_id,
                rule_text=rule_text,
                nl_description=nl_description
            )
        
        return result

    def build_rule_meta_k(self) -> dict:
        """Build RULE_META_K dictionary."""
        result = {}
        
        for rule_id in sorted(set(self.RULE_SEMANTIC_DEFINITIONS_K.keys()) | set(self.RULE_DESCRIPTIONS_NOTATION_COMPREHENSION_K.keys())):
            family, construct, category = self._K_RULE_META_MAP.get(rule_id, ("F0", "Unknown", "UNKNOWN"))
            
            rule_text = self.RULE_SEMANTIC_DEFINITIONS_K.get(rule_id, "")        
            rule_text = rule_text.format(**self._replaced_map)
            nl_description = self.RULE_DESCRIPTIONS_NOTATION_COMPREHENSION_K.get(rule_id, "")
            nl_description = nl_description.format(**self._replaced_map)
            
            result[f"R{rule_id}"] = RuleMeta(
                family=family,
                construct=construct,
                category=category,
                rule_id=rule_id,
                rule_text=rule_text,
                nl_description=nl_description
            )
        
        return result

# ssalc
