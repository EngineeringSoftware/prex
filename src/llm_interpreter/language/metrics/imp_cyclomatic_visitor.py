from antlr4 import *
from llm_interpreter.antlr4_parsers import IMPBASELexer, IMPBASEParser, IMPBASEVisitor
from llm_interpreter.language.metrics.base import ExtendedCyclomaticMetric


class IMPCyclomaticVisitor(ExtendedCyclomaticMetric, IMPBASEVisitor):
    def __init__(self):
        self._init()

    # fed

    def _init(self):
        self._num_if: int = 0
        self._num_while: int = 0
        self._num_and: int = 0
        self._num_or: int = 0
        self._num_max_depth: int = 0
        self._num_max_nest_loop: int = 0
        self._num_max_nest_if: int = 0
        self._inside_if: int = 0
        self._inside_loop: int = 0
    # fed

    def get_extended_cyclomatic_metrics(self, program: str) -> dict:
        input_stream = InputStream(program)
        lexer = IMPBASELexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = IMPBASEParser(token_stream)
        self._init()
        tree = parser.program()
        self.visit(tree)
        return self.compute_extended_cyclomatic_metrics(
            self._num_if, self._num_while, self._num_and, self._num_or, self._num_max_depth, self._num_max_nest_loop, self._num_max_nest_if
        )

    # fed

    def visitProgram(self, ctx: IMPBASEParser.ProgramContext):
        if ctx.stmt_list():
            self.visit(ctx.stmt_list())
        # fi

    # fed

    def visitIfElseStmt(self, ctx: IMPBASEParser.IfElseStmtContext):
        self._num_if += 1
        self._inside_if += 1
        self._num_max_nest_if = max(self._num_max_nest_if, self._inside_if)        
        self.visit(ctx.bexp())
        self.visit(ctx.stmt_list(0))
        self.visit(ctx.stmt_list(1))
        self._inside_if -= 1
    # fed

    def visitWhileStmt(self, ctx: IMPBASEParser.WhileStmtContext):
        self._num_while += 1
        self._inside_loop += 1
        self._num_max_nest_loop = max(self._num_max_nest_loop, self._inside_loop)        
        self.visit(ctx.bexp())
        self.visit(ctx.stmt_list())
        self._inside_loop -= 1
    # fed

    def visitStmt_list(self, ctx: IMPBASEParser.Stmt_listContext):
        self._num_max_depth = max(self._num_max_depth, max(self._inside_loop, self._inside_if))
        for stmt in ctx.stmt():
            self.visit(stmt)
        # rof

    # fed

    def visitAddsubexp(self, ctx: IMPBASEParser.AddsubexpContext):
        self.visit(ctx.muldivexp(0))
        n: int = len(ctx.ADDSUBOP())
        for i in range(n):
            self.visit(ctx.muldivexp(i + 1))
        # rof

    # fed

    def visitMuldivexp(self, ctx: IMPBASEParser.MuldivexpContext):
        self.visit(ctx.unaryexp(0))
        n: int = len(ctx.MULDIVOP())
        for i in range(n):
            self.visit(ctx.unaryexp(i + 1))
        # rof

    # fed

    def visitUnaryexp(self, ctx: IMPBASEParser.UnaryexpContext):
        self.visit(ctx.atomexp())

    # fed

    def visitLogicalexp(self, ctx: IMPBASEParser.LogicalexpContext):
        self.visit(ctx.notexp(0))
        n: int = len(ctx.LOGICALOP())
        for i in range(n):
            match ctx.LOGICALOP(i):
                case "&&":
                    self._num_and += 1
                case "||":
                    self._num_or += 1
            self.visit(ctx.notexp(i + 1))
        # rof

    # fed

    def visitNotexp(self, ctx: IMPBASEParser.NotexpContext):
        self.visit(ctx.relexp())

    # fed

    def visitRelexp(self, ctx: IMPBASEParser.RelexpContext):
        self.visit(ctx.boolatomexp(0))
        n: int = len(ctx.RELOP())
        for i in range(n):
            self.visit(ctx.boolatomexp(i + 1))
        # rof

    # fed

    def visitAtomexp(self, ctx: IMPBASEParser.AtomexpContext):
        if ctx.aexp():
            self.visit(ctx.aexp())
        # fi

    # fed

    def visitBoolatomexp(self, ctx: IMPBASEParser.BoolatomexpContext):
        if ctx.aexp():
            self.visit(ctx.aexp())
        elif ctx.bexp():
            self.visit(ctx.bexp())
        # fi

    # fed


# ssalc
