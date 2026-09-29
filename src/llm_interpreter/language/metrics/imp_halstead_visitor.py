from antlr4 import *
from llm_interpreter.antlr4_parsers import IMPBASELexer, IMPBASEParser, IMPBASEVisitor
from llm_interpreter.language.metrics.base import HalsteadMetric


class IMPHalsteadVisitor(HalsteadMetric, IMPBASEVisitor):
    def __init__(self):
        self._init()

    # fed

    def _init(self):
        self._operators: list[str] = []
        self._operands: list[str] = []

    # fed

    def get_halstead_metrics(
        self, program: str, print_operators: bool = False, print_operands: bool = False
    ) -> dict:
        input_stream = InputStream(program)
        lexer = IMPBASELexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = IMPBASEParser(token_stream)
        self._init()
        tree = parser.program()
        self.visit(tree)
        if print_operators:
            print(self._operators)
        # fi
        if print_operands:
            print(self._operands)
        # fi
        distinct_operators: set[str] = set(self._operators)
        distinct_operands: set[str] = set(self._operands)
        num_distinct_operands: int = len(distinct_operands)
        num_distinct_operators: int = len(distinct_operators)
        total_operators: int = len(self._operators)
        total_operands: int = len(self._operands)
        return self.compute_halstead_metrics(
            num_distinct_operators,
            num_distinct_operands,
            total_operators,
            total_operands,
        )

    # fed

    def visitProgram(self, ctx: IMPBASEParser.ProgramContext):
        if ctx.stmt_list():
            self.visit(ctx.stmt_list())
        # fi

    # fed

    def visitDeclStmt(self, ctx: IMPBASEParser.DeclStmtContext):
        if ctx.ids():
            self._operators.append("int")
            self.visit(ctx.ids())
        # fi

    # fed

    def visitIds(self, ctx: IMPBASEParser.IdsContext):
        for id in ctx.ID():
            self._operands.append(id.getText())
        # rof

    # fed

    def visitAssignStmt(self, ctx: IMPBASEParser.AssignStmtContext):
        self._operands.append(ctx.ID().getText())
        self._operators.append("=")
        self.visit(ctx.aexp())

    # fed

    def visitIfElseStmt(self, ctx: IMPBASEParser.IfElseStmtContext):
        self._operators.append("if")
        self._operators.append("else")
        self._operators.append("{")
        self._operators.append("}")
        self._operators.append("{")
        self._operators.append("}")
        self._operators.append("(")
        self._operators.append(")")
        self.visit(ctx.bexp())
        self.visit(ctx.stmt_list(0))
        self.visit(ctx.stmt_list(1))

    # fed

    def visitWhileStmt(self, ctx: IMPBASEParser.WhileStmtContext):
        self._operators.append("while")
        self._operators.append("{")
        self._operators.append("}")
        self._operators.append("(")
        self._operators.append(")")
        self.visit(ctx.bexp())
        self.visit(ctx.stmt_list())

    # fed

    def visitHaltStmt(self, ctx: IMPBASEParser.HaltStmtContext):
        self._operators.append("halt")

    # fed

    def visitContinueStmt(self, ctx: IMPBASEParser.ContinueStmtContext):
        self._operators.append("continue")

    # fed

    def visitBreakStmt(self, ctx: IMPBASEParser.BreakStmtContext):
        self._operators.append("break")

    # fed

    def visitStmt_list(self, ctx: IMPBASEParser.Stmt_listContext):
        for stmt in ctx.stmt():
            self.visit(stmt)
            self._operators.append(";")
        # rof

    # fed

    def visitAddsubexp(self, ctx: IMPBASEParser.AddsubexpContext):
        self.visit(ctx.muldivexp(0))
        n: int = len(ctx.ADDSUBOP())
        for i in range(n):
            self.visit(ctx.muldivexp(i + 1))
            self._operators.append(ctx.ADDSUBOP(i).getText())
        # rof

    # fed

    def visitMuldivexp(self, ctx: IMPBASEParser.MuldivexpContext):
        self.visit(ctx.unaryexp(0))
        n: int = len(ctx.MULDIVOP())
        for i in range(n):
            self.visit(ctx.unaryexp(i + 1))
            self._operators.append(ctx.MULDIVOP(i).getText())
        # rof

    # fed

    def visitUnaryexp(self, ctx: IMPBASEParser.UnaryexpContext):
        self.visit(ctx.atomexp())
        if ctx.ADDSUBOP():
            self._operators.append(ctx.ADDSUBOP().getText())
        # fi

    # fed

    def visitLogicalexp(self, ctx: IMPBASEParser.LogicalexpContext):
        self.visit(ctx.notexp(0))
        n: int = len(ctx.LOGICALOP())
        for i in range(n):
            self.visit(ctx.notexp(i + 1))
            self._operators.append(ctx.LOGICALOP(i).getText())
        # rof

    # fed

    def visitNotexp(self, ctx: IMPBASEParser.NotexpContext):
        self.visit(ctx.relexp())
        if ctx.LOGNOT():
            self._operators.append(ctx.LOGNOT().getText())
        # fi

    # fed

    def visitRelexp(self, ctx: IMPBASEParser.RelexpContext):
        self.visit(ctx.boolatomexp(0))
        n: int = len(ctx.RELOP())
        for i in range(n):
            self.visit(ctx.boolatomexp(i + 1))
            self._operators.append(ctx.RELOP(i).getText())
        # rof

    # fed

    def visitAtomexp(self, ctx: IMPBASEParser.AtomexpContext):
        if ctx.LITERAL():
            self._operands.append(ctx.LITERAL().getText())
        elif ctx.ID():
            self._operands.append(ctx.ID().getText())
        else:
            self.visit(ctx.aexp())
            self._operators.append("(")
            self._operators.append(")")
        # fi

    # fed

    def visitBoolatomexp(self, ctx: IMPBASEParser.BoolatomexpContext):
        if ctx.BOOL():
            self._operands.append(ctx.BOOL().getText())
        elif ctx.aexp():
            self.visit(ctx.aexp())
        elif ctx.bexp():
            self.visit(ctx.bexp())
            self._operators.append("(")
            self._operators.append(")")
        # fi

    # fed


# ssalc
