from antlr4 import *
from llm_interpreter.antlr4_parsers import IMPBASELexer, IMPBASEParser, IMPBASEVisitor
from llm_interpreter.language.metrics.base import DepDegreeMetric


class IMPDepDegreeVisitor(DepDegreeMetric, IMPBASEVisitor):
    def __init__(self):
        self._init()

    # fed

    def _init(self):
        self._node_counter = -1
        self._nodes: list = []

    # fed

    def get_depdegree_metrics(self, program: str) -> dict:
        input_stream = InputStream(program)
        lexer = IMPBASELexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = IMPBASEParser(token_stream)
        self._init()
        tree = parser.program()
        self.visit(tree)
        return self.compute_depdegree(self._nodes)

    # fed

    def visitProgram(self, ctx: IMPBASEParser.ProgramContext):
        if ctx.stmt_list():
            self.visit(ctx.stmt_list())
        # fi

    # fed

    def visitDeclStmt(self, ctx: IMPBASEParser.DeclStmtContext):
        defs: set[str] = set()
        if ctx.ids():
            defs: set[str] = self.visit(ctx.ids())
        # fi
        self._nodes.append(
            DepDegreeMetric.Node(
                f"B{self._node_counter}", defs, set(), [self._node_counter + 1]
            )
        )

    # fed

    def visitIds(self, ctx: IMPBASEParser.IdsContext):
        defs: set[str] = set()
        for id in ctx.ID():
            defs.add(id.getText())
        # rof
        return defs

    # fed

    def visitAssignStmt(self, ctx: IMPBASEParser.AssignStmtContext):
        uses: set[str] = self.visit(ctx.aexp())
        self._nodes.append(
            DepDegreeMetric.Node(
                f"B{self._node_counter}",
                set([ctx.ID().getText()]),
                uses,
                [self._node_counter + 1],
            )
        )

    # fed

    def visitIfElseStmt(self, ctx: IMPBASEParser.IfElseStmtContext):
        uses: set[str] = self.visit(ctx.bexp())
        if_index: int = self._node_counter
        self._nodes.append(
            DepDegreeMetric.Node(
                f"B{self._node_counter}", set(), uses, [self._node_counter + 1]
            )
        )
        self.visit(ctx.stmt_list(0))
        if_end_index: int = self._node_counter
        self._nodes[if_index].succs.append(if_end_index + 1)
        self.visit(ctx.stmt_list(1))
        self._nodes[if_end_index].succs = [self._node_counter + 1]

    # fed

    def visitWhileStmt(self, ctx: IMPBASEParser.WhileStmtContext):
        uses: set[str] = self.visit(ctx.bexp())
        while_index: int = self._node_counter
        self._nodes.append(
            DepDegreeMetric.Node(
                f"B{self._node_counter}", set(), uses, [self._node_counter + 1]
            )
        )
        self.visit(ctx.stmt_list())
        self._nodes[self._node_counter].succs = [while_index]
        self._nodes[while_index].succs.append(self._node_counter + 1)

    # fed

    def visitHaltStmt(self, ctx: IMPBASEParser.HaltStmtContext):
        self._nodes.append(
            DepDegreeMetric.Node(
                f"B{self._node_counter}", set(), set(), [self._node_counter + 1]
            )
        )

    # fed

    def visitContinueStmt(self, ctx: IMPBASEParser.ContinueStmtContext):
        self._nodes.append(
            DepDegreeMetric.Node(
                f"B{self._node_counter}", set(), set(), [self._node_counter + 1]
            )
        )

    # fed

    def visitBreakStmt(self, ctx: IMPBASEParser.BreakStmtContext):
        self._nodes.append(
            DepDegreeMetric.Node(
                f"B{self._node_counter}", set(), set(), [self._node_counter + 1]
            )
        )

    # fed

    def visitStmt_list(self, ctx: IMPBASEParser.Stmt_listContext):
        for stmt in ctx.stmt():
            self._node_counter += 1
            self.visit(stmt)
        # rof

    # fed

    def visitAddsubexp(self, ctx: IMPBASEParser.AddsubexpContext):
        uses: set[str] = self.visit(ctx.muldivexp(0))
        n: int = len(ctx.ADDSUBOP())
        for i in range(n):
            uses = uses | self.visit(ctx.muldivexp(i + 1))
        # rof
        return uses

    # fed

    def visitMuldivexp(self, ctx: IMPBASEParser.MuldivexpContext):
        uses: set[str] = self.visit(ctx.unaryexp(0))
        n: int = len(ctx.MULDIVOP())
        for i in range(n):
            uses = uses | self.visit(ctx.unaryexp(i + 1))
        # rof
        return uses

    # fed

    def visitUnaryexp(self, ctx: IMPBASEParser.UnaryexpContext):
        return self.visit(ctx.atomexp())

    # fed

    def visitLogicalexp(self, ctx: IMPBASEParser.LogicalexpContext):
        uses: set[str] = self.visit(ctx.notexp(0))
        n: int = len(ctx.LOGICALOP())
        for i in range(n):
            uses = uses | self.visit(ctx.notexp(i + 1))
        # rof
        return uses

    # fed

    def visitNotexp(self, ctx: IMPBASEParser.NotexpContext):
        return self.visit(ctx.relexp())

    # fed

    def visitRelexp(self, ctx: IMPBASEParser.RelexpContext):
        uses: set[str] = self.visit(ctx.boolatomexp(0))
        n: int = len(ctx.RELOP())
        for i in range(n):
            uses = uses | self.visit(ctx.boolatomexp(i + 1))
        # rof
        return uses

    # fed

    def visitAtomexp(self, ctx: IMPBASEParser.AtomexpContext):
        if ctx.LITERAL():
            return set()
        elif ctx.ID():
            return set([ctx.ID().getText()])
        else:
            return self.visit(ctx.aexp())
        # fi

    # fed

    def visitBoolatomexp(self, ctx: IMPBASEParser.BoolatomexpContext):
        if ctx.BOOL():
            return set()
        elif ctx.aexp():
            return self.visit(ctx.aexp())
        elif ctx.bexp():
            return self.visit(ctx.bexp())
        # fi

    # fed


# ssalc
