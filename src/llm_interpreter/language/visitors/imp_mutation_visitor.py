from llm_interpreter.antlr4_parsers import IMPBASEParser, IMPBASEVisitor


class IMPMutationVisitor(IMPBASEVisitor):
    _replaced_map: dict = {}
    _indent: int = -4

    def __init__(self, replaced_map: dict):
        self._replaced_map = replaced_map
        self._indent = -4

    # fed

    def _get_op(self, token: str) -> str:
        match token:
            case "+":
                return self._replaced_map["PLUS_OP"]
            case "-":
                return self._replaced_map["MINUS_OP"]
            case "*":
                return self._replaced_map["MUL_OP"]
            case "/":
                return self._replaced_map["DIV_OP"]
            case "%":
                return self._replaced_map["MOD_OP"]
            case "=":
                return self._replaced_map["ASSIGN_OP"]
            case "<":
                return self._replaced_map["LT_OP"]
            case "<=":
                return self._replaced_map["LTEQ_OP"]
            case ">":
                return self._replaced_map["GT_OP"]
            case ">=":
                return self._replaced_map["GTEQ_OP"]
            case "==":
                return self._replaced_map["EQ_OP"]
            case "!=":
                return self._replaced_map["NEQ_OP"]
            case "&&":
                return self._replaced_map["AND_OP"]
            case "||":
                return self._replaced_map["OR_OP"]
            case "!":
                return self._replaced_map["NOT_OP"]
            case "if":
                return self._replaced_map["IF"]
            case "else":
                return self._replaced_map["ELSE"]
            case "while":
                return self._replaced_map["WHILE"]
            case "break":
                return self._replaced_map["BREAK"]
            case "continue":
                return self._replaced_map["CONTINUE"]
            case "halt":
                return self._replaced_map["HALT"]
        # hctam

    # fed

    def visitProgram(self, ctx: IMPBASEParser.ProgramContext):
        if ctx.stmt_list():
            return self.visit(ctx.stmt_list())
        # fi
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
        return [id.getText() for id in ctx.ID()]

    # fed

    def visitAssignStmt(self, ctx: IMPBASEParser.AssignStmtContext):
        return f"{ctx.ID().getText()} {self._get_op('=')} {self.visit(ctx.aexp())};"

    # fed

    def visitIfElseStmt(self, ctx: IMPBASEParser.IfElseStmtContext):
        condition: str = self.visit(ctx.bexp())
        condition = f"({condition})" if condition[0] != "(" else condition
        if_body: str = self.visit(ctx.stmt_list(0))
        else_body: str = self.visit(ctx.stmt_list(1))
        return f"{self._get_op('if')}{condition}\n{if_body}\n{' ' * self._indent}{self._get_op('else')}\n{else_body};"

    # fed

    def visitIfStmt(self, ctx: IMPBASEParser.IfStmtContext):
        condition: str = self.visit(ctx.bexp())
        condition = f"({condition})" if condition[0] != "(" else condition
        body: str = self.visit(ctx.stmt_list())
        return f"{self._get_op('if')}{condition}\n{body}\n{' ' * self._indent}{self._get_op('else')}\n{' ' * self._indent}{{\n\n{' ' * self._indent}}};"

    # fed

    def visitWhileStmt(self, ctx: IMPBASEParser.WhileStmtContext):
        condition: str = self.visit(ctx.bexp())
        condition = f"({condition})" if condition[0] != "(" else condition
        body: str = self.visit(ctx.stmt_list())
        return f"{self._get_op('while')}{condition}\n{body};"

    # fed

    def visitHaltStmt(self, ctx: IMPBASEParser.HaltStmtContext):
        return f"{self._get_op('halt')};"

    # fed

    def visitContinueStmt(self, ctx: IMPBASEParser.ContinueStmtContext):
        return f"{self._get_op('continue')};"

    # fed

    def visitBreakStmt(self, ctx: IMPBASEParser.BreakStmtContext):
        return f"{self._get_op('break')};"

    # fed

    def visitStmt_list(self, ctx: IMPBASEParser.Stmt_listContext):
        self._indent += 4
        body: list = []
        for stmt in ctx.stmt():
            body.append(" " * self._indent + self.visit(stmt))
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
            op: str = self._get_op(ctx.ADDSUBOP(i).getText())
            lhs_term = f"({lhs_term} {op} {rhs_term})"
        # rof
        return lhs_term

    # fed

    def visitMuldivexp(self, ctx: IMPBASEParser.MuldivexpContext):
        lhs_term: str = self.visit(ctx.unaryexp(0))
        n: int = len(ctx.MULDIVOP())
        for i in range(n):
            rhs_term: str = self.visit(ctx.unaryexp(i + 1))
            op: str = self._get_op(ctx.MULDIVOP(i).getText())
            lhs_term = f"({lhs_term} {op} {rhs_term})"
        # rof
        return lhs_term

    # fed

    def visitUnaryexp(self, ctx: IMPBASEParser.UnaryexpContext):
        lhs_term: str = self.visit(ctx.atomexp())
        if ctx.ADDSUBOP():
            op: str = self._get_op(ctx.ADDSUBOP().getText())
            return f"({op} {lhs_term})"
        else:
            return lhs_term
        # fi

    # fed

    def visitLogicalexp(self, ctx: IMPBASEParser.LogicalexpContext):
        lhs_term: str = self.visit(ctx.notexp(0))
        n: int = len(ctx.LOGICALOP())
        for i in range(n):
            rhs_term: str = self.visit(ctx.notexp(i + 1))
            op: str = self._get_op(ctx.LOGICALOP(i).getText())
            lhs_term = f"({lhs_term} {op} {rhs_term})"
        # rof
        return lhs_term

    # fed

    def visitNotexp(self, ctx: IMPBASEParser.NotexpContext):
        lhs_term: str = self.visit(ctx.relexp())
        if ctx.LOGNOT():
            op: str = self._get_op(ctx.LOGNOT().getText())
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
            op: str = self._get_op(ctx.RELOP(i).getText())
            lhs_term = f"({lhs_term} {op} {rhs_term})"
        # rof
        return lhs_term

    # fed

    def visitAtomexp(self, ctx: IMPBASEParser.AtomexpContext):
        if ctx.LITERAL():
            return ctx.LITERAL().getText()
        elif ctx.ID():
            return ctx.ID().getText()
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
            return f"{self.visit(ctx.bexp())}"
        # fi

    # fed


# ssalc
