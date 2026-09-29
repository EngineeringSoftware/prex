# Generated from IMPBASE.g4 by ANTLR 4.13.2
from antlr4 import *

if "." in __name__:
    from .IMPBASEParser import IMPBASEParser
else:
    from IMPBASEParser import IMPBASEParser

# This class defines a complete generic visitor for a parse tree produced by IMPBASEParser.


class IMPBASEVisitor(ParseTreeVisitor):
    # Visit a parse tree produced by IMPBASEParser#program.
    def visitProgram(self, ctx: IMPBASEParser.ProgramContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#ids.
    def visitIds(self, ctx: IMPBASEParser.IdsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#stmt_list.
    def visitStmt_list(self, ctx: IMPBASEParser.Stmt_listContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#DeclStmt.
    def visitDeclStmt(self, ctx: IMPBASEParser.DeclStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#AssignStmt.
    def visitAssignStmt(self, ctx: IMPBASEParser.AssignStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#IfElseStmt.
    def visitIfElseStmt(self, ctx: IMPBASEParser.IfElseStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#IfStmt.
    def visitIfStmt(self, ctx: IMPBASEParser.IfStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#WhileStmt.
    def visitWhileStmt(self, ctx: IMPBASEParser.WhileStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#HaltStmt.
    def visitHaltStmt(self, ctx: IMPBASEParser.HaltStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#ContinueStmt.
    def visitContinueStmt(self, ctx: IMPBASEParser.ContinueStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#BreakStmt.
    def visitBreakStmt(self, ctx: IMPBASEParser.BreakStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#aexp.
    def visitAexp(self, ctx: IMPBASEParser.AexpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#addsubexp.
    def visitAddsubexp(self, ctx: IMPBASEParser.AddsubexpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#muldivexp.
    def visitMuldivexp(self, ctx: IMPBASEParser.MuldivexpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#unaryexp.
    def visitUnaryexp(self, ctx: IMPBASEParser.UnaryexpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#bexp.
    def visitBexp(self, ctx: IMPBASEParser.BexpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#logicalexp.
    def visitLogicalexp(self, ctx: IMPBASEParser.LogicalexpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#notexp.
    def visitNotexp(self, ctx: IMPBASEParser.NotexpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#relexp.
    def visitRelexp(self, ctx: IMPBASEParser.RelexpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#atomexp.
    def visitAtomexp(self, ctx: IMPBASEParser.AtomexpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by IMPBASEParser#boolatomexp.
    def visitBoolatomexp(self, ctx: IMPBASEParser.BoolatomexpContext):
        return self.visitChildren(ctx)


del IMPBASEParser
