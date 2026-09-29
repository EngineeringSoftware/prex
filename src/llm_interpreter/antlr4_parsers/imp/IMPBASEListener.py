# Generated from IMPBASE.g4 by ANTLR 4.13.2
from antlr4 import *

if "." in __name__:
    from .IMPBASEParser import IMPBASEParser
else:
    from IMPBASEParser import IMPBASEParser


# This class defines a complete listener for a parse tree produced by IMPBASEParser.
class IMPBASEListener(ParseTreeListener):
    # Enter a parse tree produced by IMPBASEParser#program.
    def enterProgram(self, ctx: IMPBASEParser.ProgramContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#program.
    def exitProgram(self, ctx: IMPBASEParser.ProgramContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#ids.
    def enterIds(self, ctx: IMPBASEParser.IdsContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#ids.
    def exitIds(self, ctx: IMPBASEParser.IdsContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#stmt_list.
    def enterStmt_list(self, ctx: IMPBASEParser.Stmt_listContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#stmt_list.
    def exitStmt_list(self, ctx: IMPBASEParser.Stmt_listContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#DeclStmt.
    def enterDeclStmt(self, ctx: IMPBASEParser.DeclStmtContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#DeclStmt.
    def exitDeclStmt(self, ctx: IMPBASEParser.DeclStmtContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#AssignStmt.
    def enterAssignStmt(self, ctx: IMPBASEParser.AssignStmtContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#AssignStmt.
    def exitAssignStmt(self, ctx: IMPBASEParser.AssignStmtContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#IfElseStmt.
    def enterIfElseStmt(self, ctx: IMPBASEParser.IfElseStmtContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#IfElseStmt.
    def exitIfElseStmt(self, ctx: IMPBASEParser.IfElseStmtContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#IfStmt.
    def enterIfStmt(self, ctx: IMPBASEParser.IfStmtContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#IfStmt.
    def exitIfStmt(self, ctx: IMPBASEParser.IfStmtContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#WhileStmt.
    def enterWhileStmt(self, ctx: IMPBASEParser.WhileStmtContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#WhileStmt.
    def exitWhileStmt(self, ctx: IMPBASEParser.WhileStmtContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#HaltStmt.
    def enterHaltStmt(self, ctx: IMPBASEParser.HaltStmtContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#HaltStmt.
    def exitHaltStmt(self, ctx: IMPBASEParser.HaltStmtContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#ContinueStmt.
    def enterContinueStmt(self, ctx: IMPBASEParser.ContinueStmtContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#ContinueStmt.
    def exitContinueStmt(self, ctx: IMPBASEParser.ContinueStmtContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#BreakStmt.
    def enterBreakStmt(self, ctx: IMPBASEParser.BreakStmtContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#BreakStmt.
    def exitBreakStmt(self, ctx: IMPBASEParser.BreakStmtContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#aexp.
    def enterAexp(self, ctx: IMPBASEParser.AexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#aexp.
    def exitAexp(self, ctx: IMPBASEParser.AexpContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#addsubexp.
    def enterAddsubexp(self, ctx: IMPBASEParser.AddsubexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#addsubexp.
    def exitAddsubexp(self, ctx: IMPBASEParser.AddsubexpContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#muldivexp.
    def enterMuldivexp(self, ctx: IMPBASEParser.MuldivexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#muldivexp.
    def exitMuldivexp(self, ctx: IMPBASEParser.MuldivexpContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#unaryexp.
    def enterUnaryexp(self, ctx: IMPBASEParser.UnaryexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#unaryexp.
    def exitUnaryexp(self, ctx: IMPBASEParser.UnaryexpContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#bexp.
    def enterBexp(self, ctx: IMPBASEParser.BexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#bexp.
    def exitBexp(self, ctx: IMPBASEParser.BexpContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#logicalexp.
    def enterLogicalexp(self, ctx: IMPBASEParser.LogicalexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#logicalexp.
    def exitLogicalexp(self, ctx: IMPBASEParser.LogicalexpContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#notexp.
    def enterNotexp(self, ctx: IMPBASEParser.NotexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#notexp.
    def exitNotexp(self, ctx: IMPBASEParser.NotexpContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#relexp.
    def enterRelexp(self, ctx: IMPBASEParser.RelexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#relexp.
    def exitRelexp(self, ctx: IMPBASEParser.RelexpContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#atomexp.
    def enterAtomexp(self, ctx: IMPBASEParser.AtomexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#atomexp.
    def exitAtomexp(self, ctx: IMPBASEParser.AtomexpContext):
        pass

    # Enter a parse tree produced by IMPBASEParser#boolatomexp.
    def enterBoolatomexp(self, ctx: IMPBASEParser.BoolatomexpContext):
        pass

    # Exit a parse tree produced by IMPBASEParser#boolatomexp.
    def exitBoolatomexp(self, ctx: IMPBASEParser.BoolatomexpContext):
        pass


del IMPBASEParser
