import sys
import json
from antlr4 import *
from MyLangLexer import MyLangLexer
from MyLangParser import MyLangParser
from EvalVisitor import EvalVisitor


def main():
    input_stream = FileStream(sys.argv[1], encoding="utf-8")

    lexer = MyLangLexer(input_stream)
    tokens = CommonTokenStream(lexer)
    parser = MyLangParser(tokens)
    tree = parser.program()

    visitor = EvalVisitor()
    visitor.visit(tree)

    print(json.dumps(visitor.states, sort_keys=True))


if __name__ == "__main__":
    main()
