grammar IMPBASE;

program     : stmt_list;
ids         : ID (',' ID)* (',' '.Ids')? ;
stmt_list   : (stmt ';'?)* ;
stmt        : 'int' ids                                                    # DeclStmt
            | ID '=' aexp                                                  # AssignStmt
            | 'if' '(' bexp ')' '{' stmt_list '}' 'else' '{' stmt_list '}' # IfElseStmt
            | 'if' '(' bexp ')' '{' stmt_list '}'                          # IfStmt
            | 'while' '(' bexp ')' '{' stmt_list '}'                       # WhileStmt
            | 'halt'                                                       # HaltStmt
            | 'continue'                                                   # ContinueStmt
            | 'break'                                                      # BreakStmt
            ;
aexp        : addsubexp;
addsubexp   : muldivexp (ADDSUBOP muldivexp)* ;
muldivexp   : unaryexp (MULDIVOP unaryexp)* ;
unaryexp    : ADDSUBOP? atomexp ;

bexp        : logicalexp ;
logicalexp  : notexp (LOGICALOP notexp)* ;
notexp      : LOGNOT? '('? relexp ')'?;
relexp      : boolatomexp (RELOP boolatomexp)* ;

atomexp     : LITERAL | ID  | '(' aexp ')' ;
boolatomexp : aexp | '(' bexp ')' | BOOL ;

BOOL        : 'true' | 'false' ;
ADDSUBOP    : '+' | '-' ;
MULDIVOP    : '*' | '/' | '%' ;
RELOP       : '<' | '<=' | '>' | '>=' | '==' | '!=' ;
LOGNOT      : '!' ;
LOGICALOP   : '&&' | '||' ;
ID          : LETTER (LETTER | DIGIT)* ;
LITERAL     : DIGIT+ ;
WS          : [ \t\r\n]+ -> skip ;
fragment LETTER : [a-zA-Z] ;
fragment DIGIT  : [0-9] ;
