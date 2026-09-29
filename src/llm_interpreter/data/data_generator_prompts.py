translation_c_to_imp_prompt = """As an expert in programming languages, your primary objective is to generate an **executable and valid IMP program** based on the provided C++ code. The generated IMP program **must strictly adhere to all specified grammar rules and constraints**.

## IMP Grammar:

```
program     : decl
            | stmt
            | decl stmts;
decl        : 'int' ids ';' ;
ids         : ID (',' ID)* (',' '.Ids')? ;
stmts       : (stmt)* ;
stmt        : block                                 # BlockStmt
            | ID '=' aexp ';'                     # AssignStmt
            | 'if' '(' bexp ')' block 'else' block  # IfElseStmt
            | 'if' '(' bexp ')' block               # IfStmt
            | 'while' '(' bexp ')' block            # WhileStmt
            | 'halt' ';'                          # HaltStmt
            | 'continue' ';'                      # ContinueStmt
            | 'break' ';'                         # BreakStmt
            ;
block       : '{{' stmts '}}';
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
```

## Strict Instructions for IMP Program Generation:

  1.  **All Variables Must Be Declared at the Start:** Every variable used in the IMP code **must be declared first and only once** at the very beginning of the program. Variables not explicitly assigned a value will default to 0.
  2.  **Input Simulation via Test Case:** If the C++ program uses input (e.g., `scanf`, `cin`), directly assign the values from the provided test case to the corresponding IMP variables **before any other operations**.
  3.  **Unsupported C++ Features: Simplified Workaround:** When encountering C++ features or code snippets that are not supported by IMP (such as strings, arrays, user-defined functions, advanced data structures, or complex C++ constructs that cannot be directly and functionally converted), the LLM **must generate a simplified, valid, and executable IMP program snippet as a workaround**. This workaround should maintain the overall program flow where possible but **should not attempt to functionally replicate the unsupported C++ logic if a direct translation is not feasible**. For example, if a complex C++ function cannot be converted, simply use a supported computation. **Do not attempt to simulate unsupported features with complex or non-compliant workarounds.**
  4.  **No C++ Libraries or Built-in Functions:** The IMP program **must NOT use any external libraries, built-in functions, or system calls**. All operations must be implemented using only the basic constructs defined in the IMP grammar.
  5.  **Strict Adaptation of Logic to IMP:** The C++ code's logic **must be fully adapted to fit within the strict constraints of IMP**. This includes, but is not limited to, restructuring `for` loops into `while` loops, simplifying complex conditional statements, and breaking down intricate computations into basic arithmetic or variable assignments.
  6.  **Mandatory `ans` Variable:** The generated IMP translation **must include a variable named `ans`**. This variable **will hold the final computed result** or simulate the return value of the original C++ function.

---

### IMP Constraints (Strict Adherence Required):

  * **Integer-Only:** All variables and operations **must strictly use integer types**. No floating-point numbers or other data types are allowed.
  * **While Loops Only:** **`for` loops are strictly forbidden.** All iterative logic must be implemented using `while` loops.
  * **No Functions:** IMP **does not support any form of functions**, procedures, or subroutines. All code must reside within the main program flow.
  * **No Data Structures:** Arrays, structs, classes, or any other complex data structures are **strictly unsupported**. Only simple integer variables are permitted.
  * **No OOP:** Object-oriented programming constructs are **not present and are forbidden**.
  * **Single Program Flow:** The IMP program will execute sequentially from top to bottom.
  * **`halt` Statement**: `halt` can be used to terminate the program execution.
  * **Prioritize Simplicity for Untranslatable Logic:** If a C++ code segment cannot be directly or functionally translated into IMP while adhering to all constraints, **prioritize generating a minimal, valid IMP snippet that maintains program executability rather than attempting a complex, non-compliant, or approximate functional translation.**

Generate the IMP program based on the following C++ code. The generated IMP program **must be complete, valid according to the provided grammar, executable, and must strictly follow all instructions and constraints.**
```cpp
{code}
```

Here is the public test case for the C++ code. Use this to simulate inputs.
```
{test}
```

Here is one example of a valid IMP program for reference:
```imp
int a, b, ans;
ans = 0;
a = (3045 %% ans);
b = 1078;
ans = (a + b);
```

Task:
Generate the IMP program that adheres to all the rules, instructions, grammar, and constraints detailed above. Wrap the generated IMP code in a imp block.

"""


dgc_prompt = """You are an expert in programming languages. I designed a new programming language called IMP that is similiar to C.
I need your help to translate a C program into a *syntactically-valid* IMP program.

## IMP grammar

Below is the grammar for `IMP`:
```
program     : decl
            | stmt
            | decl stmts;
decl        : 'int' ids ';' ;
ids         : ID (',' ID)* (',' '.Ids')? ;
stmts       : (stmt)* ;
stmt        : block                                 # BlockStmt
            | ID '=' aexp ';'                       # AssignStmt
            | 'if' '(' bexp ')' block 'else' block  # IfElseStmt
            | 'if' '(' bexp ')' block               # IfStmt
            | 'while' '(' bexp ')' block            # WhileStmt
            | 'halt' ';'                            # HaltStmt
            | 'continue' ';'                        # ContinueStmt
            | 'break' ';'                           # BreakStmt
            ;
block       : '{{' stmts '}}';
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
```

## Notes

There are some notable differences between C and IMP you should be aware of when translating:
1. All the variables must be declared at the beginning of the program in IMP.
2. IMP does not support `for` loops. You should convert all `for` loops into `while` loops.
3. IMP does not support increment or decrement like '+=' and '-='.
4. IMP does not support casting, so expressions like `a = (10 > b)` are not allowed and you can not use integers directly in boolean conditions like `if (a)`. Instead you should use `if (a != 0)`.


You are allowed to make some changes to the original C program to accommodate the differences between C and IMP. **The most important thing is to ensure the translated program is syntactically valid in IMP.**

## Examples

Below are two examples of valid IMP program for reference:

Example 1:
```
int n, a, b, i, ans;
n = 6;
a = 1;
b = 1;
i = 3;
if((n == 1) || (n == 2))
{{
    ans = 1;
}}
else
{{
    while(i <= n)
    {{
        ans = (a + b);
        b = a;
        a = ans;
        i = (i + 1);
    }}
}}
```

Example 2:
```
int ans, i, n;
i = 2;
n = 10;
while((i * i) <= n)
{{
    if((n % i) == 0)
    {{
        ans = (n / i);
        halt;
    }}
    i = (i + 1);
}}
ans = 1;
```

## Task

Your task is to translate the following C code into a *syntactically-valid* IMP program:
```c
{code}
```

Please make sure the translated program adheres to the grammar rules listed above and be careful about the notes on some notable differences.

Wrap the generated IMP code in a block and do not include any additional text or comments.

"""


dgc_new_syntax_prompt = """You are an expert in programming languages. I designed a new programming language called `IMP` that is similiar to C.
I need your help to translate a C program into a *syntactically-valid* IMP program.

## IMP grammar

Below is the grammar for `IMP`:
```
program     : stmt_list;
stmt_list   : (stmt ';')* ;
stmt        : 'int' ID                                                  # DeclStmt
            | ID '=' aexp                                               # AssignStmt
            | 'if' bexp '{{' stmt_list '}}' 'else' '{{' stmt_list '}}'  # IfElseStmt
            | 'while' bexp '{{' stmt_list '}}'                          # WhileStmt
            | 'halt'                                                    # HaltStmt
            | 'continue'                                                # ContinueStmt
            | 'break'                                                   # BreakStmt
            ;

aexp        : ID                                      # AId
            | LITERAL                                 # ALit
            | '(' aexp MATHOP aexp ')'                # ABinary
            | '(' '+' aexp ')'                        # APlus
            | '(' '-' aexp ')'                        # AMinus
            ;

bexp        : '(' BOOL ')'                            # BLit
            | '(' aexp RELOP aexp ')'                 # BRel
            | '(' '!' bexp ')'                        # BNot
            | '(' bexp LOGICALOP bexp ')'             # BBin
            ;

BOOL        : 'true' | 'false' ;
MATHOP      : '+' | '-' | '*' | '/' | '%' ;
RELOP       : '<' | '<=' | '>' | '>=' | '==' | '!=' ;
LOGICALOP   : '&&' | '||' ;
ID          : LETTER (LETTER | DIGIT)* ;
LITERAL     : DIGIT+ ;
WS          : [ \t\r\n]+ -> skip ;
fragment LETTER : [a-zA-Z] ;
fragment DIGIT  : [0-9] ;
```

## Notes

There are some notable differences between C and IMP you should be aware of when translating:
1. All the variables must be declared at the beginning of the program in IMP.
2. IMP does not support `for` loops. You should convert all `for` loops into `while` loops.
3. IMP does not support increment or decrement like '+=' and '-='.
4. IMP does not support casting, so expressions like `a = (10 > b)` are not allowed.
5. IMP only supports 'if else' statement, so if there is no 'else' part, you should add an empty 'else' branch.


You are allowed to make some changes to the original C program to accommodate the differences between C and IMP. **The most important thing is to ensure that the translated program is syntactically valid in IMP.**

## Examples

Below are two examples of valid `IMP` program:

Example 1:
```
int ans;
int i;
int n;
int q;
ans = 0;
i = 0;
n = 100;
while(i < n)
{{
    if(((i %% 11) == 0) || ((i %% 13) == 0))
    {{
        q = i;
        while(q > 0)
        {{
            if((q %% 10) == 7)
            {{
                ans = (ans + 1);
            }}
            else
            {{
                ans = (ans + 1);
            }};
            q = (q / 10);
        }};
    }}
    else
    {{
        ans = (ans + 1);
    }};
    i = (i + 1);
}};
```

Example 2:
```
int ans;
int n;
int x;
x = 2;
ans = 1;
n = 10;
while(n != 0)
{{
    if((n %% 2) != 0)
    {{
        ans = (ans * x);
    }}
    else
    {{

    }};
    x = (x * x);
    n = (n / 2);
}};
```

## Task

Your task is to translate the following C code into a *syntactically-valid* IMP program:
```c
{code}
```

Please make sure the translated program adheres to the grammar rules listed above and be careful about the notes on some notable differences.

Wrap the generated IMP code in a block and do not include any additional text or comments.

"""
