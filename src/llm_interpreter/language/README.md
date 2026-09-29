# IMP Semantics Rule Maps

## K-Framework Rules (36 rules)

| Rule # | Category | Description (summary) |
|--------|----------|------------------------|
| 1–2 | Values / Variables | Variable lookup; undefined variable → ERROR |
| 3–11 | Arithmetic | `+`, `-`, `*`, `/`, `%`, unary `+`/`-` on values |
| 7, 9 | Arithmetic errors | Division/modulo by zero → ERROR |
| 12–20 | Relational / Boolean | Comparisons, `!`, `&&`, `\|\|` on values |
| 21 | Assignment | Update declared variable |
| 22–23 | Conditional | `if`/`else` branch selection |
| 24–25 | While loop | Loop setup and `WHILE1` unfolding |
| 26 | Halt | Terminate execution |
| 27–35 | Break / Continue | Markers, propagation, out-of-loop errors |
| 36 | Declaration | `int x;` initializes to 0 |

Generated LaTeX table (`papers/lmpl26/tables/imp_k_rules_table.tex`) columns: **Rule** | **Full rule** (``k.txt`` layout) | **Description**. Rule bodies from `imp/k.txt`; descriptions from `RULE_DESCRIPTIONS_NOTATION_COMPREHENSION_K`. Style matches SOS `imp_rules_table.tex` (vertical bars, `\hline` between rows).

---

## IMP SOS Rule Map

| Rule # | Category               | Description                                                              |
|--------|------------------------|--------------------------------------------------------------------------|
| 1–2    | Values and Variables   | Integer literal evaluation and variable lookup                          |
| 3      | Declaration            | Declare uninitialized variable (`int x;`)                               |
| 4      | Assignment             | Assign evaluated expression to declared variable                        |
| 5–9    | Arithmetic Ops         | Evaluation of `+`, `-`, `*`, `/`, `%` with step-by-step breakdown       |
| 10–21  | Relational Ops         | Evaluation of `<`, `<=`, `>`, `>=`, `==`, `!=`                          |
| 22–25  | Boolean Ops            | `&&`, `\|\|` with truth table expansion                                   |
| 26–27  | Boolean Literals       | Evaluation of `true` and `false`                                        |
| 28–29  | Boolean Negation       | Evaluation of `!b`                                                      |
| 30–31  | If (no else)           | `if (b) {S}` with branch execution                                      |
| 32–33  | If-else                | `if (b) {S1} else {S2}` branch selection                                |
| 34–35  | While loop             | Loop unfolding when condition is true/false                             |
| 36     | Sequential Composition | Executing `S1; S2` in order                                             |
| 37     | Block Execution        | Unwrap block `{S1}` into `S1`                                           |
| 38     | Multi-declaration      | Multiple variable declarations `int x1, x2;`                            |
| 39     | Redeclaration Error    | HALT on re-declaring already-declared variable                         |
| 40     | Assignment Error       | HALT on assigning to undeclared variable                               |
| 41     | Halt                   | `halt;` terminates program                                              |
| 42     | While with Continue    | Loop re-entry after `continue`                                         |
| 43     | While with Break       | Exit loop early on `break`                                             |
| 44–47  | Precedence Rules       | Associativity and precedence for arithmetic combinations (e.g., `+`, `*`, `/`) |
