from enum import StrEnum
import textwrap


class PROMPT_STRATEGY(StrEnum):
    COT = "cot"
    DA = "da"


system_prompt = """You are an expert in Programming Language and Formal Methods."""

concise_reason_prompt = """Keep the reasoning concise and to the point"""


# Task descriptions prompts
pep_task_desc = """
## TASK: predict if the given program can execute under the given semantics or not.
- If you think the program can execute successfully, answer with the special word '##success##':

    <ans>##success##</ans>

- If you believe the program cannot execute due to semantically invalid statements, answer with the special word '##error##' and the rule that prevents execution:

    <ans>##error##</ans>
    <rule>[Rule leading to the 〈{ERROR},𝜎,χ〉terminal configuration due to semantically invalid statement]</rule>

Here is an example:

** Program **
```
int a;
int b;
{CONTINUE};
```

The final expected output is:
<ans>##error##</ans>
<rule>76</rule>

Only write the answer. Note that you **MUST** wrap your prediction with `<ans>` tags and the violated rule with `<rule>` tags if any.
"""

pep_task_cot_desc = """
## TASK: predict if the given program can execute under the given semantics or not.
- If you think the program can execute successfully, answer with the special word '##success##':

    <ans>##success##</ans>

- If you believe the program cannot execute due to semantically invalid statements, answer with the special word '##error##' with the reason and the rule that prevents execution:

    <ans>##error##</ans>
    <rule>[Rule leading to the 〈{ERROR},𝜎,χ〉terminal configuration due to semantically invalid statement]</rule>

Here is an example:

** Program **
```
int a;
int b;
{CONTINUE};
```

The final expected output is:
<ans>##error##</ans>
<rule>76</rule>

Explain your reasoning step-by-step **before** answering. Wrap your reasoning in `<reason>` tags.
Note that you **MUST** wrap your reasoning steps with `<reason>` tags, the prediction with `<ans>` tags, and the violated rule with `<rule>` tags if any.
"""

pep_k_task_desc = """
## TASK: predict if the given program can execute under the given semantics or not.
- If you think the program can execute successfully, answer with the special word '##success##':

    <ans>##success##</ans>

- If you believe the program cannot execute due to semantically invalid statements, answer with the special word '##error##' and the rule that prevents execution:

    <ans>##error##</ans>
    <rule>[Re-write rule leading to the {ERROR} due to semantically invalid statement]</rule>

Here is an example:

** Program **
```
int a;
int b;
{CONTINUE};
```

The final expected output is:
<ans>##error##</ans>
<rule>31</rule>

The rule name/index is given in the `[]` next to the rule.
Only write the answer. Note that you **MUST** wrap your prediction with `<ans>` tags and the violated rule with `<rule>` tags if any.
"""

pep_k_task_cot_desc = """
## TASK: predict if the given program can execute under the given semantics or not.
- If you think the program can execute successfully, answer with the special word '##success##':

    <ans>##success##</ans>

- If you believe the program cannot execute due to semantically invalid statements, answer with the special word '##error##' and the rule that prevents execution:

    <ans>##error##</ans>
    <rule>[Re-write rule leading to the {ERROR} due to semantically invalid statement]</rule>

Here is an example:

** Program **
```
int a;
int b;
{CONTINUE};
```

The final expected output is:
<ans>##error##</ans>
<rule>31</rule>

The rule name/index is given in the `[]` next to the rule.
Explain your reasoning step-by-step **before** answering. Wrap your reasoning in `<reason>` tags.
Note that you **MUST** wrap your reasoning steps with `<reason>` tags, the prediction with `<ans>` tags, and the violated rule with `<rule>` tags if any.
"""

op_task_desc = """
## TASK: predict the value of the variable `ans` after executing the above program.
- If you think the program will never terminate, answer with the special word '##timeout##':

    <ans> ##timeout## </ans>

- If you believe the program has reached the terminal configuration 〈{ERROR},𝜎,χ〉 or has undefined behavior, answer with the special word '##error##':

    <ans> ##error## </ans>

- Otherwise, provide the predicted value of ans in the following format:

    <ans> [Your answer] </ans>

Output the statement at line 500 in the program.
You **MUST** wrap your prediction with `<ans>` tags.
"""

op_task_cot_desc = """
## TASK: predict the value of the variable `ans` after reasoning abou the execution of the above program.
- If you think the program will never terminate, answer with the special word '##timeout##':

    <ans> ##timeout## </ans>

- If you believe the program has reached the terminal configuration 〈{ERROR},𝜎,χ〉 or has undefined behavior, answer with the special word '##error##':

    <ans> ##error## </ans>

- Otherwise, provide the predicted value of ans in the following format:

    <ans> [Your answer] </ans>

Explain your reasoning step-by-step **before** answering. Wrap your reasoning in `<reason>` tags.
Note that you **MUST** wrap your reasoning steps with `<reason>` tags and the prediction with `<ans>` tags.
"""

new_op_task_desc = """
## TASK: predict the values of all the declared variables after executing the above program.
- If you think the program will never terminate, answer with the special word '##timeout##':

    <answer>##timeout##</answer>

- If you believe the program has an error or has undefined behavior, answer with the special word '##error##':

    <answer>##error##</answer>

- Otherwise, provide the predicted values of all the declared variables in the following format:

    <answer>[Your answer]</answer>


Here is one example:
** Program **
```
int a;
int b;
int ans;
int c;
a {ASSIGN_OP} 10;
b {ASSIGN_OP} 23;
c {ASSIGN_OP} 12;
ans {ASSIGN_OP} a {ADD_OP} b;
```

The final expected output is:
<answer>
<a>10</a>
<b>23</b>
<c>12</c>
<ans>33</ans>
</answer>

Only write the answer. You **MUST** wrap your prediction with `<answer>` tags.
"""

new_op_task_cot_desc = """
## TASK: predict the values of all the declared variables after executing the above program.
- If you think the program will never terminate, answer with the special word '##timeout##':

    <answer>##timeout##</answer>

- If you believe the program has an error or has undefined behavior, answer with the special word '##error##':

    <answer>##error##</answer>

- Otherwise, provide the predicted values of all the declared variables in the following format:

    <answer>[Your answer]</answer>


Here is one example:
** Program **
```
int a;
int b;
int ans;
int c;
a {ASSIGN_OP} 10;
b {ASSIGN_OP} 23;
c {ASSIGN_OP} 12;
ans {ASSIGN_OP} a {ADD_OP} b;
```

The final expected output is:
<answer>
<a>10</a>
<b>23</b>
<c>12</c>
<ans>33</ans>
</answer>

Explain your reasoning step-by-step **before** answering. Wrap your reasoning in `<reason>` tags.
Note that you **MUST** wrap your reasoning steps with `<reason>` tags and the prediction with `<answer>` tags.
"""

new_op_task_five_shot_desc = """
## TASK: predict the values of all the declared variables after executing the above program.
- If you think the program will never terminate, answer with the special word '##timeout##':

    <answer>##timeout##</answer>

- If you believe the program has an error or has undefined behavior, answer with the special word '##error##':

    <answer>##error##</answer>

- Otherwise, provide the predicted values of all the declared variables in the following format:

    <answer>[Your answer]</answer>


Here are five examples:

Example 1:
** Program **
```
int a;
int b;
int ans;
int c;
a {ASSIGN_OP} 10;
b {ASSIGN_OP} 23;
c {ASSIGN_OP} 12;
ans {ASSIGN_OP} a {ADD_OP} b;
```

The final expected output is:
<answer>
<a>10</a>
<b>23</b>
<c>12</c>
<ans>33</ans>
</answer>


Example 2:
** Program **
```
int a;
int b;
int c;
{WHILE}(c {LT_OP} 5)
{{
  a {ASSIGN_OP} (a {ADD_OP} 1);
  c {ASSIGN_OP} (c {ADD_OP} 1);
}};
```

The final expected output is:
<answer>
<a>5</a>
<b>0</b>
<c>5</c>
</answer>


Example 3:
** Program **
```
int a;
int b;
int c;
{WHILE}(c {LT_OP} 5)
{{
  a {ASSIGN_OP} (a {ADD_OP} 1);
  {IF}(a {GT_OP} 3)
  {{
    {BREAK};
  }}
  {ELSE}
  {{
    {CONTINUE};
  }};
  c {ASSIGN_OP} (c {ADD_OP} 1);
}};
```

The final expected output is:
<answer>
<a>4</a>
<b>0</b>
<c>0</c>
</answer>


Example 4:
** Program **
```
int a;
int b;
int c;
{WHILE}(c {LT_OP} 5)
{{
  a {ASSIGN_OP} (a {ADD_OP} 1);
  {IF}(a {GT_OP} 3)
  {{
    {BREAK};
  }}
  {ELSE}
  {{
    {HALT};
  }};
  c {ASSIGN_OP} (c {ADD_OP} 1);
}};
```

The final expected output is:
<answer>
<a>1</a>
<b>0</b>
<c>0</c>
</answer>


Example 5:
** Program **
```
int a;
int b;
int d;
{WHILE}(d {LT_OP} 5)
{{
  {WHILE}(b {LT_OP} 5)
  {{
    a {ASSIGN_OP} (a {SUB_OP} 1);
    b {ASSIGN_OP} (b {ADD_OP} 1);
  }};
  d {ASSIGN_OP} (d {ADD_OP} 1);
}};
```

The final expected output is:
<answer>
<a>-5</a>
<b>5</b>
<d>5</d>
</answer>


Only write the answer. You **MUST** wrap your prediction with `<answer>` tags.
"""

srp_zero_shot_task_desc = """
## TASK:
For each question below, you'll be given:
1. A program
2. The program state (`𝜎`) (variable values) before executing the program
3. The control stack (`χ`) before executing the program

Assume that all necessary variables have been declared and have the values as
indicated in the provided program state.

You must:
- Correctly identify and apply the small-step operational semantic rules required to evaluate the program to completion
- List them in the correct order of application

A program is executed completely when its evaluation reaches one of the terminal
configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉.


Here is one example:
** Program:**
```
{WHILE} (n {LTEQ_OP} 0)
{{
    {HALT};
}};
```

**Program state(𝜎) before execution:**
{{'n': 100, 'sum': 0}}

**Control stack(χ) before execution:**
 `ε`


This is the sequence of steps:
1. First, we transform the `{WHILE}` into `{LOOP}` using **Rule 67**.
2. Reduce the loop predicate using **Rule 68**.
3. The loop predicate is a {LTEQ_OP} operator which triggers **Rule 32** to first reduce the left-hand side `n` to a literal using **Rule 1**.
4. The right-hand side is already a literal and since `100` is not less-than or equal to `0`. We use **Rule 35** to evaluate this operation to `false`.
5. Since the loop predicate is `false`, we use **Rule 69** to terminate the loop.
6. Since there are no more statements left, we have reached the terminal configuration 〈ε,𝜎,χ〉 and the program evaluation terminates.

Therefore, the final answer is:
<ans>
  <answer id="1">
    <rule>67</rule>
    <rule>68</rule>
    <rule>32</rule>
    <rule>1</rule>
    <rule>35</rule>
    <rule>69</rule>
  </answer>
</ans>


## Questions:
{questions}

## Response Format:
Respond with an XML block structured as follows:

<ans>
  <answer id="1">
    <rule>1</rule>
    <rule>2</rule>
    ...
  </answer>
  <answer id="2">
    <rule>1</rule>
    <rule>2</rule>
    ...
  </answer>
  ...
</ans>

### Notes:
- Each `<answer id="N">` element corresponds to the N-th question.
- Inside each `<answer>` block, list each semantic rule in the correct order using `<rule>` tags.

## Important Notes:
- The **order** of rules matters and should reflect the evaluation sequence.
- A single rule may be needed to be applied multiple times during evaluation.
- You must include **all** semantic rules required for complete execution.
- Base your analysis solely on the provided semantics, not on general programming knowledge.

Only output the `<ans>` XML block. Do not include any other content.
"""

srp_zero_shot_cot_task_desc = """
## TASK:
For each question below, you'll be given:
1. A program
2. The program state (`𝜎`) (variable values) before executing the program
3. The control stack (`χ`) before executing the program

Assume that all necessary variables have been declared and have the values as
indicated in the provided program state.

You must:
- Correctly identify and apply the small-step operational semantic rules required to evaluate the program to completion
- List them in the correct order of application

A program is executed completely when its evaluation reaches one of the terminal
configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉.


Here is one example:
** Program:**
```
{WHILE} (n {LTEQ_OP} 0)
{{
    {HALT};
}};
```

**Program state(𝜎) before execution:**
{{'n': 100, 'sum': 0}}

**Control stack(χ) before execution:**
 `ε`


This is the sequence of steps:
1. First, we transform the `{WHILE}` into `{LOOP}` using **Rule 67**.
2. Reduce the loop predicate using **Rule 68**.
3. The loop predicate is a {LTEQ_OP} operator which triggers **Rule 32** to first reduce the left-hand side `n` to a literal using **Rule 1**.
4. The right-hand side is already a literal and since `100` is not less-than or equal to `0`. We use **Rule 35** to evaluate this operation to `false`.
5. Since the loop predicate is `false`, we use **Rule 69** to terminate the loop.
6. Since there are no more statements left, we have reached the terminal configuration 〈ε,𝜎,χ〉 and the program evaluation terminates.

Therefore, the final answer is:
<ans>
  <answer id="1">
    <rule>67</rule>
    <rule>68</rule>
    <rule>32</rule>
    <rule>1</rule>
    <rule>35</rule>
    <rule>69</rule>
  </answer>
</ans>


## Questions:
{questions}

## Response Format:
Respond with an XML block structured as follows:

<ans>
  <answer id="1">
    <rule>1</rule>
    <rule>2</rule>
    ...
  </answer>
  <answer id="2">
    <rule>1</rule>
    <rule>2</rule>
    ...
  </answer>
  ...
</ans>

### Notes:
- Each `<answer id="N">` element corresponds to the N-th question.
- Inside each `<answer>` block, list each semantic rule in the correct order using `<rule>` tags.

## Important Notes:
- The **order** of rules matters and should reflect the evaluation sequence.
- A single rule may be needed to be applied multiple times during evaluation.
- You must include **all** semantic rules required for complete execution.
- Base your analysis solely on the provided semantics, not on general programming knowledge.

Only output the `<ans>` XML block. Do not include any other content.

Explain your reasoning step-by-step **before** answering. Wrap your reasoning in `<reason>` tags.
"""

srp_zero_shot_k_task_desc = """
## TASK:
For each question below, you'll be given:
1. A program
2. The program state (`𝜎`) (variable values) before executing the program
3. The control stack (`χ`) before executing the program

Assume that all necessary variables have been declared and have the values as
indicated in the provided program state.

You must:
- Correctly identify and apply the K-semantic rules required to evaluate the program to completion
- List them in the correct order of application


Here is one example:
** Program:**
```
{WHILE} (n {LTEQ_OP} 0)
{{
    {HALT};
}};
```

**Program state(𝜎) before execution:**
{{'n': 100, 'sum': 0}}

**Control stack(χ) before execution:**
 `ε`


This is the sequence of steps:
1. First, we transform the `{WHILE}` into `{WHILE}1` while also inserting a `breakMarker` after `{WHILE}1` using **Rule 24**.
2. Next we transform the `{WHILE}1` into an `{IF}-{ELSE}` with the `{WHILE}1` as the body of the `{IF}` using **Rule 25**.
3. We then reduce the loop predicate to a boolean by first reducing left-hand-side which is a variable using **Rule 1** and then applying the `{LTEQ_OP}` using **Rule 13**.
4. Since the loop predicate evaluates to `false`, we apply the `{IF}` not taken rule **Rule 23** to take the `{ELSE}` branch which is empty.
5. Finally, we evaluate the `breakMarker` statement using **Rule 27** to conclude the program execution.

Therefore, the final answer is:
<ans>
  <answer id="1">
    <rule>24</rule>
    <rule>25</rule>
    <rule>1</rule>
    <rule>13</rule>
    <rule>23</rule>
    <rule>27</rule>
  </answer>
</ans>


## Questions:
{questions}

## Response Format:
Respond with an XML block structured as follows:

<ans>
  <answer id="1">
    <rule>1</rule>
    <rule>2</rule>
    ...
  </answer>
  <answer id="2">
    <rule>1</rule>
    <rule>2</rule>
    ...
  </answer>
  ...
</ans>

### Notes:
- Each `<answer id="N">` element corresponds to the N-th question.
- Inside each `<answer>` block, list each semantic rule in the correct order using `<rule>` tags.

## Important Notes:
- The **order** of rules matters and should reflect the evaluation sequence.
- Only rules that have names indicated in `[]` adjacent to it must be reported in the answer.
- A single rule may be needed to be applied multiple times during evaluation.
- You must include **all** semantic rules required for complete execution.
- Base your analysis solely on the provided semantics, not on general programming knowledge.

Only output the `<ans>` XML block. Do not include any other content.
"""

srp_zero_shot_k_task_cot_desc = """
## TASK:
For each question below, you'll be given:
1. A program
2. The program state (`𝜎`) (variable values) before executing the program
3. The control stack (`χ`) before executing the program

Assume that all necessary variables have been declared and have the values as
indicated in the provided program state.

You must:
- Correctly identify and apply the K-semantic rules required to evaluate the program to completion
- List them in the correct order of application


Here is one example:
** Program:**
```
{WHILE} (n {LTEQ_OP} 0)
{{
    {HALT};
}};
```

**Program state(𝜎) before execution:**
{{'n': 100, 'sum': 0}}

**Control stack(χ) before execution:**
 `ε`


This is the sequence of steps:
1. First, we transform the `{WHILE}` into `{WHILE}1` while also inserting a `breakMarker` after `{WHILE}1` using **Rule 24**.
2. Next we transform the `{WHILE}1` into an `{IF}-{ELSE}` with the `{WHILE}1` as the body of the `{IF}` using **Rule 25**.
3. We then reduce the loop predicate to a boolean by first reducing left-hand-side which is a variable using **Rule 1** and then applying the `{LTEQ_OP}` using **Rule 13**.
4. Since the loop predicate evaluates to `false`, we apply the `{IF}` not taken rule **Rule 23** to take the `{ELSE}` branch which is empty.
5. Finally, we evaluate the `breakMarker` statement using **Rule 27** to conclude the program execution.

Therefore, the final answer is:
<ans>
  <answer id="1">
    <rule>24</rule>
    <rule>25</rule>
    <rule>1</rule>
    <rule>13</rule>
    <rule>23</rule>
    <rule>27</rule>
  </answer>
</ans>


## Questions:
{questions}

## Response Format:
Respond with an XML block structured as follows:

<ans>
  <answer id="1">
    <rule>1</rule>
    <rule>2</rule>
    ...
  </answer>
  <answer id="2">
    <rule>1</rule>
    <rule>2</rule>
    ...
  </answer>
  ...
</ans>

### Notes:
- Each `<answer id="N">` element corresponds to the N-th question.
- Inside each `<answer>` block, list each semantic rule in the correct order using `<rule>` tags.

## Important Notes:
- The **order** of rules matters and should reflect the evaluation sequence.
- Only rules that have names indicated in `[]` adjacent to it must be reported in the answer.
- A single rule may be needed to be applied multiple times during evaluation.
- You must include **all** semantic rules required for complete execution.
- Base your analysis solely on the provided semantics, not on general programming knowledge.

Only output the `<ans>` XML block. Do not include any other content.

Explain your reasoning step-by-step **before** answering. Wrap your reasoning in `<reason>` tags.
"""


# etp

etp_task_desc = """## TASK:
Given a program and its semantics, predict the execution trace. Your goal is to simulate execution, step by step of executing the program using the given small-step operational semantics rules. Do not skip any rules that are needed to evaluate the program. You will output your answer in the following format.

## Response Format:
Respond with an XML block structured as follows:

<answer>
  <step>
    <rule>1</rule>
    <program_state>
        <n>0</n>
        <sum>0</sum>
    </program_state>
  </step>
  <step>
    <rule>2</rule>
    <program_state>
      <n>100</n>
      <sum>0</sum>
    </program_state>
  </step>
  ...
</answer>

## Here is an example:

Here is the {language} program:
```
int i;
int j;
i {ASSIGN_OP} 0;
{WHILE} (i {LT_OP} 2)
{{
    {HALT};
}};
```

## Expected output:

<answer>
  <step>
    <rule>3</rule>
    <program_state>
        <i>0</i>
    </program_state>
  </step
  <step>
    <rule>3</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>5</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>67</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>68</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>28</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>1</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>30</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>70</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>78</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
</answer>


## Notes:
- Each `<step>` must correspond to **exactly one small-step operational semantics rule** that is needed to evaluate a statement in the given program.
- The `<rule>` must indicate a rule used in the evaluation of a statement.
- The `<program_state>` must represent the **entire program state immediately after** the execution of that rule.
- The program state must list **all variables currently in scope**, using the variable names as XML tags and their current values as tag content.
- Include variables even if they did not change.
- Do not skip any step or merge multiple steps into one.
- Do not skip any rules (including those used to reduce expressions and variables) that are needed to evaluate the program.
- The program execution is complete when on of the terminal configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉 is reached

Only output the `<answer>` XML block. Do not include explanations, comments, or any other text.
"""

etp_task_cot_desc = """## TASK:
Given a program and its semantics, predict the execution trace. Your goal is to simulate execution, step by step of executing the program using the given small-step operational semantics rules. Do not skip any rules that are needed to evaluate the program. You will output your answer in the following format.

## Response Format:
Respond with an XML block structured as follows:

<answer>
  <step>
    <rule>1</rule>
    <program_state>
        <n>0</n>
        <sum>0</sum>
    </program_state>
  </step>
  <step>
    <rule>2</rule>
    <program_state>
      <n>100</n>
      <sum>0</sum>
    </program_state>
  </step>
  ...
</answer>

## Here is an example:

Here is the {language} program:
```
int i;
int j;
i {ASSIGN_OP} 0;
{WHILE} (i {LT_OP} 2)
{{
    {HALT};
}};
```

## Expected output:

<answer>
  <step>
    <rule>3</rule>
    <program_state>
        <i>0</i>
    </program_state>
  </step
  <step>
    <rule>3</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>5</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>67</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>68</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>28</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>1</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>30</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>70</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>78</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
</answer>


## Notes:
- Each `<step>` must correspond to **exactly one small-step operational semantics rule** that is needed to evaluate a statement in the given program.
- The `<rule>` must indicate a rule used in the evaluation of a statement.
- The `<program_state>` must represent the **entire program state immediately after** the execution of that rule.
- The program state must list **all variables currently in scope**, using the variable names as XML tags and their current values as tag content.
- Include variables even if they did not change.
- Do not skip any step or merge multiple steps into one.
- Do not skip any rules (including those used to reduce expressions and variables) that are needed to evaluate the program.
- The program execution is complete when on of the terminal configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉 is reached

Explain your reasoning step-by-step **before** answering. Wrap your reasoning in `<reason>` tags.
Note that you **MUST** wrap your reasoning steps with `<reason>` tags, the prediction with `<answer>` tags.
"""

etp_k_task_desc = """## TASK:
Given a program and its semantics, predict the execution trace. Your goal is to simulate execution, step by step of executing the program using the given K-semantics rules. Do not skip any rules that are needed to evaluate the program. You will output your answer in the following format.

## Response Format:
Respond with an XML block structured as follows:

<answer>
  <step>
    <rule>1</rule>
    <program_state>
        <n>0</n>
        <sum>0</sum>
    </program_state>
  </step>
  <step>
    <rule>2</rule>
    <program_state>
      <n>100</n>
      <sum>0</sum>
    </program_state>
  </step>
  ...
</answer>

## Here is an example:

Here is the {language} program:
```
int i;
int j;
i {ASSIGN_OP} 0;
{WHILE} (i {LT_OP} 2)
{{
    {HALT};
}};
```

## Expected output:

<answer>
  <step>
    <rule>36</rule>
    <program_state>
        <i>0</i>
    </program_state>
  </step
  <step>
    <rule>36</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>21</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>24</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>25</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>1</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>12</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>22</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>26</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
</answer>


## Notes:
- Each `<step>` must correspond to **exactly one K-semantics re-write rule** that is needed to evaluate a statement in the given program.
- Only rules that have names indicated in `[]` adjacent to it must be reported in the answer.
- The `<rule>` must indicate a rule used in the evaluation of a statement.
- The `<program_state>` must represent the **entire program state immediately after** the execution of that rule.
- The program state must list **all variables currently in scope**, using the variable names as XML tags and their current values as tag content.
- Include variables even if they did not change.
- Do not skip any step or merge multiple steps into one.
- Do not skip any rules (including those used to reduce expressions and variables) that are needed to evaluate the program.

Only output the `<answer>` XML block. Do not include explanations, comments, or any other text.
"""

etp_k_task_cot_desc = """## TASK:
Given a program and its semantics, predict the execution trace. Your goal is to simulate execution, step by step of executing the program using the given K-semantics rules. Do not skip any rules that are needed to evaluate the program. You will output your answer in the following format.

## Response Format:
Respond with an XML block structured as follows:

<answer>
  <step>
    <rule>1</rule>
    <program_state>
        <n>0</n>
        <sum>0</sum>
    </program_state>
  </step>
  <step>
    <rule>2</rule>
    <program_state>
      <n>100</n>
      <sum>0</sum>
    </program_state>
  </step>
  ...
</answer>

## Here is an example:

Here is the {language} program:
```
int i;
int j;
i {ASSIGN_OP} 0;
{WHILE} (i {LT_OP} 2)
{{
    {HALT};
}};
```

## Expected output:

<answer>
  <step>
    <rule>36</rule>
    <program_state>
        <i>0</i>
    </program_state>
  </step
  <step>
    <rule>36</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>21</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>24</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>25</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>1</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>12</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>22</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
  <step>
    <rule>26</rule>
    <program_state>
        <i>0</i>
        <j>0</j>
    </program_state>
  </step>
</answer>


## Notes:
- Each `<step>` must correspond to **exactly one K-semantics re-write rule** that is needed to evaluate a statement in the given program.
- Only rules that have names indicated in `[]` adjacent to it must be reported in the answer.
- The `<rule>` must indicate a rule used in the evaluation of a statement.
- The `<program_state>` must represent the **entire program state immediately after** the execution of that rule.
- The program state must list **all variables currently in scope**, using the variable names as XML tags and their current values as tag content.
- Include variables even if they did not change.
- Do not skip any step or merge multiple steps into one.
- Do not skip any rules (including those used to reduce expressions and variables) that are needed to evaluate the program.

Explain your reasoning step-by-step **before** answering. Wrap your reasoning in `<reason>` tags.
Note that you **MUST** wrap your reasoning steps with `<reason>` tags, the prediction with `<answer>` tags.
"""

#######################
# Prompts for each task
#######################
pep_with_semantics_sos_cot_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax of {language} in EBNF and its semantics using small-step operational semantics. All the rules that lead to the terminal configuration〈{ERROR},𝜎,χ〉, are describing semantically invalid statements. If the execution of a {language} program requires such rules then that program is not executable. Your task is to check if the given {language} program can execute or not. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the syntax of {language} in EBNF
    ```
    {syntax}
    ```

    Here is the small-step operational semantics of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + pep_task_cot_desc
)


pep_with_semantics_sos_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax of {language} in EBNF and its semantics using small-step operational semantics. All the rules that lead to the terminal configuration〈{ERROR},𝜎,χ〉, are describing semantically invalid statements. If the execution of a {language} program requires such rules then that program is not executable. Your task is to check if the given {language} program can execute or not. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the syntax of {language} in EBNF
    ```
    {syntax}
    ```

    Here is the small-step operational semantics of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + pep_task_desc
)


pep_with_semantics_k_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax and the semantics of {language} using the K-framework. All the re-write rules that re-write operations/statements to {ERROR}, are describing semantically invalid statements. If the execution of a {language} program requires such rules then that program is not executable. Your task is to check if the given {language} program can execute or not. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the K-framework formalization of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + pep_k_task_desc
)


pep_with_semantics_k_cot_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax and the semantics of {language} using the K-framework. All the re-write rules that re-write operations/statements to {ERROR}, are describing semantically invalid statements. If the execution of a {language} program requires such rules then that program is not executable. Your task is to check if the given {language} program can execute or not. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the K-framework formalization of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + pep_k_task_cot_desc
)


op_with_semantics_sos_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax for {language} in EBNF and its semantics using small-step operational semantics. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct. A program has finished execution when one of the terminal configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉 is reached.
    Here is the syntax of {language} in EBNF
    ```
    {syntax}
    ```

    Here is the small-step operational semantics of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + new_op_task_desc
)

op_with_semantics_five_shot_sos_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax for {language} in EBNF and its semantics using small-step operational semantics. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct. A program has finished execution when one of the terminal configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉 is reached.
    Here is the syntax of {language} in EBNF
    ```
    {syntax}
    ```

    Here is the small-step operational semantics of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + new_op_task_five_shot_desc
)


op_with_semantics_sos_cot_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax for {language} in EBNF and its semantics using small-step operational semantics. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct. A program has finished execution when one of the terminal configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉 is reached.
    Here is the syntax of {language} in EBNF
    ```
    {syntax}
    ```

    Here is the semantics of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + new_op_task_cot_desc
)

op_with_semantics_k_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax and the semantics of the language using the K-framework. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the K-framework formalization of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + new_op_task_desc
)

op_with_semantics_five_shot_k_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax and the semantics of the language using the K-framework. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the K-framework formalization of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + new_op_task_five_shot_desc
)

op_with_semantics_k_cot_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax and the semantics of the language using the K-framework. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the K-framework formalization of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + new_op_task_cot_desc
)


op_no_semantics_prompt = (
    textwrap.dedent("""\
    You are an interpreter for my language called {language}.

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + new_op_task_desc
)

op_no_semantics_cot_prompt = (
    textwrap.dedent("""\
    You are an interpreter for my language called {language}.

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + new_op_task_cot_desc
)

## --
# srp
## --

srp_zero_shot_sos_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax for {language} in EBNF and its semantics using small-step operational semantics. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct. A program has finished execution when one of the terminal configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉 is reached.
    Here is the syntax of {language} in EBNF
    ```
    {syntax}
    ```

    Here is the small-step operational semantics of {language}
    ```
    {semantics}
    ```
    """)
    + srp_zero_shot_task_desc
)

srp_zero_shot_sos_cot_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax for {language} in EBNF and its semantics using small-step operational semantics. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct. A program has finished execution when one of the terminal configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉 is reached.
    Here is the syntax of {language} in EBNF
    ```
    {syntax}
    ```

    Here is the small-step operational semantics of {language}
    ```
    {semantics}
    ```
    """)
    + srp_zero_shot_cot_task_desc
)

srp_zero_shot_k_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax and the semantics of the language using the K-framework. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the K-framework formalization of {language}
    ```
    {semantics}
    ```
    """)
    + srp_zero_shot_k_task_desc
)

srp_zero_shot_k_cot_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax and the semantics of the language using the K-framework. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the K-framework formalization of {language}
    ```
    {semantics}
    ```
    """)
    + srp_zero_shot_k_task_cot_desc
)


srp_stmt_prompt = textwrap.dedent("""\
    ### Question {index}:
    ** Program:**
    ```
    {statement}
    ```

**Program state(𝜎) before execution:**
{program_state}

**Control stack(χ) before execution:**
{control_stack}
    """)


# ETP prompts
etp_imp_sos_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax for {language} in EBNF and its semantics using small-step operational semantics. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct. A program has finished execution when one of the terminal configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉 is reached.
    Here is the syntax of {language} in EBNF
    ```
    {syntax}
    ```

    Here is the small-step operational semantics of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + etp_task_desc
)

etp_imp_sos_cot_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax for {language} in EBNF and its semantics using small-step operational semantics. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct. A program has finished execution when one of the terminal configurations 〈ε,𝜎,χ〉,〈{HALT},𝜎,χ〉,〈{ERROR},𝜎,χ〉 is reached.
    Here is the syntax of {language} in EBNF
    ```
    {syntax}
    ```

    Here is the small-step operational semantics of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + etp_task_cot_desc
)

etp_imp_k_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax and the semantics of the language using the K-framework. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the K-framework formalization of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + etp_k_task_desc
)

etp_imp_k_cot_prompt = (
    textwrap.dedent("""\
    You are an interpreter for a language called {language}. I will describe the syntax and the semantics of the language using the K-framework. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct.
    Here is the K-framework formalization of {language}
    ```
    {semantics}
    ```

    Here is the {language} program
    ```
    {program}
    ```
    """)
    + etp_k_task_cot_desc
)

# Translate to Python Code

python_code_translation_prompt = (
    """You are an interpreter for my language called {language}. I will describe the syntax for {language} in EBNF form and I will describe the semantics for {language} using small-step operational semantics. You will use this to execute a {language} program. You will only use the rules described in the semantics I provide. Assume all the rules in the semantics I give are correct. Here is the syntax of {language} in EBNF form
```
{syntax}
```

Here is the semantics of {language}
```
{semantics}
```

Here is the {language} program
```
{program}
```
"""
    + """
TASK: Translate the following {language} program into an executable Python program with the following requirements:
- The Python program must be semantically equivalent to the original {language} program, producing the same outputs and behavior.
- Insert a print statement so that the final value of the variable `ans` is output to stdout at the moment the program finishes execution. This print statement should serve as a way to observe the value of `ans` after running the Python program.
- Ensure that `print(ans)` is executed whenever the program terminates, including when the program exits early. Place the print statement immediately before any early termination as needed.

Notes:
- Wrap the Python code in ``` tags.
- Use `print(ans)` for outputting the value.

"""
)


# Generate interpreter in code

ig_imp_sos_prompt = (
    """You are to implement an interpreter for a new programming language called {language}. The syntax of {language} is defined using **{syntax_type}** notation:
```
{syntax}
```

The semantics of {language} are defined using **structural operational semantics**:
```
{semantics}
```
"""
    + """## TASK:

Write a complete interpreter for {language} that meets the following specifications:
- **Implementation Language**: Write the interpreter in a single `{impl_language}` file.
- **Input**: The interpreter takes **one command-line argument**, which is the path to the input program file.
- **Dependency**: Use only standard libraries—do **not** rely on any third-party packages (including the ANTLR library, instead implement a parser based on the syntax rules from scratch).
- **Error handling**:
  - If the input program is **syntactically invalid**, print an appropriate error message to stdout and exit with non-zero status.
    - Message format: `Syntax Error: [error message]`
  - If the program is **semantically invalid**, print an appropriate error message to stdout and exit with non-zero status.
    - Message format: `Semantic Error: [error message]`
- **Output**: If the program is valid, interpret it according to the given semantics and print the **final value** of all declared variables in **JSON** format.

Notes:
- Your response must **only** contain the interpreter code in `{impl_language}`.
- Do **not** include explanations, comments, or any other output.
{additional_notes}

## EXAMPLES:

### Example 1 (Valid Program)
**Input:**
```
int a;
int b;
int ans;
a = 3045;
b = 1078;
ans = (a + b);
```

**Expected Output:**
```
{{
  "a": 3045,
  "ans": 4123,
  "b": 1078
}}
```

### Example 2 (Syntax Error)
**Input:**
```
long a;
int b;
int ans;
a = 3045;
b = 1078;
ans = (a + b);
```

**Expected Output:**
```
Syntax Error: 'long' is not a valid keyword.
```

### Example 3 (Semantic Error)
**Input:**
```
int a;
int b;
int ans;
ans = 0;
a = (3045 %% ans);
b = 1078;
ans = (a + b);
```

**Expected Output:**
```
Semantic Error: Modulo by zero.
```
"""
)


## Generate interpreter using ANTLR visitor

iga_prompt = """
## Role and Goal
You are an expert in Programming Languages and their implementation. Your goal is to implement an interpreter for a programming language called {language} using ANTLR. The interpreter must correctly evaluate programs according to the language's specified syntax and semantics.


## Language Specification

### Syntax
The syntax of {language} is defined by the following ANTLR grammar:
```
{syntax}
```

### Semantics
The semantics of {language} are defined by the following **structural operational semantics**:
```
{semantics}
```


## Implementation Task
Your task is to write a complete and correct interpreter for {language} by extending the ANTLR-generated base visitor class.

### Base Visitor
ANTLR will generate a lexer, parser and a base visitor. You do not need to handle the parsing part yourself. To implement the interpreter, you **only** need to extend the base visitor. The base visitor provides methods for each grammar rule, which you will override to implement the semantics. The base visitor looks like this:
```{impl_language}
{base_visitor_code}
```

ANLTR also generates the parser file, you can ignore most parts of it as it is
automatically generated, but you may find some useful APIs to access the parse
tree. The parser file looks like this:
```{impl_language}
{parser_code}
```

### Your Visitor Class
You must implement the interpreter in a single response with code written in `{impl_language}`.
The file should contain a class called `{extend_visitor_name}` that extends the base visitor class. The starter code looks like this:
```{impl_language}
{extend_visitor_code}
```

## Execution and Output

### Program Execution
Your code will be invoked by a main driver, which will parse the input source code and pass the resulting parse tree to your visitor implementation.
The main driver code looks like this:
```{impl_language}
{main_code}
```

### Expected Behavior
- **Sucessful Execution**: If the program is semantically valid and runs to completion, the driver code will serialize the final contents of the `{states_field}` dictionary to a JSON object and print it to standard output. Your code must correctly maintain the `{states_field}` field correctly according to the semantics.
- **Semantic Errors**: If the program is semantically invalid (i.e., there is no corresponding semantic rule for a specific state), your interpreter must:
  1. Print an error message to standard output in the format: `Semantic Error: [error message]`
  2. Exit the program immediately with a non-zero status code.


## Notes:
- Do not use any other third-party libraries except ANTLR, you can use standard libraries provided by {impl_language}.
- Your response must **only** contain the full code list of the extended visitor class `{extend_visitor_name}`.
{additional_notes}

## EXAMPLES:
Here are two examples demonstrating the expected behavior for valid and invalid programs.

### Example 1 (Valid Program)
**Input File Content:**
```
{valid_example_0}
```

**Expected Output:**
```
{valid_output_0}
```

### Example 2 (Semantic Error)
**Input File Content:**
```
{invalid_example_0}
```

**Expected Output (Exit with non-zero status code):**
```
{invalid_output_0}
```


## Final Request
Please provide only the complete {impl_language} code for the {extend_visitor_name} class.
Do not include any additional surrounding text in your response.
"""


igaf_timeout_prompt = """
## Execution for {pgm_type} example {idx} timed out:

- Example content:
```
{code}
```

- Actual stdout:
```
{stdout}
```

- Actual stderr:
```
{stderr}
```

- Expected stdout:
```
{expected_output}
```
"""


igaf_build_fail_prompt = """
## Build failed for the generated interpreter:

- Stdout:
```
{stdout}
```

- Stderr:
```
{stderr}
```
"""


igaf_valid_wr_prompt = """
## Execution for valid example {idx} exits unexpectedly:

- Example content:
```
{code}
```

- Actual stdout:
```
{stdout}
```

- Actual stderr:
```
{stderr}
```

- Expected stdout:
```
{expected_output}
```
"""


igaf_valid_wa_prompt = """
## Execution for valid example {idx} produces wrong output:

- Example content:
```
{code}
```

- Actual stdout:
```
{stdout}
```

- Expected stdout:
```
{expected_output}
```
"""


igaf_invalid_wr_prompt = """
## Execution for valid example {idx} should exit with an error but did not:

- Example content:
```
{code}
```

- Actual stdout:
```
{stdout}
```

- Expected stdout:
```
{expected_output}
```
"""


igaf_invalid_wa_prompt = """
## Execution for invalid example {idx} does not produce the expected error message:

- Example content:
```
{code}
```

- Actual stdout:
```
{stdout}
```

- Actual stderr:
```
{stderr}
```

- Expected stdout:
```
{expected_output}
```
"""


igaf_prefix = """
The generated interpreter does not behave as expected. Below is some feedback:
"""


igaf_suffix = """
Please fix the issues and regenerate the interpreter code.
**Only provide the improved code in the response, do not include any additional text around it.**
"""


## Generate an AST for given program from grammar and semantics.
ast_prompt = """
##
You are an expert in Programming Languages and their implementation. Your goal is to generate the AST for statements in my language called {language}. I will give you the EBNF grammar and small-step operational semantics of {language}. You will use only the grammar rules in the EBNF grammar I provide to generate the AST.


## Language Specification

### Syntax
The syntax of {language} is given in EBNF notation:
```
{syntax}
```

### Semantics
The semantics of {language} are defined by the following **small-step operational semantics**:
```
{semantics}
```

### Statement
The statement in a program of {language}.
```
{statement}
```

## Implementation Task
Your task is to generate the AST for the statement I provided using only the grammar rules in the EBNF syntax I provided.

## Final Request
Please wrap your answer in the `<ans>` tags like <ans> YOUR ANSWER </ans>
"""

# RULE_DESCRIPTIONS = {
#   "RULE_1_DESCRIPTION": """If a variable `x` is bound to the value `v` in the current program state `σ`, then evaluating the expression `x` produces the value `v`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_2_DESCRIPTION": """If a variable `x` is not bound (i.e., σ(x) = ⟂) in the current program state `σ`, then evaluating the expression `x` produces a configuration 〈{ERROR},𝜎,χ〉. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_3_DESCRIPTION": """If a statement `int x` appears at the head of the statement list, then evaluating it initializes `x` to `0` and removes the statement from the list. This rule updates the program state `σ` but does not modify the control stack `χ`.""",
#   "RULE_4_DESCRIPTION": """The right-hand expression `a` in `x {ASSIGN_OP} a` is an arithmetic subexpression that can be reduced to `a'`, so the evaluation of `x {ASSIGN_OP} a` starts by reducing `a`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_5_DESCRIPTION": """If the variable `x` is bound to a value in the current program state `σ`, then evaluating the statement `x {ASSIGN_OP} v` writes the value `v` to `x` in the program state `σ` and removes this statement from the statement list. This rule updates the program state `σ` but does not modify the control stack `χ`.""",
#   "RULE_6_DESCRIPTION": """If the variable `x` is not bound (i.e., σ(x) = ⟂) in the current program state `σ`, then evaluating the statement `x {ASSIGN_OP} v` produces a configuration 〈{ERROR},𝜎,χ〉. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Arithmetic operations
#   # Addition
#   "RULE_7_DESCRIPTION": """The left-hand operand `a1` in `a1 {ADD_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {ADD_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_8_DESCRIPTION": """The right-hand operand `a2` in `v1 {ADD_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {ADD_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_9_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {ADD_OP} v2` are integer values, so the expression `v1 {ADD_OP} v2` evaluates to the value of `v1 + v2` (addition of v1 and v2). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Subtraction
#   "RULE_10_DESCRIPTION": """The left-hand operand `a1` in `a1 {MINUS_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {MINUS_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_11_DESCRIPTION": """The right-hand operand `a2` in `v1 {MINUS_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {MINUS_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_12_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {MINUS_OP} v2` are integer values, so the expression `v1 {MINUS_OP} v2` evaluates to the value of `v1 - v2` (subtraction of v1 and v2). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Multiplication
#   "RULE_13_DESCRIPTION": """The left-hand operand `a1` in `a1 {MUL_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {MUL_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_14_DESCRIPTION": """The right-hand operand `a2` in `v1 {MUL_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {MUL_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_15_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {MUL_OP} v2` are integer values, so the expression `v1 {MUL_OP} v2` evaluates to the value of `v1 * v2` (multiplication of v1 and v2). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Division
#   "RULE_16_DESCRIPTION": """The left-hand operand `a1` in `a1 {DIV_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {DIV_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_17_DESCRIPTION": """The right-hand operand `a2` in `v1 {DIV_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {DIV_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_18_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {DIV_OP} v2` are integer values and the second operand `v2` is not zero, so the expression `v1 {DIV_OP} v2` evaluates to the value of `v1 / v2` (integer division of v1 by v2). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_19_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {DIV_OP} v2` are integer values and the second operand `v2` is zero, so the expression `v1 {DIV_OP} v2` produces a configuration 〈{ERROR},𝜎,χ〉. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Modulus
#   "RULE_20_DESCRIPTION": """The left-hand operand `a1` in `a1 {MOD_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {MOD_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_21_DESCRIPTION": """The right-hand operand `a2` in `v1 {MOD_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {MOD_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_22_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {MOD_OP} v2` are integer values and the second operand `v2` is not zero, so the expression `v1 {MOD_OP} v2` evaluates to the value of `v1 % v2` (modulus of v1 by v2). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_23_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {MOD_OP} v2` are integer values and the second operand `v2` is zero, so the expression `v1 {MOD_OP} v2` produces a configuration 〈{ERROR},𝜎,χ〉. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Unary minus
#   "RULE_24_DESCRIPTION": """The operand `a` in `{MINUS_OP} a` is an arithmetic subexpression that can be reduced to `a'`, so the evaluation of `{MINUS_OP} a` starts by reducing `a`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_25_DESCRIPTION": """The operand `v1` in `{MINUS_OP} v1` is an integer value, so the expression `{MINUS_OP} v1` evaluates to the value of `-v1` (negation of v1). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Unary plus
#   "RULE_26_DESCRIPTION": """The operand `a` in `{PLUS_OP} a` is an arithmetic subexpression that can be reduced to `a'`, so the evaluation of `{PLUS_OP} a` starts by reducing `a`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_27_DESCRIPTION": """The operand `v1` in `{PLUS_OP} v1` is an integer value, so the expression `{PLUS_OP} v1` evaluates to the value of `v1` (identity of v1). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Less-than
#   "RULE_28_DESCRIPTION": """The left-hand operand `a1` in `a1 {LT_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {LT_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_29_DESCRIPTION": """The right-hand operand `a2` in `v1 {LT_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {LT_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_30_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {LT_OP} v2` are integer values and `v1 < v2` (v1 is less than v2), so the expression `v1 {LT_OP} v2` evaluates to `true`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_31_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {LT_OP} v2` are integer values and `v1 ≥ v2` (v1 is greater than or equal to v2), so the expression `v1 {LT_OP} v2` evaluates to `false`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Less-than-equal
#   "RULE_32_DESCRIPTION": """The left-hand operand `a1` in `a1 {LTEQ_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {LTEQ_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_33_DESCRIPTION": """The right-hand operand `a2` in `v1 {LTEQ_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {LTEQ_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_34_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {LTEQ_OP} v2` are integer values and `v1 ≤ v2` (v1 is less than or equal to v2), so the expression `v1 {LTEQ_OP} v2` evaluates to `true`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_35_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {LTEQ_OP} v2` are integer values and `v1 > v2` (v1 is greater than v2), so the expression `v1 {LTEQ_OP} v2` evaluates to `false`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Greater-than
#   "RULE_36_DESCRIPTION": """The left-hand operand `a1` in `a1 {GT_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {GT_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_37_DESCRIPTION": """The right-hand operand `a2` in `v1 {GT_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {GT_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_38_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {GT_OP} v2` are integer values and `v1 > v2` (v1 is greater than v2), so the expression `v1 {GT_OP} v2` evaluates to `true`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_39_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {GT_OP} v2` are integer values and `v1 ≤ v2` (v1 is less than or equal to v2), so the expression `v1 {GT_OP} v2` evaluates to `false`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Greater-than-equal
#   "RULE_40_DESCRIPTION": """The left-hand operand `a1` in `a1 {GTEQ_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {GTEQ_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_41_DESCRIPTION": """The right-hand operand `a2` in `v1 {GTEQ_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {GTEQ_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_42_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {GTEQ_OP} v2` are integer values and `v1 ≥ v2` (v1 is greater than or equal to v2), so the expression `v1 {GTEQ_OP} v2` evaluates to `true`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_43_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {GTEQ_OP} v2` are integer values and `v1 < v2` (v1 is less than v2), so the expression `v1 {GTEQ_OP} v2` evaluates to `false`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Equality
#   "RULE_44_DESCRIPTION": """The left-hand operand `a1` in `a1 {EQ_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {EQ_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_45_DESCRIPTION": """The right-hand operand `a2` in `v1 {EQ_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {EQ_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_46_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {EQ_OP} v2` are integer values and `v1 = v2` (v1 is equal to v2), so the expression `v1 {EQ_OP} v2` evaluates to `true`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_47_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {EQ_OP} v2` are integer values and `v1 ≠ v2` (v1 is not equal to v2), so the expression `v1 {EQ_OP} v2` evaluates to `false`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Inequality
#   "RULE_48_DESCRIPTION": """The left-hand operand `a1` in `a1 {NEQ_OP} a2` is an arithmetic subexpression that can be reduced to `a1'`, so the evaluation of `a1 {NEQ_OP} a2` starts by reducing `a1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_49_DESCRIPTION": """The right-hand operand `a2` in `v1 {NEQ_OP} a2` is an arithmetic subexpression that can be reduced to `a2'`, so the evaluation of `v1 {NEQ_OP} a2` starts by reducing `a2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_50_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {NEQ_OP} v2` are integer values and `v1 ≠ v2` (v1 is not equal to v2), so the expression `v1 {NEQ_OP} v2` evaluates to `true`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_51_DESCRIPTION": """Both operands `v1` and `v2` in `v1 {NEQ_OP} v2` are integer values and `v1 = v2` (v1 is equal to v2), so the expression `v1 {NEQ_OP} v2` evaluates to `false`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Boolean operations
#   # Boolean and
#   "RULE_52_DESCRIPTION": """The left-hand operand `b1` in `b1 {AND_OP} b2` is a boolean subexpression that can be reduced to `b1'`, so the evaluation of `b1 {AND_OP} b2` starts by reducing `b1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_53_DESCRIPTION": """The right-hand operand `b2` in `q1 {AND_OP} b2` is a boolean subexpression that can be reduced to `b2'`, so the evaluation of `q1 {AND_OP} b2` starts by reducing `b2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_54_DESCRIPTION": """Both operands `q1` and `q2` in `q1 {AND_OP} q2` are boolean values and `q1 = true` and `q2 = true`, so the expression `q1 {AND_OP} q2` evaluates to `true` (true ⋀ true is true). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_55_DESCRIPTION": """Both operands `q1` and `q2` in `q1 {AND_OP} q2` are boolean values and either `q1` or `q2` is false, so the expression `q1 {AND_OP} q2` evaluates to `false` (true ⋀ false is false or false ⋀ true is false). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Boolean or
#   "RULE_56_DESCRIPTION": """The left-hand operand `b1` in `b1 {OR_OP} b2` is a boolean subexpression that can be reduced to `b1'`, so the evaluation of `b1 {OR_OP} b2` starts by reducing `b1`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_57_DESCRIPTION": """The right-hand operand `b2` in `q1 {OR_OP} b2` is a boolean subexpression that can be reduced to `b2'`, so the evaluation of `q1 {OR_OP} b2` starts by reducing `b2`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_58_DESCRIPTION": """Both operands `q1` and `q2` in `q1 {OR_OP} q2` are boolean values and either `q1` or `q2` is true, so the expression `q1 {OR_OP} q2` evaluates to `true` (true ⋁ true is true or false ⋁ true is true). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_59_DESCRIPTION": """Both operands `q1` and `q2` in `q1 {OR_OP} q2` are boolean values and both `q1` and `q2` are false, so the expression `q1 {OR_OP} q2` evaluates to `false` (false ⋁ false is false). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Boolean not
#   "RULE_60_DESCRIPTION": """The operand `b` in `{NOT_OP} b` is a boolean subexpression that can be reduced to `b'`, so the evaluation of `{NOT_OP} b` starts by reducing `b`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_61_DESCRIPTION": """The operand `q` in `{NOT_OP} q` is a boolean value and `q = false`, so the expression `{NOT_OP} q` evaluates to `true` (logical negation of `q`). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_62_DESCRIPTION": """The operand `q` in `{NOT_OP} q` is a boolean value and `q = true`, so the expression `{NOT_OP} q` evaluates to `false` (logical negation of `q`). This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # Sequence
#   "RULE_63_DESCRIPTION": """The head statement `s` in `s :: SL` is a statement that can be reduced to `s'`, so the evaluation of `s :: SL` starts by reducing `s`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # If-else
#   "RULE_64_DESCRIPTION": """The predicate `b` in the `{IF}(b) {{SL1}} {ELSE} {{SL2}}` statement is a boolean subexpression that can be reduced to `b'`, so the evaluation of `{IF}(b) {{SL1}} {ELSE} {{SL2}}` starts by reducing `b`. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_65_DESCRIPTION": """The predicate `q` in the `{IF}(q) {{SL1}} {ELSE} {{SL2}}` statement is a boolean value and `q = true`, therefore statements in statement list `SL1` ({IF}-branch) are prepended to the statement list `SL` containing the remaining statements in the program and `{IF}(q) {{SL1}} {ELSE} {{SL2}}` is removed from the head of this list. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   "RULE_66_DESCRIPTION": """The predicate `q` in the `{IF}(q) {{SL1}} {ELSE} {{SL2}}` statement is a boolean value and `q = false`, therefore statements in statement list `SL2` ({ELSE}-branch) are prepended to the statement list `SL` containing the remaining statements in the program and `{IF}(q) {{SL1}} {ELSE} {{SL2}}` is removed from the head of this list. This rule does not modify the program state `σ` or the control stack `χ`.""",
#   # While-loop 
# }


# Formal semantics notation comprehension prompt NL ---> Rule
formal_semantics_notation_comprehension_prompt_sos_nl_2_rule = textwrap.dedent(
  """I have defined the syntax and the semantics of a programming language called {language} formally using EBNF and small-step operational semantics.
    
    Here is the EBNF syntax of {language}:
    ```
{syntax}
    ```

TASK:
You will be given a set of {num_rules} rules from the small-step operational semantics of {language}
along with a natural language description of exactly **one** of those rules.
Your task is to identify which rule correctly captures the operational behavior of the given natural language description.

    Below is the glossary explaining the symbols and metavariables used in the small-step operational semantics rules:
    ```
    {semantics_glossary}
    ```

    Here is the set of {num_rules} small-step operational semantics rules:
    ```
{rules}
    ```

    Here is the natural language description of **one** of the given small-step operational semantics rules:
    ```
    {description}
    ```

# Note 1: Only one of the rules will match the natural language description.
# Note 2: You must only use the rules provided and the natural language description to identify the rule.
# Note 3: Assume that all the rules and the natural language description given are correct.
# Note 4: You should not select a rule solely based on surface syntax or keyword matching; the choice must be based on the operational semantics behavior described in the natural language description.
# Note 5: You must wrap your final answer with `<answer>` tags.
    
For example if `Rule X` matches the natural language description, then your response should be:
```
<answer>Rule X</answer>
```""")

formal_semantics_notation_comprehension_prompt_k_nl_2_rule = textwrap.dedent(
  """I have defined the syntax and the semantics of a programming language called {language} formally using the K-framework.
    
    Here is the K-framework syntax of {language}:
    ```
{syntax}
    ```

TASK:
You will be given a set of {num_rules} rules from the K-framework semantics of {language}
along with a natural language description of exactly **one** of those rules.
Your task is to identify which rule correctly captures the operational behavior of the given natural language description.

    Below is the glossary explaining the configuration of the K-cell and auxiliary rules and functions used in the K-framework semantics rules:
    ```
    {semantics_glossary}
    ```

    Here is the set of {num_rules} K-framework semantics rules:
    ```
{rules}
    ```

    Here is the natural language description of **one** of the given K-framework semantics rules:
    ```
    {description}
    ```

# Note 1: Only one of the rules will match the natural language description.
# Note 2: You must only use the rules provided and the natural language description to identify the rule.
# Note 3: Assume that all the rules and the natural language description given are correct.
# Note 4: You should not select a rule solely based on surface syntax or keyword matching; the choice must be based on the K-framework semantics behavior described in the natural language description.
# Note 5: You must wrap your final answer with `<answer>` tags.
    
For example if `Rule X` matches the natural language description, then your response should be:
```
<answer>Rule X</answer>
```""")

# Formal semantics notation comprehension prompt Rule ---> NL
formal_semantics_notation_comprehension_prompt_sos_rule_2_nl = textwrap.dedent(
  """I have defined the syntax and the semantics of a programming language called {language} formally using EBNF and small-step operational semantics.
    
    Here is the EBNF syntax of {language}:
    ```
{syntax}
    ```

TASK:
You will be given a set of {num_descriptions} natural language descriptions of unique rules from the small-step 
operational semantics of {language} along with exactly **one** small-step operational semantics rule.
Your task is to identify which natural language description correctly captures the operational behavior of the given small-step operational semantics rule.

    Below is the glossary explaining the symbols and metavariables used in the small-step operational semantics rules:
    ```
    {semantics_glossary}
    ```

    Here is the set of {num_descriptions} natural language descriptions of unique small-step operational semantics rules:
    ```
{descriptions}
    ```

    Here is the small-step operational semantics rule:
    ```
    {rule}
    ```

# Note 1: Only one of the natural language descriptions will match the given rule.
# Note 2: You must only use the natural language descriptions provided and the given rule to identify the natural language description.
# Note 3: Assume that all the natural language descriptions and the given rule are correct.
# Note 4: You should not select a natural language description solely based on surface syntax or keyword matching; the choice must be based on the operational semantics behavior described in the given rule.
# Note 5: You must wrap your final answer with `<answer>` tags.
    
For example if `Description X` matches the given small-step operational semantics rule, then your response should be:
```
<answer>Description X</answer>
```""")

formal_semantics_notation_comprehension_prompt_k_rule_2_nl = textwrap.dedent(
  """I have defined the syntax and the semantics of a programming language called {language} formally using the K-framework.
    
    Here is the K-framework syntax of {language}:
    ```
{syntax}
    ```

TASK:
You will be given a set of {num_descriptions} natural language descriptions of unique rules from the K-framework 
semantics of {language} along with exactly **one** K-framework semantics rule.
Your task is to identify which natural language description correctly captures the operational behavior of the given K-framework semantics rule.

    Below is the glossary explaining the configuration of the K-cell and auxiliary rules and functions used in the K-framework semantics rules:
    ```
    {semantics_glossary}
    ```

    Here is the set of {num_descriptions} natural language descriptions of unique K-framework semantics rules:
    ```
{descriptions}
    ```

    Here is the K-framework semantics rule:
    ```
    {rule}
    ```

# Note 1: Only one of the natural language descriptions will match the given rule.
# Note 2: You must only use the natural language descriptions provided and the given rule to identify the natural language description.
# Note 3: Assume that all the natural language descriptions and the given rule are correct.
# Note 4: You should not select a natural language description solely based on surface syntax or keyword matching; the choice must be based on the K-framework semantics behavior described in the given rule.
# Note 5: You must wrap your final answer with `<answer>` tags.
    
For example if `Description X` matches the given K-framework semantics rule, then your response should be:
```
<answer>Description X</answer>
```""")

