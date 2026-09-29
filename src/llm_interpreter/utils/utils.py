from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence, List
from datasets import DatasetDict, load_dataset, Dataset, Value
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError
import re
import tiktoken
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


def num_tokens_from_string(string: str, model_name: str = "gpt-4o") -> int:
    """
    Returns the number of tokens in a text string for a specific model. Default: gpt-4o.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Model not found, using cl100k_base encoding as fallback (used by GPT-4, GPT-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def detect_vector_list_switch(cpp_code: str) -> bool:
    """
    Detects if the given C++ code uses std::vector, std::list, std::string, or switch.
    (Updated by Gemini)

    Args:
        cpp_code: A string containing the C++ code.

    Returns:
        bool: True if the code uses std::vector, std::list, std::string, or switch, False otherwise.
    """
    uses_vector = False
    uses_list = False
    uses_string = False
    uses_switch = False

    # Remove single-line comments
    code_without_single_line_comments = re.sub(r"//.*", "", cpp_code)

    # Remove multi-line comments
    code_without_comments = re.sub(
        r"/\*[\s\S]*?\*/", "", code_without_single_line_comments
    )

    # Convert to lowercase for case-insensitive search
    lower_code = code_without_comments.lower()

    # Check for include directives
    if "#include <vector>" in lower_code:
        uses_vector = True
    if "#include <list>" in lower_code:
        uses_list = True
    if "#include <string>" in lower_code:
        uses_string = True

    # Check for declarations and usage (more robust)
    if re.search(r"\bstd::vector\s*<", lower_code):
        uses_vector = True
    elif re.search(r"\bvector\s*<", lower_code):
        if "#include <vector>" in lower_code or "using namespace std;" in lower_code:
            uses_vector = True

    if re.search(r"\bstd::list\s*<", lower_code):
        uses_list = True
    elif re.search(r"\blist\s*<", lower_code):
        if "#include <list>" in lower_code or "using namespace std;" in lower_code:
            uses_list = True

    if re.search(r"\bstd::string\b", lower_code):
        uses_string = True
    elif re.search(r"\bstring\b", lower_code):
        if "#include <string>" in lower_code or "using namespace std;" in lower_code:
            uses_string = True

    # Check for the 'switch' keyword
    if re.search(r"\bswitch\b", lower_code):
        uses_switch = True

    return uses_string or uses_vector or uses_list or uses_switch


def extract_content_between_tags(text: str, tag: str) -> str:
    """
    Extract the content between the last occurrence of the specified tag in the text.
    """
    # first preprocess
    last_opening = text.rfind(f"<{tag}>")
    if last_opening == -1:
        return ""
    text = text[last_opening:]
    # Find all matches of content between <ans> and </ans> tags
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the last match if found, otherwise return empty string
    if matches:
        return matches[-1]
    else:
        return ""


def extract_boxed_answers(latex_text: str):
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.findall(pattern, latex_text)
    if matches:
        return matches[-1]
    else:
        return ""


def extract_all_content_between_tags(text: str, tag: str) -> List[str]:
    """
    Extract all content between the specified tags in the text.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def parse_srp_prediction(xml_string: str) -> List:
    """
    Parse the model-generated XML and return a list of rules.
    """
    result = []
    try:
        root = ET.fromstring(xml_string)
        for answer in root.findall("answer"):
            question_id = int(answer.attrib.get("id"))
            rules = [rule.text for rule in answer.findall("rule")]
            if rules == [None]:
                rules = []
            result.append({"question_id": question_id, "rules": rules})
    except Exception as e:
        print(f"Failed to parse XML: {e}")
    pred_rules = []
    for i, pred in enumerate(result):
        if pred["question_id"] == i + 1:
            pred_rules.append(pred["rules"])
        else:
            pred_rules.append([])
    return pred_rules


def parse_op_prediction(xml_string: str):
    """
    Parse the model-generated XML and return a list of variable values in lexicographical order.
    """
    result = []
    variables: dict = {}
    try:
        root = ET.fromstring(xml_string)
        variables = {var.tag: int(var.text.strip()) for var in root}
        # sort it
        sorted_keys = sorted(variables)
        for key in sorted_keys:
            result.append(variables[key])
        # rof
    except Exception as e:
        print(f"Failed to parse XML: {e}")
    return (result, variables)


def parse_etp_prediction(xml_string: str) -> List[dict]:
    """
    Parse the model-generated XML and return a list of dictionaries containing
    linenumber and program_state.
    """
    # Extract steps
    result = []
    xml_string = "\n".join(
        [
            line
            for line in xml_string.splitlines()
            if "⟂" not in line and "--" not in line and "halt" not in line
        ]
    )
    try:
        root = ET.fromstring(xml_string)

        for step in root.findall("step"):
            rule = step.find("rule").text
            m = re.search(r"(?i)\brule\b\s*([0-9]+)\b", rule)
            if m:
                rule = int(m.group(1))
            else:
                m = re.search(r"\b([0-9]+)\b", rule)
                rule = int(m.group(1)) if m else -1
            # fi

            state_elem = step.find("program_state")
            state = {var.tag: int(var.text) for var in state_elem if var.tag != "ble"}
            # sort it
            state = dict(sorted(state.items()))
            result.append({"rule": rule, "program_state": state})
    except Exception as e:
        print(f"Failed to parse XML: {e}")
    return result


def generate_xml_trace(trace: List[tuple]) -> str:
    """
    trace: List of tuples. Each tuple is (rule, program_state_dict)
           where program_state_dict is a dict mapping variable names to values
    """
    root = Element("answer")

    for rule, state in trace:
        step = SubElement(root, "step")
        lineno = SubElement(step, "rule")
        lineno.text = str(rule[5:])

        prog_state = SubElement(step, "program_state")
        for var_name, var_value in state.items():
            var_elem = SubElement(prog_state, str(var_name))
            var_elem.text = str(var_value)

    # Pretty print
    rough_string = tostring(root, encoding="unicode")
    pretty_xml = parseString(rough_string).toprettyxml(indent="  ")
    return "\n".join(pretty_xml.splitlines()[1:])


def push_jsonl_split_to_hf(
    *,
    repo_id: str,
    config_name: str = None,
    split: str = None,
    data_list: List[Dict],
    allowed_splits: Set[str],
    private: bool = True,
) -> None:
    """
    Push JSONL data to a Hugging Face *dataset* repo, selectively updating only chosen splits.

    - repo_id: e.g., "your-username/your-dataset"
    - train_jsonl/test_jsonl: paths to local .jsonl files
    - push_splits: subset of {"train", "test"} to update on the Hub
    - private: create repo as private if it doesn't exist
    - revision: optional branch name (e.g., "main", "dev"); if None uses default

    Requirements:
      pip install datasets huggingface_hub
      huggingface-cli login
    """

    if split not in allowed_splits:
        raise ValueError(f"Unknown split in push_to_split: {split}. Allowed: {sorted(allowed_splits)}")

    new_ds = Dataset.from_list(data_list)
    new_ds.push_to_hub(repo_id, config_name=config_name, split=split, private=private)

    print(f"✅ Pushed split {split} to {repo_id} (other splits preserved).")


if __name__ == "__main__":
    # Examples:

    # 1) Push ONLY train (preserve test if it already exists)
    # push_jsonl_two_splits(
    #     repo_id="your-username/semantics-prompts",
    #     train_jsonl="data/train.jsonl",
    #     push_splits=("train",),
    # )

    # 2) Push ONLY test (preserve train)
    # push_jsonl_two_splits(
    #     repo_id="your-username/semantics-prompts",
    #     test_jsonl="data/test.jsonl",
    #     push_splits=("test",),
    # )

    # 3) Push BOTH
    # push_jsonl_two_splits(
    #     repo_id="your-username/semantics-prompts",
    #     train_jsonl="data/train.jsonl",
    #     test_jsonl="data/test.jsonl",
    #     push_splits=("train", "test"),
    # )
    pass
