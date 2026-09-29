import os
import sys
import json
import shutil
import importlib.util
from pathlib import Path
from llm_interpreter.macros import Macros


def read_from_txt_file(txt_file: str):
    content: str = None
    with open(txt_file, "r", encoding="utf-8") as file:
        content = file.read()
    # htiw
    return content


# fed


def clear_dir(dir_path: str):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print(f"Error: {e}")
    else:
        print(f"Directory '{dir_path}' does not exist.")


# fed


def read_from_json_file(json_file: str):
    content: dict = {}
    with open(json_file, "r", encoding="utf-8") as file:
        content = json.load(file)
    # htiw
    return content


# fed


def write_to_tmp(content: str, txt_file: str):
    with open(f"{Macros.tmp_dir}/{txt_file}", "w", encoding="utf-8") as file:
        file.write(content)
    # htiw


# fed


def mk_tmp_dir(dir_name: str):
    os.mkdir(f"{Macros.tmp_dir}/{dir_name}")


# fed


def mk_dir(dir_name: str, path: str):
    os.mkdir(f"{path}/{dir_name}")


# fed


def write_to_dir(content: str, txt_file: str, path: str):
    with open(f"{path}/{txt_file}", "w", encoding="utf-8") as file:
        file.write(content)
    # htiw


# fed


def load_class_from_file(filepath: str, class_name: str):
    filepath = Path(filepath)
    module_name = filepath.stem
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


# fed
