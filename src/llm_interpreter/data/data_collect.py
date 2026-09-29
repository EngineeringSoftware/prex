import seutil as su
from typing import List
import os
import json
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


from llm_interpreter.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)


class DataCollector:
    def find_c_cpp_code(self, repository_path: str) -> List[str]:
        c_cpp_code = []
        for root, _, files in os.walk(repository_path):
            for file in files:
                if file.endswith((".c", ".cpp")):
                    c_cpp_code.append(
                        su.io.load(os.path.join(root, file), su.io.Fmt.txt)
                    )
        return c_cpp_code

    def collect_cpp_code(self):
        """
        Collect the C++ competition data.
        """
        collected_repos = self.get_repo_file()
        all_cpp_code = []
        for repo in collected_repos:
            repo_path = Macros.downloads_dir / repo["full_name"]
            c_cpp_code = self.find_c_cpp_code(repo_path)
            all_cpp_code.extend(c_cpp_code)
        logger.info(f"Found {len(all_cpp_code)} C/C++ code in all repos")
        su.io.dump(
            Macros.data_dir / "cpp_competition_code.jsonl",
            all_cpp_code,
        )

    def collect_submissions_with_tests(self):
        """
        Combines the Codeforces problems and submissions datasets to create a
        single file containing C++ solutions and their public tests.
        """

        # 1. Load the problems dataset and create a lookup for public tests
        # =================================================================
        print("Loading the problems dataset to get public tests...")
        problems_dataset = load_dataset("open-r1/codeforces", split="train")

        problem_tests_lookup = {}
        for problem in tqdm(problems_dataset, desc="Processing problems"):
            # Create a unique problem_id to match the submissions dataset
            problem_id = f"{problem['contest_id']}/{problem['index']}"
            if problem["official_tests"]:
                problem_tests_lookup[problem_id] = problem["official_tests"]

        print(f"Loaded tests for {len(problem_tests_lookup)} problems.")

        # 2. Stream submissions and combine them with the tests
        # =======================================================
        print("\nLoading the 'selected_accepted' submissions stream...")
        # We use the 'selected_accepted' subset as it is pre-filtered and efficient.
        # streaming=True is memory-efficient.
        submissions_dataset = load_dataset(
            "open-r1/codeforces-submissions",
            "selected_accepted",
            split="train",
            streaming=True,
        )

        output_file = "codeforces_cpp_with_tests.jsonl"
        print(f"Processing submissions and writing to {output_file}...")

        with open(output_file, "w") as f:
            for submission in tqdm(submissions_dataset, desc="Processing submissions"):
                # Filter for C++ submissions, consistent with the previous request
                if "c++" in submission["programmingLanguage"].lower():
                    # Look up the tests using the submission's problem_id
                    public_tests = problem_tests_lookup.get(submission["problem_id"])

                    if public_tests:
                        # Create the final data point
                        data_point = {
                            "problem_id": submission["problem_id"],
                            "sourceCode": submission["source"],
                            "publicTests": public_tests,
                        }

                        # Write the JSON object as a single line in the file
                        f.write(json.dumps(data_point) + "\n")

        print(f"\n Script finished. Data is saved in {output_file}")

    def get_cpp_data_token_distribution(self) -> List[int]:
        data_file = Macros.data_dir / "dataset" / "dataset-data-gen-uk-IMP-SOS.jsonl"
        self.get_tokens_distribution(
            data_file, model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
        )

    ##  Helper functions

    def get_tokens_distribution(self, data_file: str, model_name: str) -> List[int]:
        from llm_interpreter.experiments.prompt_maker import (
            make_dataset_generation_prompt,
        )

        filtered_data = []
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load data
        token_lengths = []
        data = su.io.load(data_file, iter_line=True)
        for dt in tqdm(data, desc="Processing data for token lengths"):
            chat = make_dataset_generation_prompt(dt)
            tokenized_chat = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_special_tokens=True,
                add_generation_prompt=False,  # Do not add prompt for the *next* assistant turn for length calculation
                return_length=True,
                truncation=False,  # Get full length even if it exceeds max_model_input_length
            )
            if len(tokenized_chat) > 10000:
                print(
                    f"Warning: Token length {len(tokenized_chat)} exceeds 10000 for chat: {chat}"
                )
                continue
            token_lengths.append(len(tokenized_chat))
            filtered_data.append(dt)
        #

        lengths_array = np.array(token_lengths)

        print("\n--- Token Length Statistics for Chat Conversations ---")
        print(f"Number of conversations: {len(lengths_array)}")
        print(f"Minimum length: {np.min(lengths_array)}")
        print(f"Maximum length: {np.max(lengths_array)}")
        print(f"Average length: {np.mean(lengths_array):.2f}")
        print(f"Median length: {np.median(lengths_array)}")
        print(f"Standard deviation: {np.std(lengths_array):.2f}")
        data_file = Macros.data_dir / "dataset" / "dataset-data-gen-uk-IMP-SOS.jsonl"
        su.io.dump(data_file, filtered_data)

    def get_repo_file(self) -> List:
        return su.io.load(
            Macros.downloads_dir / "cpp_competition_repos/repos.json",
            su.io.Fmt.jsonPretty,
        )
