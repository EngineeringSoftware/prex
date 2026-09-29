"""
Batch experiment implementation.
"""

import os
import json
from typing import Union, List, Dict
from pathlib import Path
from tqdm import tqdm
import seutil as su
import dataclasses
from enum import IntEnum, Enum
import anthropic
from openai import OpenAI
import time

from llm_interpreter.macros import Macros
from llm_interpreter.experiments.base import BaseRunner
from llm_interpreter.experiments.prompts import system_prompt

logger = su.log.get_logger(__name__, su.log.INFO)

class VENDOR(IntEnum):
    ANTHROPIC = 1
    GOOGLE = 2
    OPENAI = 3
#

@dataclasses.dataclass(frozen=True)
class MODEL:
    name: str = ""
    reasoning: bool = False
    vendor: VENDOR = None
#

MODELS: Dict[str, MODEL] = {
    "claude-sonnet-4-6": MODEL(
        name="claude-sonnet-4-6",
        reasoning=True,
        vendor=VENDOR.ANTHROPIC,
    ),
    "gpt-5": MODEL(
        name="gpt-5",
        reasoning=True,
        vendor=VENDOR.OPENAI,
    ),
    "gpt-5-mini": MODEL(
        name="gpt-5-mini",
        reasoning=True,
        vendor=VENDOR.OPENAI,
    ),
    "gpt-5.4": MODEL(
        name="gpt-5.4",
        reasoning=True,
        vendor=VENDOR.OPENAI,
    ),
    "gpt-5.4-mini": MODEL(
        name="gpt-5.4-mini",
        reasoning=True,
        vendor=VENDOR.OPENAI,
    ),
    "gemini-3-pro": MODEL(
        name="gemini-3-pro",
        reasoning=True,
        vendor=VENDOR.GOOGLE,
    ),
    "gemini-2.5-pro": MODEL(
        name="gemini-2.5-pro",
        reasoning=True,
        vendor=VENDOR.GOOGLE,
    ),
}
#

class BatchRunner(BaseRunner):
    def __init__(
        self,
        max_tokens: int,
        reasoning_effort: str = "medium",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.model = MODELS[self.args.model_name]
        self.reasoning_effort = reasoning_effort
    # fed

    def _get_price_1_million_input_token(self) -> float:
        raise NotImplementedError("Not implemented for batch mode")

    # fed

    def _get_price_1_million_output_token(self) -> float:
        raise NotImplementedError("Not implemented for batch mode")

    # fed

    def _query(
        self,
        chat: list[dict],
        stop: list[str] = [],
    ) -> list[str]:
        raise NotImplementedError("Not implemented for batch mode")
    
    # fed

    def run(self, dataset: List, print_output: bool = False) -> List:
        """
        Run the experiments on the provided dataset
        """
        logger.info(f"Start running experiments: {str(self.args)} ...")
        # temp = 0.0
        results: List[Dict[str,str]] = []
        prompts: List[Dict[str,str]] = []
        for i, p in enumerate(tqdm(dataset, total=len(dataset))):
            result: Dict[str, str] = (self._prepare_prompt(p))[0]
            result["custom_id"] = str(i)
            if self.args.setup_name != "nk" and self.model.vendor == VENDOR.ANTHROPIC:
                match self.args.task:
                    case "op" | "etp":
                        content: str = result["content"]
                        content_partition: List[str] = result["content"].split("Here is the IMP program")
                        del result["content"]
                        result["content-prefix"] = content_partition[0]
                        result["content-suffix"] = f"Here is the IMP program\n{content_partition[1]}"
                #hctam
            #fi
            for key, val in p.items():
                if key not in {"syntax", "semantics"}:
                    result[key]=val
                #
            #
            results.append(result)
        #
        match self.model.vendor:
            case VENDOR.ANTHROPIC:
                results = self._run_anthropic_batch(results)
            case VENDOR.OPENAI:
                results = self._run_openai_batch(results)
            case VENDOR.GOOGLE:
                results = self._run_google_batch(results)
        #
        return results
    #fed

    def _run_anthropic_batch(self, results: List[Dict[str, str]]):
        requests: List[Dict] = []
        client = anthropic.Anthropic()
        for result in results:
            request: Dict = {}
            request["model"] = self.model.name
            request["max_tokens"] = self.max_tokens
            if "content-prefix" in result:
                request["system"] = [
                    {
                        "type": "text",
                        "text": f"{system_prompt}\n{result['content-prefix']}",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                request["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            #
            request["messages"] = [
                {
                    "role": "user",
                    "content": result["content-suffix"] if "content-suffix" in result else result["content"]
                }
            ]
            if self.model.reasoning:
                request["thinking"] = {"type": "adaptive"}
                request["output_config"] = {"effort": self.reasoning_effort}
            #
            requests.append(
                {
                    "custom_id": result["custom_id"],
                    "params": request
                }
            )
        #
        logger.info(f"Writing to batch file: {str(self.args)} ...")
        batch_file_name: str = "-".join([
            self.args.task,
            self.args.setup_name,
            self.args.expr_name,
            self.args.dataset_name,
            self.model.name,
            "batch",
            str(self.args.random_seed),
        ])
        su.io.dump(Macros.batch_dir / f"{batch_file_name}.jsonl", requests)
        message_batch = client.messages.batches.create(requests=requests)
        while True:
            message_batch = client.messages.batches.retrieve(message_batch.id)
            if message_batch.processing_status == "ended":
                break
            print(f"Batch {message_batch.id} is still processing...")
            time.sleep(300)
        contents: Dict[str,str] = {}
        for entry in client.messages.batches.results(message_batch.id):
            if entry.result.type == "succeeded":
                content = next((b.text for b in entry.result.message.content if b.type == "text"), None)
                if content is None:
                    contents[entry.custom_id] = ""
                else:
                    contents[entry.custom_id] = content
            #
        #

        for result in results:
            if result["custom_id"] in contents:
                result["model-prediction"] = contents[result["custom_id"]]
            else:
                result["model-prediction"] = ""
        #
        return results
    #

    

    def _run_openai_batch(self, results: List[Dict[str, str]]):
        requests: List[Dict] = []
        client = OpenAI()
        for result in results:
            request: Dict = {}
            request["model"] = self.model.name
            request["max_completion_tokens"] = self.max_tokens
            request["messages"] = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": result["content"]
                }
            ]
            if self.model.reasoning:
                request["reasoning_effort"] = "medium"
            #
            requests.append(
                {
                    "custom_id": result["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request
                }
            )
        #
        logger.info(f"Writing to batch file: {str(self.args)} ...")
        batch_file_name: str = "-".join([
            self.args.task,
            self.args.setup_name,
            self.args.expr_name,
            self.args.dataset_name,
            self.model.name,
            "batch",
            str(self.args.random_seed),
        ])
        su.io.dump(Macros.batch_dir / f"{batch_file_name}.jsonl", requests)

        batch_file = client.files.create(
            file=open(Macros.batch_dir / f"{batch_file_name}.jsonl", "rb"),
            purpose="batch"
        )
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        while batch.status not in ("completed", "failed", "expired"):
            time.sleep(300)
            batch = client.batches.retrieve(batch.id)
            print(f"Batch ID: {batch.id}, Status: {batch.status}")

        content = client.files.content(batch.output_file_id)
        contents: Dict[str,str] = {}
        for line in content.text.strip().split("\n"):
            r = json.loads(line)
            
            if r["error"] is not None:
                continue

            if r["response"]["status_code"] != 200:
                continue

            choices = r["response"]["body"].get("choices", [])

            if not choices:
                continue

            choice = choices[0]
            finish = choice.get("finish_reason")
            text = choice["message"].get("content", "")
            contents[r["custom_id"]] = text
        #
        
        for result in results:
            if result["custom_id"] in contents:
                result["model-prediction"] = contents[result["custom_id"]]
            else:
                result["model-prediction"] = ""
        #
        return results
    #

    def _run_google_batch(self, results: List[Dict[str, str]]):
        raise NotImplementedError("Not implemented")
    #

# ssalc
