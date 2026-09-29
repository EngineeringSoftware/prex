"""
Ollama experiment runner implementation.
"""

import seutil as su
import requests

from llm_interpreter.experiments.base import BaseRunner

logger = su.log.get_logger(__name__, su.log.INFO)


class OllamaRunner(BaseRunner):
    def __init__(self, model_config_file: str, **kwargs):
        super().__init__(**kwargs)
        # load model's config
        self.model_config = su.io.load(model_config_file)
        # self.gpt_model = gpt_model   (openai specific)
        self.ollama_base_url = kwargs.get("ollama_base_url", "http://localhost:11434")
        self.test_client()

    # fed

    def test_client(self):
        # Test connection to Ollama
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code != 200:
                logger.warning(f"Ollama connection test failed: {response.status_code}")
            else:
                logger.info(
                    f"Successfully connected to Ollama at {self.ollama_base_url}"
                )
        except Exception as e:
            logger.warning(
                f"Could not connect to Ollama at {self.ollama_base_url}: {e}"
            )

    def _assemble_chat(self, chat_prompt: list[dict]) -> list[dict]:
        """Override to include system prompt for Ollama."""
        # Check if the prompt already has a system message
        if chat_prompt and chat_prompt[0].get("role") == "system":
            # If it already has a system message, don't add another one
            chat = chat_prompt
        else:
            # Otherwise, add the default system prompt
            chat = self._system + chat_prompt

        return chat

    def _query(
        self,
        chat: list[dict],
        stop: list[str] = [],
    ) -> list[str]:
        try:
            # Convert chat format to Ollama format
            messages = []
            for message in chat:
                messages.append(
                    {"role": message["role"], "content": message["content"]}
                )

            # Prepare the request payload for Ollama API
            if "temperature" not in self.model_config:
                raise ValueError("temperature not found in model config")
            if "max_completion_tokens" not in self.model_config:
                raise ValueError("max_completion_tokens not found in model config")

            payload = {
                "model": self.args.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.model_config["temperature"],
                    "num_predict": self.model_config["max_completion_tokens"],
                },
            }

            if stop:
                payload["options"]["stop"] = stop

            # Make the API call to Ollama
            response = requests.post(f"{self.ollama_base_url}/api/chat", json=payload)

            if response.status_code == 200:
                result = response.json()
                return [result["message"]["content"]]
            else:
                logger.warning(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                return []

        except Exception as e:
            error_msg = (
                f"Error while running query with model {self.args.model_name}: {e}"
            )
            logger.warning(error_msg)
            return [error_msg]

    # fed
