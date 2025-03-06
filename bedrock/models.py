from typing import Dict
from config import config

from langchain_aws import ChatBedrockConverse


class ChatModel:
    def __init__(self, model_name: str, model_kwargs: Dict):
        self.model_config = config["models"][model_name]
        self.model_id = self.model_config["model_id"]
        self.model_kwargs = model_kwargs

        if self.model_config["thinking"]:
            self.llm = ChatBedrockConverse(
                model=self.model_id,
                max_tokens=self.model_kwargs["max_tokens"],
                additional_model_request_fields={
                    "thinking": {"type": "enabled", "budget_tokens": 8192},
                },
            )
        elif "mistral" in self.model_id:
            self.llm = ChatBedrockConverse(
                model=self.model_id,
                max_tokens=self.model_kwargs["max_tokens"],
                temperature=self.model_kwargs["temperature"],
                top_p=self.model_kwargs["top_p"],
            )
        elif "nova" in self.model_id:
            self.llm = ChatBedrockConverse(
                model=self.model_id,
                max_tokens=self.model_kwargs["max_tokens"],
                temperature=self.model_kwargs["temperature"],
                top_p=self.model_kwargs["top_p"],
                additional_model_request_fields={
                    "inferenceConfig": {"topK": self.model_kwargs["top_k"]}
                },
            )
        else:
            self.llm = ChatBedrockConverse(
                model=self.model_id,
                max_tokens=self.model_kwargs["max_tokens"],
                temperature=self.model_kwargs["temperature"],
                top_p=self.model_kwargs["top_p"],
                additional_model_request_fields={"top_k": self.model_kwargs["top_k"]},
            )
