from typing import Dict
from config import config
from langchain_aws import ChatBedrockConverse


class ChatModel:
    def __init__(self, model_name: str, model_kwargs: Dict):
        self.model_config = config["models"][model_name]
        self.model_id = self.model_config["model_id"]

        # Basic parameters, including only the essential ones
        base_kwargs = {
            "model": self.model_id,
            "max_tokens": model_kwargs["max_tokens"],
        }

        # Add additional parameters based on the model configuration
        if self.model_config.get("thinking") and "anthropic" in self.model_id:
            base_kwargs["additional_model_request_fields"] = {
                "thinking": {"type": "enabled", "budget_tokens": 8192}
            }
        else:
            # Add temperature and top_p only in non-thinking mode or deepseek r1
            base_kwargs.update(
                {
                    "temperature": model_kwargs.get("temperature"),
                    "top_p": model_kwargs.get("top_p"),
                }
            )

            # Add top_k related configuration
            if "nova" in self.model_id:
                base_kwargs["additional_model_request_fields"] = {
                    "inferenceConfig": {"topK": model_kwargs["top_k"]}
                }
            elif "anthropic" in self.model_id:
                base_kwargs["additional_model_request_fields"] = {
                    "top_k": model_kwargs["top_k"]
                }

        self.llm = ChatBedrockConverse(**base_kwargs)
