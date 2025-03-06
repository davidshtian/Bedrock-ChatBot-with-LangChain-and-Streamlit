from typing import Dict
from config import config
from langchain_aws import ChatBedrockConverse

class ChatModel:
    def __init__(self, model_name: str, model_kwargs: Dict):
        self.model_config = config["models"][model_name]
        self.model_id = self.model_config["model_id"]
        print(f"david: {self.model_id}")
        
        # 基础参数，仅包含必需的参数
        base_kwargs = {
            "model": self.model_id,
            "max_tokens": model_kwargs["max_tokens"],
        }
        
        # 根据模型配置添加额外参数
        if self.model_config.get("thinking"):
            base_kwargs["additional_model_request_fields"] = {
                "thinking": {"type": "enabled", "budget_tokens": 8192}
            }
        else:
            # 只在非thinking模式下添加temperature和top_p
            base_kwargs.update({
                "temperature": model_kwargs.get("temperature"),
                "top_p": model_kwargs.get("top_p")
            })
            
            # 添加top_k相关配置
            if "nova" in self.model_id:
                base_kwargs["additional_model_request_fields"] = {
                    "inferenceConfig": {"topK": model_kwargs["top_k"]}
                }
            elif not "mistral" in self.model_id:
                base_kwargs["additional_model_request_fields"] = {
                    "top_k": model_kwargs["top_k"]
                }
            
        self.llm = ChatBedrockConverse(**base_kwargs)
