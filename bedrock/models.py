from typing import Dict, List, Union
from config import config

from langchain_aws import ChatBedrock


class ChatModel:
    def __init__(self, model_name: str, model_kwargs: Dict):
        self.model_config = config["models"][model_name]
        self.model_id = self.model_config["model_id"]
        self.model_kwargs = model_kwargs
        self.llm = ChatBedrock(model_id=self.model_id, model_kwargs=model_kwargs, streaming=True)