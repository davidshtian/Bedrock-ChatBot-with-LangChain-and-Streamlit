from typing import Dict, List, Union
from config import config

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import BedrockChat


class ChatModel:
    def __init__(self, model_name: str, model_kwargs: Dict):
        self.model_config = config["models"][model_name]
        self.model_id = self.model_config["model_id"]
        self.model_kwargs = model_kwargs
        self.llm = BedrockChat(model_id=self.model_id, model_kwargs=model_kwargs, streaming=True)

    def format_messages(self, messages: List[Union[AIMessage, HumanMessage]]) -> List[Union[AIMessage, HumanMessage]]:
        """
        Format the messages for the LangChain conversation chain.
        """
        for i, message in enumerate(messages):
            if isinstance(message.content, list):
                if "role" in message.content[0]:
                    if message.type == "ai":
                        message = AIMessage(message.content[0]["content"])
                    if message.type == "human":
                        message = HumanMessage(message.content[0]["content"])
                    messages[i] = message
        return messages

    def format_prompt(self, prompt: str) -> Union[str, List[Dict]]:
        """
        Format the input prompt according to the model's requirements.
        """
        model_config = self.model_config
        if model_config.get("input_format") == "text":
            return prompt
        elif model_config.get("input_format") == "list_of_dicts":
            prompt_text = {"type": "text", "text": prompt}
            return [prompt_text]
        else:
            raise ValueError(f"Unsupported input format for model: {self.model_id}")
