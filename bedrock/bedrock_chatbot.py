import base64
import random
from io import BytesIO
from typing import Tuple

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from PIL import Image

from config import config
from models import ChatModel

CLAUDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input"),
    ]
)

INIT_MESSAGE = {
    "role": "assistant",
    "content": "Hi! I'm your AI Bot on Bedrock. How may I help you?",
}

class StreamHandler(BaseCallbackHandler):
    """
    Callback handler to stream the generated text to Streamlit.
    """

    def __init__(self, container: st.container) -> None:
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Append the new token to the text and update the Streamlit container.
        """
        self.text += token
        self.container.markdown(self.text)

def render_sidebar() -> Tuple[float, float, int, int, int, str]:
    """
    Render the sidebar UI and return the inference parameters.
    """
    with st.sidebar:
        st.markdown("## Inference Parameters")
        model_name_select = st.selectbox(
            'Model',
            list(config["models"].keys()),
            key=f"{st.session_state['widget_key']}_Model_Id",
        )

        st.session_state["model_name"] = model_name_select

        model_config = config["models"][model_name_select]

        if model_config.get("system_prompt_disabled", False):
            system_prompt = st.text_area(
                "System Prompt",
                "",
                key=f"{st.session_state['widget_key']}_System_Prompt",
                disabled=True
            )
        else:
            system_prompt = st.text_area(
                "System Prompt",
                model_config.get("default_system_prompt", ""),
                key=f"{st.session_state['widget_key']}_System_Prompt",
            )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=model_config.get("temperature", 1.0),
            step=0.1,
            key=f"{st.session_state['widget_key']}_Temperature",
        )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                top_p = st.slider(
                    "Top-P",
                    min_value=0.0,
                    max_value=1.0,
                    value=model_config.get("top_p", 1.0),
                    step=0.01,
                    key=f"{st.session_state['widget_key']}_Top-P",
                )
            with col2:
                top_k = st.slider(
                    "Top-K",
                    min_value=1,
                    max_value=model_config.get("max_top_k", 500),
                    value=model_config.get("top_k", 500),
                    step=5,
                    key=f"{st.session_state['widget_key']}_Top-K",
                )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                max_tokens = st.slider(
                    "Max Token",
                    min_value=0,
                    max_value=4096,
                    value=model_config.get("max_tokens", 4096),
                    step=8,
                    key=f"{st.session_state['widget_key']}_Max_Token",
                )
            with col2:
                memory_window = st.slider(
                    "Memory Window",
                    min_value=0,
                    max_value=10,
                    value=model_config.get("memory_window", 10),
                    step=1,
                    key=f"{st.session_state['widget_key']}_Memory_Window",
                )

    return temperature, top_p, top_k, max_tokens, memory_window, system_prompt

def render_chat_area(chat_model: ChatModel) -> None:
    """
    Render the chat area UI and handle message generation.
    """
    conversation = ConversationChain(
        llm=chat_model.llm,
        verbose=True,
        memory=ConversationBufferWindowMemory(
            k=chat_model.model_kwargs.get("memory_window", 10),
            ai_prefix="Assistant",
            chat_memory=StreamlitChatMessageHistory(),
            return_messages=True,
        ),
        prompt=CLAUDE_PROMPT,
    )

    # Store LLM generated responses
    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]

    # Modify langchain_messages format
    st.session_state["langchain_messages"] = chat_model.format_messages(
        st.session_state.get("langchain_messages", [])
    )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                if isinstance(message["content"], str):
                    st.markdown(message["content"])
                elif isinstance(message["content"], dict):
                    st.markdown(message["content"]["input"][0]["content"][0]["text"])
                else:
                    st.markdown(message["content"][0]["text"])

            if message["role"] == "assistant":
                if isinstance(message["content"], str):
                    st.markdown(message["content"])
                elif "response" in message["content"]:
                    st.markdown(message["content"]["response"])

    # User-provided prompt
    prompt = st.chat_input()

    if prompt:
        formatted_prompt = chat_model.format_prompt(prompt)
        st.session_state.messages.append({"role": "user", "content": formatted_prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = conversation.invoke(
                {"input": [{"role": "user", "content": formatted_prompt}]},
                {"callbacks": [StreamHandler(st.empty())]},
            )
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

def render_image_uploader() -> None:
    """
    Render the image uploader UI.
    """
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    model_config = config["models"][st.session_state["model_name"]]
    if model_config.get("image_upload_disabled", False):
        uploaded_files = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=st.session_state["file_uploader_key"],
            disabled=True
        )
    else:
        uploaded_files = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=st.session_state["file_uploader_key"],
        )

    # TODO: Implement image handling logic

def main():
    st.set_page_config(page_title="🤖 Chat with Bedrock", layout="wide")
    st.title("🤖 Chat with Bedrock")

    # Generate a unique widget key only once
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))

    # Add a button to start a new chat
    st.sidebar.button("New Chat", on_click=lambda: st.session_state.clear(), type="primary")

    sidebar_params = render_sidebar()
    model_config = config["models"][st.session_state["model_name"]]
    model_kwargs = {
        "temperature": sidebar_params[0],
        "top_p": sidebar_params[1],
        "top_k": sidebar_params[2],
        "max_tokens": sidebar_params[3],
    }
    if not model_config.get("system_prompt_disabled", False):
        model_kwargs["system"] = sidebar_params[5]

    chat_model = ChatModel(st.session_state["model_name"], model_kwargs)

    render_image_uploader()
    render_chat_area(chat_model)

if __name__ == "__main__":
    main()


