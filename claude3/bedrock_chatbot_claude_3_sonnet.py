import streamlit as st
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler


MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

CLAUDE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

INIT_MESSAGE = {"role": "assistant",
                "content": "Hi! I'm Claude on Bedrock. How may I help you?"}

SYSTEM_PROMPT = "You're a cool assistant, love to response with emoji."

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Set Streamlit page configuration
st.set_page_config(page_title='🤖 Chat with Bedrock', layout='wide')
st.title("🤖 Chat with Bedrock")

# Sidebar info
with st.sidebar:
    st.markdown("## Inference Parameters")
    TEMPERATURE = st.slider("Temperature", min_value=0.0,
                            max_value=1.0, value=1.0, step=0.1)
    TOP_P = st.slider("Top-P", min_value=0.0,
                      max_value=1.0, value=1.00, step=0.01)
    TOP_K = st.slider("Top-K", min_value=1,
                      max_value=500, value=500, step=5)
    MAX_TOKENS = st.slider("Max Token", min_value=0,
                           max_value=4096, value=4096, step=8)
    MEMORY_WINDOW = st.slider("Memory Window", min_value=0,
                              max_value=10, value=10, step=1)


# Initialize the ConversationChain
def init_conversationchain() -> ConversationChain:
    model_kwargs = {'temperature': TEMPERATURE,
                    'top_p': TOP_P,
                    'top_k': TOP_K,
                    'max_tokens': MAX_TOKENS,
                    'system': SYSTEM_PROMPT}

    llm = BedrockChat(
        model_id=MODEL_ID,
        model_kwargs=model_kwargs,
        streaming=True
    )

    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferWindowMemory(
            k=MEMORY_WINDOW, ai_prefix="Assistant", chat_memory=StreamlitChatMessageHistory(), return_messages=True),
        prompt=CLAUDE_PROMPT
    )

    # Store LLM generated responses

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [INIT_MESSAGE]

    return conversation


def generate_response(conversation: ConversationChain, input_text: str) -> str:
    return conversation.run(input=input_text, callbacks=[StreamHandler(st.empty())])


# Re-initialize the chat
def new_chat() -> None:
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []
    conv_chain = init_conversationchain()


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

# Initialize the chat
conv_chain = init_conversationchain()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User-provided prompt
prompt = st.chat_input()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # print(st.session_state.messages)
        response = generate_response(conv_chain, prompt)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
