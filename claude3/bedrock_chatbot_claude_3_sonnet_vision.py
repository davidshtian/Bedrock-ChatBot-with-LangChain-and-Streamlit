import streamlit as st
import random
import base64
from io import BytesIO
from PIL import Image
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler


MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

CLAUDE_PROMPT = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input")
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


def generate_response(conversation: ConversationChain, input) -> str:
    return conversation.invoke(input, {"callbacks": [StreamHandler(st.empty())]})

# Re-initialize the chat
def new_chat() -> None:
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []
    st.session_state["file_uploader_key"] = random.randint(1,100)
    conv_chain = init_conversationchain()

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

# Initialize the chat
conv_chain = init_conversationchain()

# Image
if "file_uploader_key" not in st.session_state.keys():
    st.session_state["file_uploader_key"] = 0

uploaded_files = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=st.session_state["file_uploader_key"])

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if uploaded_files:
            if "images" in message.keys() and message["images"] != []:
                num_cols = 10 
                cols = st.columns(num_cols)
                i = 0

                for image in message["images"]:
                    for uploaded_file in uploaded_files:
                        if image == uploaded_file.file_id:
                            img = Image.open(uploaded_file)
                            
                            with cols[i]:
                                st.image(img, caption="", width=75)
                                i += 1
                            
                            if i >= num_cols:
                                i = 0
        if message["role"] == "user":                                   
            if type(message["content"]) is str:
                st.markdown(message["content"])
            elif type(message["content"]) is dict:
                st.markdown(message["content"]["input"][0]["content"][0]["text"])
            else:
                st.markdown(message["content"][0]["text"])
        
        if message["role"] == "assistant":                                   
            if type(message["content"]) is str:
                st.markdown(message["content"])
            elif "response" in message["content"].keys():
                st.markdown(message["content"]["response"])

# User-provided prompt
prompt = st.chat_input()

# Get images from previous messages
message_images_list = []
for message in st.session_state.messages:
    if message["role"] == 'user' and "images" in message.keys() and message["images"] != []:
        message_images_list += message["images"]

# Show image in corresponding chat box
uploaded_file_ids = []
if uploaded_files and len(message_images_list) < len(uploaded_files):
    with st.chat_message("user"):
        num_cols = 10 
        cols = st.columns(num_cols)
        i = 0
        content_images = []
        
        for uploaded_file in uploaded_files:
            if uploaded_file.file_id not in message_images_list:
                uploaded_file_ids.append(uploaded_file.file_id)
                img = Image.open(uploaded_file)
                with BytesIO() as output_buffer:
                    img.save(output_buffer, format=img.format)
                    content_image = base64.b64encode(output_buffer.getvalue()).decode('utf8')
                content_images.append(content_image)
                with cols[i]:
                    st.image(img, caption="", width=75)
                    i += 1
                if i >= num_cols:
                    i = 0 
                                    
        if prompt:

            prompt_text = {"type": "text", "text": prompt}
            prompt_new = [prompt_text]
            for i in content_images:
                prompt_image = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": i}}
                prompt_new += [prompt_image]
            st.session_state.messages.append({"role": "user", "content": prompt_new, "images": uploaded_file_ids})
            st.markdown(prompt)

elif prompt:
    prompt_text = { "type": "text", "text": prompt}
    prompt_new = [prompt_text]
    st.session_state.messages.append({"role": "user", "content": prompt_new})
    with st.chat_message("user"):
        st.markdown(prompt)
        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = generate_response(conv_chain, [{"role": "user", "content": prompt_new}])
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
