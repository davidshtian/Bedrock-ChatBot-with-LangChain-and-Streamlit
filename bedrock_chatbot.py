import streamlit as st
from langchain.llms import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

DEFAULT_CLAUDE_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
Assistant:"""

CLAUDE_PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=DEFAULT_CLAUDE_TEMPLATE)

INIT_MESSAGE = {"role": "assistant", "content": "Hi! I'm Claude on Bedrock. How may I help you?"}


# Set Streamlit page configuration
st.set_page_config(page_title='🤖 Chat with Bedrock', layout='wide')
st.title("🤖 Chat with Bedrock")

# Sidebar info
with st.sidebar:
    st.markdown("## Inference Parameters")
    TEMPERATURE = st.slider("Temperature", min_value=0.0,
                            max_value=1.0, value=0.1, step=0.1)
    TOP_P = st.slider("Top-P", min_value=0.0,
                      max_value=1.0, value=0.9, step=0.01)
    TOP_K = st.slider("Top-K", min_value=1,
                      max_value=500, value=10, step=5)
    MAX_TOKENS = st.slider("Max Token", min_value=0,
                           max_value=2048, value=1024, step=8)
    MEMORY_WINDOW = st.slider("Memory Window", min_value=0,
                              max_value=10, value=3, step=1)

# Initialize the ConversationChain


def init_conversationchain():
    model_kwargs = {'temperature': TEMPERATURE,
                    'top_p': TOP_P,
                    'top_k': TOP_K,
                    'max_tokens_to_sample': MAX_TOKENS}

    llm = Bedrock(
        model_id="anthropic.claude-v2",
        model_kwargs=model_kwargs
    )

    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferWindowMemory(
            k=MEMORY_WINDOW, ai_prefix="Assistant", chat_memory=StreamlitChatMessageHistory()),
        prompt=CLAUDE_PROMPT
    )

    # Store LLM generated responses

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [INIT_MESSAGE]

    return conversation


def generate_response(conversation, input_text):
    return conversation.run(input=input_text)


# Re-initialize the chat
def new_chat():
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
        with st.spinner("Thinking..."):
            response = generate_response(conv_chain, prompt)
            st.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
