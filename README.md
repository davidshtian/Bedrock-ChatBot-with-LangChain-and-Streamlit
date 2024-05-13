# Bedrock ChatBot with LangChain and Streamlit 💬 
A simple and clear example for implement a chatbot with Bedrock + LangChain + Streamlit. Just install and run the code~ 🚀
```
pip install -r requirements.txt
streamlit run bedrock/bedrock_chatbot.py
```
> Note: if you're going to use web search function, add your serpapi key to bedrock/.env file~


## RAG (Retrieval-Augmented Generation) Feature

We have added a new feature that allows the AI model to pull in information from a large corpus of documents, providing more detailed and accurate responses. This feature uses the RAG technique, which combines the benefits of extractive and abstractive summarization.

To use the RAG feature, select 'RAG' from the 'Options' dropdown in the chatbot interface.

# 24/05/12 Updates
## Indexing Documents and Using RAG Feature

To use the RAG (Retrieval-Augmented Generation) feature, you need to index your documents using the `bedrock_indexer.py` script. This script creates a FAISS index from the documents in a directory.

Here's how to use it:

1. Add your documents to the "documents" directory. These can be text files or other types of documents that you want the RAG model to use for information retrieval.
2. Run the `bedrock_indexer.py` script:

```bash
python bedrock_indexer.py
```

# 24/04/15 Updates
**Add Web Search (via SerpAPI) and role prompt option!** 🎉🎉🎉

<img width="1439" alt="image" src="https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/45222f10-2e61-45d1-866b-5afc807e7b00">

# 24/04/14 Updates
**Thanks [@madtank](https://github.com/madtank) for adding PDF/CSV/PY file upload feature!** 🎉🎉🎉

<img width="1439" alt="image" src="https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/080daef6-6fbf-413a-b08b-a6d75a665f66">

# 24/04/05 Updates
**1. Add Dockfile for container enviroment and remove the packages installation!** 🎉🎉🎉

You could build your own, and I've also uploaded a public container image at public.ecr.aws for you~
```
docker run -d -v $HOME/.aws/config:/root/.aws/config:ro -p 8502:8501 public.ecr.aws/shtian/bedrock-claude-3-langchain-streamlit:latest
```

**2. NEW! Mistral Large on Bedrock Supported!** 🎉🎉🎉
<img width="1440" alt="image" src="https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/cf008fa9-2dd2-4aef-829a-c8a3473c0555">

# 24/03/14 Updates
**NEW! Claude 3 Haiku on Bedrock Supported! Let's Go Faster!** 🎉🎉🎉
<img width="1440" alt="image" src="https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/0815f758-2d03-4763-8df8-a331cef83f50">

Install via the command:
```
pip install -r requirements.txt
```

# 24/03/11 Updates
**NEW! Claude 3 Sonnet on Bedrock Supported! New Message API Plus Vision Multimodal Chat!** 🎉🎉🎉

Add system prompt option.

<img width="1440" alt="image" src="https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/fe6c7aea-1275-48ee-b05f-3a4e3c9f7325">


# 24/03/10 Updates
**NEW! Claude 3 Sonnet on Bedrock Supported! New Message API Plus Vision Multimodal Chat!** 🎉🎉🎉

Install langchain from source, for new Bedrock API support.

> Note: No need to hack in bedrock code! Just change the langchain_messages state of streamlit in the app code. Complete this code with the help of Claude itself :)
```
git clone https://github.com/langchain-ai/langchain.git
pip install -e langchain/libs/langchain
```

Then run the command:
```
streamlit run bedrock_chatbot_claude_3_sonnet_vision.py
```
Bingo!
<img width="1440" alt="image" src="https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/1f380628-1353-4fa4-8459-a713c6a03454">

> Note: Some details like - smooth history catchup with new message api, support mulitple images in one chat, image keep in the thumbnail in one line, multimodal and text-only mixed chat, no some bump up after rerun and re-initialize, fix lots of format mismatch...

# 24/03/09 Updates
**NEW! Claude 3 Sonnet on Bedrock Supported~ Message API Plus Vision Multimodal!** 🎉🎉🎉

Extra action needed (till now) - install langchain from source.

> Note: A little bit [hack](https://github.com/langchain-ai/langchain/commit/a0d1614159b268f218e470c803d5288a60fa8ab2) for streamlit conversation history format mismatch, and modify langchain community bedrock source code, no impact on BedrockChat invoke ~ 
```
git clone https://github.com/davidshtian/langchain.git
pip install -e langchain/libs/langchain
```

Then run the command:
```
streamlit run bedrock_chatbot_claude_3_sonnet_vision.py
```
Bingo!
<img width="1440" alt="image" src="https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/577fc5e8-e0f2-46a7-b7fd-3ec33b918ff6">


# 24/03/08 Updates
**NEW! Claude 3 Sonnet on Bedrock Supported~ Message API**

Extra action needed (till now) - install langchain from source:
```
git clone https://github.com/langchain-ai/langchain.git
pip install -e langchain/libs/langchain
```

> Note: Only text supported now, vision later!

Then run the command:
```
streamlit run bedrock_chatbot_claude_3_sonnet.py
```

The bot is equipped with chat history using [ConversationBufferWindowMemory](https://python.langchain.com/docs/modules/memory/types/buffer_window) and [StreamlitChatMessageHistory](https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history), and provided with both simple(batch) and streaming modes. Demo shown as below:

https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/9bbcf241-134e-4b8b-8080-a9000029997b

Streaming mode demo shown as below:

https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/cdff3813-fbfc-4b2f-83bc-b4016c921265
