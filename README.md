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

# Bedrock ChatBot with LangChain and Streamlit 💬 
A simple and clear example for implement a chatbot with Bedrock(Claude) + LangChain + Streamlit. Just `cd` to the corresponding folder and run the code:
```
streamlit run bedrock_chatbot.py
```

The bot is equipped with chat history using [ConversationBufferWindowMemory](https://python.langchain.com/docs/modules/memory/types/buffer_window) and [StreamlitChatMessageHistory](https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history), and provided with both simple(batch) and streaming modes. Demo shown as below:

https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/9bbcf241-134e-4b8b-8080-a9000029997b

Streaming mode demo shown as below:

https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit/assets/14228056/cdff3813-fbfc-4b2f-83bc-b4016c921265
