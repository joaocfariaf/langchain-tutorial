from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="LoneStriker/Starling-LM-7B-beta-GGUF",
    temperature=0.1
)

from langchain_core.messages import HumanMessage, AIMessage

message = [HumanMessage(content="Hi! I'm Bob")]
message = [
    HumanMessage(content="Hi! I'm Bob"),
    AIMessage(content="Hello Bob! How can I assist you today?"),
    HumanMessage(content="What's my name?"),
]
# retorno em stream 
for chunk in model.stream(message):
    print(chunk.content, end='')

print(
    '''
    ------------------------------------
    '''
)

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "abc2"}}

# Mensagem 1
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)
print(response.content)

# Mensagem 2
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
print(response.content)