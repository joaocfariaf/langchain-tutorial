from langchain_openai import ChatOpenAI

chain = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="LoneStriker/Starling-LM-7B-beta-GGUF",
    temperature=0.1
)


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

initialMessages = [
    SystemMessage(content="You are a very polite assistant. Answer with the best maners."),
    HumanMessage(content="Hi, introduce me yourself in 10 words at most. Start with a greeting to the user, like \'Hello Sir/Madam\'."),
]

def formattedStreamAI(humanInput: str | list[BaseMessage]):
    print(f'\n    AI: ', end='')
    for chunk in chain.stream(humanInput):
        print(chunk.content, end='')
    print(f'\n\n HUMAN: ', end='')


# Starts conversation
formattedStreamAI(initialMessages)

while True:
    formattedStreamAI(input())
    break

# Messagge History
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

sessions_memo = {}

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    '''
    If the session is already in the memory, returns its stored history. 
    Otherwise, stores the current session in the memo, and returns the given information.
    '''
    if session_id not in sessions_memo:
        sessions_memo[session_id] = InMemoryChatMessageHistory()
    return sessions_memo[session_id]

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {
    "configurable":{
        "session_id":"abc1"
}}


response = with_message_history.invoke(
    [initialMessages],
    config=config
)




