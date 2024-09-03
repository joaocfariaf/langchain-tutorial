from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="LoneStriker/Starling-LM-7B-beta-GGUF",
    temperature=0.1
)


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

initialMessages = [
    SystemMessage(content="You are a very polite assistant. Answer with the best maners."),
    HumanMessage(content="Hi, introduce me yourself in 10 words at most."),
]

def printAI(message: str):
    print(f'\n    AI: {message}')
    print(f'\n HUMAN: ', end='')

# Starts conversation
printAI(model.invoke(initialMessages).content)

while True:
    humanMessage = input()
    printAI(model.invoke(humanMessage).content)
    # Streams response 
    #for chunk in model.stream(humanMessage):
    #    print(chunk.content, end='')
    continue


