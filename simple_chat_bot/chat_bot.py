from typing import Optional, Dict
from langchain_openai import ChatOpenAI

from langchain_core.messages import BaseMessage

DEFAULT_MODEL = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="LoneStriker/Starling-LM-7B-beta-GGUF",
)

class chatBot():
    def __init__(
        self, 
        model: Optional[ChatOpenAI] = DEFAULT_MODEL,
    ) -> None:
        self._set_model(model)

    def _set_model(self, model):
        self.model = model

    def _formattedStreamAI(
        self, 
        humanInput: str | list[BaseMessage], 
    ) -> None:
            
        print(f'\n    AI: ', end='')
        for chunk in self.model.stream(humanInput):
            print(chunk.content, end='')
        self._formatted_human_input()

    def _formatted_human_input(self):
        print(f'\n\n HUMAN: ', end='')

    def chat_in_terminal(
        self, 
    ) -> None:
        
        self._formatted_human_input()
        while True:
            self._formattedStreamAI(
                humanInput= input(),
            )

if __name__ == "__main__":
    chatBot().chat_in_terminal()





























