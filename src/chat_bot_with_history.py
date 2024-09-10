from typing import Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from chat_bot import chatBot, DEFAULT_MODEL
from src.sessions_manager import sessionsManager

DEFAULT_INITIAL_SYS_MSG = (
    "You are a very polite assistant. Answer with the best maners."
    "Also, you are a brazillian portuguese native speaker." 
    "So, you should respond in brazilian portuguese."
    "Also, you should try to be the most sucint you can."
    "Your name is \'FAB-GPT\'."
)

class chatBotWithHistory(chatBot):
    def __init__(
        self, 
        sessions_history: Optional[sessionsManager] = None,
        user_name: Optional[str] = 'somebody', 
        session_number: Optional[int] = None,
        model: Optional[ChatOpenAI] = DEFAULT_MODEL,
    ) -> None:
    
        if sessions_history == None:
            Warning('No sessions history has been passed. Creating a new one.')
            sessions_history = sessionsManager()
        self.sessions_history = sessions_history

        self._set_model(model)
        self._set_config(user_name, session_number)
        
    def _set_config(
        self,
        user_name: Optional[str] = 'somebody', 
        session_number: Optional[int] = None
    ):
        
        if session_number == None:
            session_id = self.sessions_history.create_session(user_name)
        else:
            session_id = f'{user_name}-{session_number}'

        self.config = {
            "configurable":{
                "session_id":session_id
        }}

    def _set_model(self, model):
        self.model = RunnableWithMessageHistory(
            model, 
            self.sessions_history.get_session_history,
            #input_messages_key="input",
            #history_messages_key="history",
        )

    def _formattedStreamAI(
        self, 
        humanInput: str | list[BaseMessage], 
        config: Dict
    ) -> None:
            
        print(f'\n    AI: ', end='')
        for chunk in self.model.stream(
            {"input": humanInput},
            config=config
        ):
            print(chunk.content, end='')
        self._formatted_human_input()

    def get_config(self):
        try:
            return self.config
        except:
            KeyError('No config defined.')
    
    def send_default_initial_messages(self, hidden=True):
        initialMessages = [
            SystemMessage(
                content=DEFAULT_INITIAL_SYS_MSG
            ),
            HumanMessage(
                content=
                """
                Olá, apresente-se com no máximo 20 palavras.
                Comece com uma saudação ao usuário, por exemplo: \'Olá Sr(a)., espero que esteja bem!\'.
                """
            ),
        ]
        
        if hidden == True:
            self.model.invoke(
                initialMessages,
                config=self.config
            )
        else:
            self._formattedStreamAI(
                humanInput= initialMessages,
                config=self.config
            )   

    def chat_in_terminal(
        self,
    ) -> None:
        while True:
            self._formattedStreamAI(
                humanInput= input(),
                config=self.config
            )   


if __name__ == "__main__":
    chat = chatBotWithHistory()
    chat.send_default_initial_messages(hidden=False)
    chat.chat_in_terminal()

