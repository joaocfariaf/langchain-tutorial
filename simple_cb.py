from typing import Optional, Any, Dict
import heapq
from langchain_openai import ChatOpenAI
# Server
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn

# RAG
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

# Messages kinds
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Message History
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

DEFAULT_MODEL = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="LoneStriker/Starling-LM-7B-beta-GGUF",
)


class sessionNumberManager:
    def __init__(self) -> None:
        self.session_count = 0
        self.session_heap = []

    def delete_session(self, session_number: int):
        heapq.heappush(self.session_heap, session_number)

    def new_session_number(self):
        if self.session_heap:
            return heapq.heappop(self.session_heap)
        else:
            self.session_count += 1
            return self.session_count


class sessionManager:
    def __init__(self) -> None:
        self.id_manager = {}
        self.memo = {}

    def get_session_history(self, session_id: str)-> Any:
        try:
            return self.memo[session_id]
        except:
            raise KeyError('SESSION NOT FOUND')
    
    def delete_session(self, session_id: str):
        try:
            self.memo.pop(session_id)
            user_name, session_number = session_id.split('-')
            self.id_manager[user_name].delete_session(session_number)
        except:
            raise Warning('SESSION NOT FOUND')
     
    def create_session(
            self, 
            user_name: str
        ) -> str:
        if user_name not in self.id_manager:
            self.id_manager[user_name] = sessionNumberManager()
        session_number = self.id_manager[user_name].new_session_number()
        session_id = f'{user_name}-{session_number}'
        self.memo[session_id] = InMemoryChatMessageHistory()

        return session_id

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class chatBotManager():
    def __init__(
            self, 
            model: Optional[ChatOpenAI] = DEFAULT_MODEL,
            sessions: Optional[sessionManager] = None,
        ) -> None:
    
        self.model = model
        if sessions == None:
            sessions = sessionManager()
        self.session_manager = sessions

    def _read_web_page(self, config):
        
        # UI
        print(f'\n    AI: OK, vou ler uma página web para você. Digite a URL.')
        url = input(f'\n HUMAN: ')
        print(f'\n    AI: 1 minuto, estou lendo tudo e interpretando...', end='')
        
        # 1. Load, chunk and index the contents of the blog to create a retriever.
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=GPT4AllEmbeddings()
        )
        
        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever()
        rag_sys_prompt = (
            "Use the following pieces of retrieved context to answer the question." 
            "If you don't know the answer, just say that you don't know."
            "Use three sentences maximum and keep the answer concise."
            "Context: {context}"
        )
        
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rag_sys_prompt),
                MessagesPlaceholder(variable_name='history'),
                ("human", "{input}"),
            ]
        )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.model
        )

        self.with_message_history = RunnableWithMessageHistory(
            rag_chain, 
            self.session_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        '''                
        # Simple test
        initialMessages = [
            SystemMessage(
                content="""
                You are a very polite assistant. Answer with the best maners.
                Also, you are a brazillian portuguese native speaker. 
                So, you should respond in brazilian portuguese.
                """
            ),
        ]
        
        self.with_message_history.invoke(initialMessages, config=config)
        '''

        # UI
        print(f'OK, já terminei de ler tudo. pode me fazer perguntar sobre isso agora.')
        print(f'\n\n HUMAN: ', end='')
        pass

    def _formattedStreamAI(
            self, 
            humanInput: str | list[BaseMessage], 
            config: Dict
        ) -> None:
            
            if humanInput[0] == '\\':
                command = humanInput[1:]
                input(f'self._{command}(config)')
                exec(f'self._{command}(config)')
                return

            print(f'\n    AI: ', end='')
            for chunk in self.with_message_history.stream(
                {"input": humanInput},
                config=config
            ):
                print(chunk.content, end='')
            print(f'\n\n HUMAN: ', end='')

    def start_local_chating(
            self, 
            user_name: Optional[str] = 'somebody', 
            session_number: Optional[int] = None
        ) -> None:

        if session_number == None:
            session_id = self.session_manager.create_session(user_name)
        else:
            session_id = f'{user_name}-{session_number}'

        config = {
            "configurable":{
                "session_id":session_id
        }}

        self.with_message_history = RunnableWithMessageHistory(
            self.model, 
            self.session_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        # Simple test
        initialMessages = [
            SystemMessage(
                content="""
                You are a very polite assistant. Answer with the best maners.
                Also, you are a brazillian portuguese native speaker. 
                So, you should respond in brazilian portuguese.
                """
            ),
            HumanMessage(
                content=
                """
                Olá, apresente-se com no máximo 20 palavras.
                Comece com uma saudação ao usuário, por exemplo: \'Olá Sr(a)., espero que esteja bem!\'.
                """
            ),
        ]
        
        #self._formattedStreamAI(
        #    humanInput=initialMessages,
        #    config=config
        #)

        while True:
            self._formattedStreamAI(
                humanInput= input(),
                config=config
            )   
    
    def lang_server_chat(            
            self, 
            user_name: Optional[str] = 'somebody', 
            session_number: Optional[int] = None
        ) -> None:

        if session_number == None:
            session_id = self.session_manager.create_session(user_name)
        else:
            session_id = f'{user_name}-{session_number}'

        config = {
            "configurable":{
                "session_id":session_id
        }}

        self.with_message_history = RunnableWithMessageHistory(
            self.model, 
            self.session_manager.get_session_history,
        )
        
        # Simple test
        initialMessages = [
            SystemMessage(
                content="""
                You are a very polite assistant. Answer with the best maners.
                Also, you are a brazillian portuguese native speaker. 
                So, you should respond in brazilian portuguese.
                """
            ),
            HumanMessage(
                content=
                """
                Olá, apresente-se com no máximo 20 palavras.
                Comece com uma saudação ao usuário, por exemplo: \'Olá Sr(a)., espero que esteja bem!\'.
                """
            ),
        ]
        
        app = FastAPI(
            title="LangChain Server",
            version="1.0",
            description="A simple API server using LangChain's Runnable interfaces",
        )

        # 5. Adding chain route
        add_routes(
            app,
            self.with_message_history,
            path="/chain",
        )

        uvicorn.run(app, host="localhost", port=8000)
    
#cb = chatBotManager().start_local_chating()

if __name__ == "__main__":
    chatBotManager().start_local_chating()





























