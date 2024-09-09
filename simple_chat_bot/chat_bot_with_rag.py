
from chat_bot import chatBot

# RAG
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough


class chatBotWithRAG(chatBot):
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
            self.sessions_history.get_session_history,
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
            
            #if humanInput[0] == '\\':
            #    command = humanInput[1:]
            #    input(f'self._{command}(config)')
            #    exec(f'self._{command}(config)')
            #    return

            print(f'\n    AI: ', end='')
            for chunk in self.with_message_history.stream(
                {"input": humanInput},
                config=config
            ):
                print(chunk.content, end='')
            print(f'\n\n HUMAN: ', end='')

    def chat_in_terminal(
            self, 
            user_name: Optional[str] = 'somebody', 
            session_number: Optional[int] = None
        ) -> None:

        if session_number == None:
            session_id = self.sessions_history.create_session(user_name)
        else:
            session_id = f'{user_name}-{session_number}'

        config = {
            "configurable":{
                "session_id":session_id
        }}

        self.with_message_history = RunnableWithMessageHistory(
            self.model, 
            self.sessions_history.get_session_history,
            #input_messages_key="input",
            #history_messages_key="history",
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
        
        self._formattedStreamAI(
            humanInput=initialMessages,
            config=config
        )

        while True:
            self._formattedStreamAI(
                humanInput= input(),
                config=config
            )   

    def config_chain_with_history(
            self, 
            user_name: Optional[str] = 'somebody', 
            session_number: Optional[int] = None
        ) -> None:

        if session_number == None:
            session_id = self.sessions_history.create_session(user_name)
        else:
            session_id = f'{user_name}-{session_number}'

        config = {
            "configurable":{
                "session_id":session_id
        }}

        self.config = config

        self.with_message_history = RunnableWithMessageHistory(
            self.model, 
            self.sessions_history.get_session_history,
            #input_messages_key="input",
            #history_messages_key="history",
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
        
        self.with_message_history.invoke(
            initialMessages,
            config=config
        )
    

