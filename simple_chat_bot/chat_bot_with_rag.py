
from chat_bot import chatBot

from typing import Dict


# RAG
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough


class chatBotWithRAG(chatBot):
    def _read_web_page(self, config):

        url = input(f'\n HUMAN: ')

        
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

    def _read_from_pdf_docs(self):
        pass



