
from chat_bot import chatBot

# To convert PDF files to text
from pdf_text_extractor import pdfTextExtractor, DATA_FILE

# RAG
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader # WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from openai import OpenAI
#from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings




class chatBotWithRAG(chatBot):

    def _read_from_pdf_docs(self):
        # Extracts the data from the given .pdf files
        #pdfTextExtractor().extract_data()

        loader = TextLoader(
            file_path=DATA_FILE,
            encoding='utf-8',
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # GPT4ALL NAO EH LOCAL
        local_embeddings = OpenAIEmbeddings(
            check_embedding_ctx_length = False,
            openai_api_key = "lm-studio",
            openai_api_base = "http://localhost:1234/v1", 
            model = "CompendiumLabs/bge-large-en-v1.5-gguf"#"nomic-ai/nomic-embed-text-v1.5-GGUF"
        )
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=local_embeddings
        )
        retriever = vectorstore.as_retriever()

        rag_sys_prompt = (
            "Use the following pieces of retrieved context to answer the question." 
            "If you don't know the answer, just say that you don't know."
            "Use three sentences maximum and keep the answer concise."
            "Keep your aswers short, talking only about what was explicity asked for."
            "Also, always answer in brazillian portuguese."
            "\n <context> \n {context} \n </context> \n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rag_sys_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.model, prompt)
        self.model = create_retrieval_chain(retriever, question_answer_chain)
        

    def _formattedStreamAI(
        self, 
        humanInput: str, 
    ) -> None:
            
        print(f'\n    AI: ', end='')
        print(self.model.invoke({"input": humanInput})['answer'], end='')
        self._formatted_human_input()

    def chat_in_terminal(self) -> None:
        self._read_from_pdf_docs()
        super().chat_in_terminal()



if __name__ == '__main__':
    chatBotWithRAG().chat_in_terminal()