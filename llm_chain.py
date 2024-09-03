from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="LoneStriker/Starling-LM-7B-beta-GGUF",
    temperature=0.1
)

message = [
    ('system', "Use 10 words at most to answer."),
    ('user', "How can langsmith help with testing.")
]
# Retorno em streaming
#for chunk in llm.stream(message):
#    print(chunk.content, end='')

# Structured Output
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
# Parece não funcionar muito bem, ao menos com o LM STUDIO  
class Joke(BaseModel):
    '''Joke to tell user'''
    setup: str = Field(description='The setup of a joke.')
    punchline: str = Field(description='The punchline to the joke')
    rating: Optional[int] = Field(description='How funny the joke is, from 1 to 10.')

#structured_llm = llm.with_structured_output(Joke)
#print(type(structured_llm.invoke('Tell me a joke about cats.')))

# Templates
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

chain = prompt | llm

user_input = "Answer with 10 words at most: how can langsmith help with testing."
#print(chain.invoke(user_input))


# Converter a resposta para string automaticamente
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
#print(chain.invoke(user_input))



# RAG
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()


# Este é um recurso que não necessariamente é provido por todos os modelos de LLM
# A openAi fornece isso sim. Porém, tem que verificar se tem isso implementado em 
# determinados modelos postos no LM STUDIO.
# Dá erro no LM-STUDIO pelo fato de os textos serem tokenizados antes de serem 
# enviados para o modelo. Isso causa erro pelo fato de ser mandad ums lista de itn
# sendo que o modelo espera uma str ou uma lista de str
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    model="nomic-embed-text-v1.5-GGUF",
)

str_array = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
]

embeddings = embeddings_model.embed_documents(str_array)
print(len(embeddings), len(embeddings[0]))


embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print(embedded_query[:5])