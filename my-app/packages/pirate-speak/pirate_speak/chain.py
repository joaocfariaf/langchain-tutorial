from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Translate user input into pirate speak",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{text}"),
    ]
)
_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="LoneStriker/Starling-LM-7B-beta-GGUF",
)

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model
