from langchain_openai import OpenAIEmbeddings
from src.config.settings import settings

def get_embedding_model() -> OpenAIEmbeddings:
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env before building the index."
        )

    return OpenAIEmbeddings(api_key=settings.openai_api_key)