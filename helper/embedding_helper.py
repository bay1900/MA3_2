import os
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings

def embedding_selector(provider: str = "mistral") :

    
    if provider == "mistral":
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        embedding_function = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
    elif provider == "deepseek":
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        embedding_function = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
    elif provider == "chatgpt":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_function = OpenAIEmbeddings(api_key=openai_api_key)
    elif provider == "gemini":
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        embedding_function = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
    elif provider == "llama":
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        embedding_function = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    return embedding_function 
