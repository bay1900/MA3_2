import re
import json

from typing import List, Dict
from langchain_community.document_loaders import TextLoader
from helper import get_env, file_helper
from langchain.schema import Document 

from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings

from langchain_community.vectorstores import Chroma


def chunk_actionplan_by_topic_gene(text: str) -> List[Dict]: 
    chunks = []
    current_topic = None
    current_gene = None
    chunk_id = 1
    metadata= []

    # Split the text into sections by detecting "Topic:" or "Gene:" headers
    sections = re.split(r"(?=Topic:|Gene:)", text)

    for section in sections:
        # Extract topic if found
        topic_match = re.search(r"Topic:\s*(.*)", section)
        if topic_match:
            current_topic = topic_match.group(1).strip()

        # Extract gene if found
        gene_match = re.search(r"Gene:\s*(.*)", section)
        if gene_match:
            current_gene = gene_match.group(1).strip()

        # Find all age-based blocks in this section
        age_blocks = re.findall(
            r"(Min Age: \d+\nMax Age: \d+\n(?:.|\n)*?)(?=\nMin Age: \d+|(?=Gene:|Topic:|$))",
            section
        )

        for block in age_blocks:
            min_age_match = re.search(r"Min Age: (\d+)", block)
            max_age_match = re.search(r"Max Age: (\d+)", block)
            headings = re.findall(r"Heading:\s*(.*)", block)

            metadata.append({
                "id": f"{chunk_id}",
                "topic": current_topic if current_topic else "Unknown",
                "gene": current_gene if current_gene else "Unknown",
                "min_age": int(min_age_match.group(1)) if min_age_match else None,
                "max_age": int(max_age_match.group(1)) if max_age_match else None,
                # "headings": [h.strip() for h in headings],
            })
            chunks.append({  "text": block.strip() })
            chunk_id += 1

    return [chunks, metadata]

def load_document(): 
    KNOWLEDGE_BASE_PATH = get_env.retreive_value("KNOWLEDGE_BASE_PATH") # PRO 
    loader = TextLoader(file_path= KNOWLEDGE_BASE_PATH, encoding="utf-8")
    text = loader.load()

    chunk_metadata = chunk_actionplan_by_topic_gene( text[0].page_content )
    chunks   = chunk_metadata[0]
    metadata = chunk_metadata[1]


    document = [
        Document( page_content=text["text"], metadata= meta)
    for text, meta in zip(chunks, metadata)
    ]
    
    return document

async def create_vector_deepseek():
    
    MISTRAL_API_KEY = get_env.retreive_value( "MISTRAL_API_KEY")
    try:
        # Load documents
        document = load_document()

        # Load agent configuration
        agent_list = file_helper.read_json("./utils/agent_list.json")
        agent_property = agent_list.get("deepseek")

        # Retrieve credentials and config
        api_key = get_env.retreive_value(agent_property.get("env_api_path"))
        vector_store_path = get_env.retreive_value(agent_property.get("vector_store"))

        # Initialize embeddings and vector store
        embedding = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)

        vectordb = Chroma.from_documents(
            documents=document,
            embedding=embedding,
            persist_directory=vector_store_path 
        )

        return {
            "status": "success",
            "model": "deepseek",
            "vector_store": vector_store_path
        }

    except Exception as e:
        return {
            "status": "error",
            "model": "deepseek",
            "vector_store": agent_property.get("vector_store", "unknown"),
            "error": str(e)
        }

async def create_vector_mistral():
    try:
        # Load documents
        document = load_document()

        # Load agent configuration
        agent_list = file_helper.read_json("./utils/agent_list.json")
        agent_property = agent_list.get("mistral")

        # Retrieve credentials and config
        api_key = get_env.retreive_value(agent_property.get("env_api_path"))
        vector_store_path = get_env.retreive_value(agent_property.get("vector_store"))

        # Initialize embeddings and vector store
        embedding = MistralAIEmbeddings(mistral_api_key=api_key)

        vectordb = Chroma.from_documents(
            documents=document,
            embedding=embedding,
            persist_directory=vector_store_path
        )

        return {
            "status": "success",
            "model": "mistral",
            "vector_store": vector_store_path
        }

    except Exception as e:
        return {
            "status": "error",
            "model": "mistral",
            "vector_store": agent_property.get("vector_store", "unknown"),
            "error": str(e)
        }

async def create_vector_gemini():
    try:
        # Load documents
        document = load_document()

        # Load agent configuration
        agent_list = file_helper.read_json("./utils/agent_list.json")
        agent_property = agent_list.get("gemini")

        # Retrieve credentials and config
        api_key = get_env.retreive_value(agent_property.get("env_api_path"))
        vector_store_path = get_env.retreive_value(agent_property.get("vector_store"))

        # Initialize embeddings and vector store
        embedding = MistralAIEmbeddings(mistral_api_key=api_key)

        vectordb = Chroma.from_documents(
            documents=document,
            embedding=embedding,
            persist_directory=vector_store_path
        )

        return {
            "status": "success",
            "model": "mistral",
            "vector_store": vector_store_path
        }

    except Exception as e:
        return {
            "status": "error",
            "model": "mistral",
            "vector_store": agent_property.get("vector_store", "unknown"),
            "error": str(e)
        }
async def create_vector_openai():
    try:
        # Load documents
        document = load_document() 

        # Load agent configuration
        agent_list = file_helper.read_json("./utils/agent_list.json")
        agent_property = agent_list.get("chatgpt")

        # Retrieve credentials and config
        api_key = get_env.retreive_value(agent_property.get("env_api_path"))
        vector_store_path = get_env.retreive_value(agent_property.get("vector_store"))

        # Initialize embeddings and vector store
        embedding = MistralAIEmbeddings(mistral_api_key=api_key)

        vectordb = Chroma.from_documents(
            documents=document,
            embedding=embedding,
            persist_directory=vector_store_path
        )

        return {
            "status": "success",
            "model": "openai",
            "vector_store": vector_store_path
        }

    except Exception as e:
        return {
            "status": "error",
            "model": "openai",
            "vector_store": agent_property.get("vector_store", "unknown"),
            "error": str(e)
        }