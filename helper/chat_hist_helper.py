from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from helper import embedding_helper

from datetime import datetime
import uuid

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_PATH = os.path.join(SCRIPT_DIR, "../chat_memory")


# --------------------
# Store user question and LLM response in vector store
# --------------------
def store_interaction(user_id, query, response):
    
    

    llm = "mistral"
    embedding_model = embedding_helper.embedding_selector(llm)
    chroma_store = Chroma(embedding_function=embedding_model, persist_directory="vector_store/chat_hist")

    document = f"User: {query}\nLLM: {response}"

    try:
        chroma_store.add_texts(
            texts=[document],
            metadatas=[{"user_id": user_id, "timestamp": str(datetime.now())}],
            ids=[str(uuid.uuid4())]
        )
        print("user question and llm response stored successfully")        
    except Exception as e:
        print(f"user question and llm response: {e}")

def load_user_memory(user_id):
    path = _get_user_memory_path(user_id)
    if path:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            return []
        except FileNotFoundError:
            print(f"Chat history file not found for user {user_id}.")
            return []
        except json.JSONDecodeError:
            print(f"Error decoding JSON from chat history file for user {user_id}.")
            return []
        except OSError as e:
            print(f"Error accessing chat history file for user {user_id}: {e}")
            return []
    else:
        return []


def _get_user_memory_path(user_id):
    try:
        if not os.path.exists(CHAT_HISTORY_PATH):
            os.makedirs(CHAT_HISTORY_PATH)
            print(f"Created directory: {CHAT_HISTORY_PATH}")  # Informative message
        file_path = os.path.join(CHAT_HISTORY_PATH, f"{user_id}_chat.json")
        # Create the file if it doesn't exist (empty file)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass  # Creates an empty file
            print(f"Created new chat history file: {file_path}") # Informative message
        return file_path
    except OSError as e:
        print(f"Error creating directory or file: {e}")
        return None
    
def save_user_memory(user_id, new_convo_history):
    path = _get_user_memory_path(user_id)
    
    with open( path, "w", encoding="utf-8") as f:
        json.dump(new_convo_history, f, indent=2, ensure_ascii=False)