
# CREATE ENVIRONMENT
python3.11 -m venv venv
source venv/bin/activate

# RUN API SERVICE
pip install -r requirements.txt  
uvicorn main:app --reload 

# RUN INTERFACE
streamlit run interface/ui.py


# ADDING NEW MODEL
add agent property :  utils > agent_list.json
add llm api key    :  .env 

# .ENV
OPENAI_API_KEY    = ""
DEEPSEEK_API_KEY  = ""
MISTRAL_API_KEY   = ""

# PATHS
USER_API_KEY_PATH    = "./utils/user_api_key.json"
PROPERTY_PATH        = "./utils/property.json"
AGENT_LIST_PATH      = "./utils/agent_list.json"
KNOWLEDGE_BASE_PATH  = "./resource/actionplan - whole.txt"
KNOWLEDGE_BASE_PATH_TEST   = "./resource/test.txt"
KNOWLEDGE_BASE_VECTOR_PATH = "vector_db" 

KNOWLEDGE_BASE_VECTOR_PATH_MISTRAL  = "vector_store/vector_db_mistral" 
KNOWLEDGE_BASE_VECTOR_PATH_OPENAI   = "vector_store/vector_db_openai" 
KNOWLEDGE_BASE_VECTOR_PATH_DEEPSEEK = "vector_store/vector_db_deepseek" 
