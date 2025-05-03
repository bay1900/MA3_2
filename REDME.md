
# create environment
python3.11 -m venv venv
source venv/bin/activate

# run api
pip install -r requirements.txt  
uvicorn main:app --reload 

# run ui
streamlit run interface/ui.py


# ######### ADDING NEW MODEL #########
add agent property :  utils > agent_list.json
add llm api key    :  .env 


pip install fastapi
pip install pydantic
pip install dotenv

pip install mistralai


pip install langchain_chroma


pip install 
pip install langchain_community
pip install langchain_mistralai
pip install langchain_deepseek