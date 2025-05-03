
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
