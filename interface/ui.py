import requests
import streamlit as st

import sys
import os

# Add the root directory of the project to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


from helper import file_helper, get_env

CHATBOT_URL = os.getenv(
    "CHATBOT_URL", "http://localhost:8000/chat"
)

AGENT_LIST_PATH = get_env.retreive_value( "AGENT_LIST_PATH")
AVAILABLE_AGENT = file_helper.read_json(AGENT_LIST_PATH).keys()


headers = {'api-key': '49062504109489e78769514b702a6669f5ff9c6aacbbc96f'}

with st.sidebar:
    st.header("About")
    st.markdown(
        """
       GenAssist
        """
    )

    st.header("Example Questions")
    st.markdown("- I have had a double mastectomy, should I still be having screening? (Gene: BRCA2)")
    st.markdown("- I am a man and have not had screening, is there any imaging I should be doing? (GENE: BRCA1)")
    st.markdown("- I haven’t started screening yet. How do I access my first imaging? (Age: 25, Gene: PALB2)")
    st.markdown("- I worry I am not doing mammograms frequently enough, should I be doing them more often? (All genes)")
    st.markdown("- What are the menopause symptoms after I get my ovaries out? (Gene: BRCA1/BRC2)")
    st.markdown("- I have had my ovaries out and am having horrible menopause symptoms. Who can I talk to about this?")
    st.markdown("- I am a man with BRCA1 and have not had screening, is there any imaging I should be doing?")

st.title("GenAssist")
st.info("""Ask me questions if you are a BRCA1/BRCA2/PALB2 gene carrier""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

col1, col2, col3 = st.columns(3)
with col1:
    age_input = st.text_input("Age", "")
with col2:
    gene_input = st.selectbox("Please select a Gene", ('BRCA1', 'BRCA2', 'PALB2'))
with col3:
    llm_input = st.selectbox("LLM", (AVAILABLE_AGENT)) 
    
option = st.selectbox('Please select a category for your question',('Deciding how to best manage your cancer risk',
'Accessing cancer risk management services', 'Worry/fear about developing cancer, Concern/fear about surgery',
'Impact of genetic risk on relationships', 'Connect with other women with a gene alteration',
'Worry about children/family members', 'Starting a family/ having more children', 'Menopausal symptom', 'Quitting smoke',
        'Dealing with the after effects of developing cancer'))

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {
                "age": age_input,
                "gene_fault": gene_input,
                "category": option,
                "llm":llm_input,
                "patient_question": prompt
            }

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, headers=headers, auth=None, json=data)

        if response.status_code == 200:
            output_text = response.json()["output"]
            explanation = response.json()["intermediate_steps"]
            elapsed_time = response.json().get("elapsed_time", "N/A")

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text
            elapsed_time = "N/A"
            
    # Display elapsed time aligned to the right
    st.markdown(
        f"""
        <div style="text-align: right; font-size: 0.9em; color: gray;">
            {elapsed_time}s.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.chat_message("assistant").markdown(output_text)
  
    st.status("How was this generated?", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
            "elapsed_time": elapsed_time,
        }
    )
