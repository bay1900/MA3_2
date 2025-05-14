
import time

from helper import get_env, file_helper, agent_get_func_helper
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain.agents import create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage


from helper import get_env, file_helper,embedding_helper, filter_helper, chat_hist_helper

import json

from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)  

# GET API KEY 
DEEPSEEK_API_KEY= get_env.retreive_value( "DEEPSEEK_API_KEY")
MISTRAL_API_KEY = get_env.retreive_value( "MISTRAL_API_KEY")
OPENAI_API_KEY  = get_env.retreive_value( "OPENAI_API_KEY")
# openai.api_key  = OPENAI_API_KEY

 
POPERTY_PATH = get_env.retreive_value( "PROPERTY_PATH")
POPERTY = file_helper.read_json( POPERTY_PATH ) 

template_str     = POPERTY.get("template_str")
tool_description = POPERTY.get("tool_description")

# Initialize the memory object
memory = ConversationBufferMemory(memory_key="chat_history")

def agent_executor( payload ):
    
    start_time = time.time()
    
   
     
    #PAYLOAD
    age        = payload.age
    gene_fault = payload.gene_fault
    category   = payload.category
    llm        = payload.llm
    patient_question = payload.patient_question
    user_id    = payload.user_id    

    
    # GET AGENT LIST
    list = file_helper.read_json( "./utils/agent_list.json" )
    agent_property= list.get( llm ) 
    
    # GET CHAT MODEL
    chat_model = agent_get_func_helper.get_chat_model(llm) 
    
    # SYSTEM
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=template_str,
        )
    )

    # HUMAN
    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}",
        )
    )
 
    # COMBIME MESSAGE
    messages = [system_prompt, human_prompt]

    # SET UP PROMPT TEMPLATE
    prompt_template = ChatPromptTemplate( input_variables=["context", "question",  "chat_history"], 
                                          messages=messages, )
    
    # GET EMBEDDING BY SELECTED LLM
    embedding_selector = embedding_helper.embedding_selector( llm )
    
    # CHROMA
    llm_vector_store_path = get_env.retreive_value( agent_property.get("vector_store") ) 
    vectordb = Chroma(
            persist_directory  = llm_vector_store_path,
            embedding_function = embedding_selector,
    )

    retriever = vectordb.as_retriever(search_type="similarity", 
                                      search_kwargs={"k": 5})
    
    
    # FILTER METADATA
    filter = filter_helper.create_filter_from_json( gene_fault, age )
        
    vector_chain = (
            {"context": retriever, 
             "question":  RunnablePassthrough(),
             "chat_history": RunnablePassthrough()} 
            | prompt_template
            | chat_model
            | StrOutputParser()
    )

    tools = [
        Tool(
            name  ="ActionPlan",
            func  =vector_chain.invoke,
            description = tool_description,
        ),
      
    ]

    # agent_prompt = hub.pull("hwchase17/openai-functions-agent") # DOWNLOAD PREBUILD PROMPT
    agent_prompt = hub.pull("hwchase17/react-chat") # Use react chat prompt that works with any llm.

    # Create Agent
    agent = create_react_agent(chat_model, tools, agent_prompt)

    context = " I am at age " + str( age ) + " and I have " + gene_fault + " gene fault. " + "My question is about " + category + "."
    question = context + " " + patient_question
    
    
    # LOAD MEMORY FROM JSON
    previous_history = chat_hist_helper.load_user_memory(user_id)
    langchain_chat_history = []
    for entry in previous_history:
        langchain_chat_history.append(HumanMessage(content=entry["human"]))
        langchain_chat_history.append(AIMessage(content=entry["ai"]))

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.chat_memory.messages = langchain_chat_history
    
    
    # Create Agent Executor
    agent_executor = AgentExecutor( agent=agent, 
                                    tools=tools, 
                                    verbose=True, 
                                    memory = memory)
    
    # response
    response = agent_executor.invoke({"input": question})
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    result = { 
               "llm"    : llm,
               "model"  : agent_property.get("model"),
               "output": response.get("output", "No answer found"),
               "explanation": response.get("explanation", "No explanation found"),
               "intermediate_steps" : response.get("intermediate_steps", "No intermediate steps found"),
               "question": question,
               "elapsed_time": f"{elapsed_time:.2f}",  # Format to 2 decimal places
            }
    
    
    # STORE INTERACTION
    chat_hist_helper.store_interaction(
                    user_id = 104829047,
                    query   = question,
                    response= response.get("output", "No answer found"),
    )
    
    
    # STORE CHAT HISTORY TO JSON
    updated_history = memory.chat_memory.messages
    new_history = []
    for i in range(0, len(updated_history), 2):
        if i + 1 < len(updated_history):
            new_history.append({
                "human": updated_history[i].content,
                "ai": updated_history[i + 1].content
            })

    chat_status = chat_hist_helper.save_user_memory(user_id, new_history)
    

    return result

