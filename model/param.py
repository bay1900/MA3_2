from pydantic import BaseModel

class ChatInput ( BaseModel ): 
    user_id: str
    age: int
    gene_fault: str
    llm: str
    category: str
    patient_question: str
    
