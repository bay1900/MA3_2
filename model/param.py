from pydantic import BaseModel

class ChatInput ( BaseModel ): 
    age: int
    gene_fault: str
    llm: str
    category: str
    patient_question: str
    
