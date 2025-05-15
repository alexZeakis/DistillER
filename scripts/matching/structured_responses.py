from pydantic import BaseModel, Field

class AnswerSelectSchema(BaseModel):
    answer: str = Field(description="The corresponding record number surrounded by \"[]\" or \"[0]\" if there is none.")

class AnswerExplainSchema(BaseModel):
    answer: str = Field(description="The corresponding record number surrounded by \"[]\" or \"[0]\" if there is none.")
    explanation: str = Field(description="A brief explanation of the answer.")
    
class AnswerConfidenceSchema(BaseModel):
    answer: str = Field(description="The corresponding record number surrounded by \"[]\" or \"[0]\" if there is none.")
    explanation: str = Field(description="A brief explanation of the answer.")
    confidence: float = Field(description="A confidence float score between 0 and 1.")