from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    """
    Data model for API query requests.
    - documents: URL to the document (PDF, DOCX, Email).
    - questions: A list of user questions (natural language).
    """
    documents: str = Field(..., description="URL to the document (PDF/DOCX/Email)")
    questions: List[str] = Field(..., description="Natural language questions")


class Clause(BaseModel):
    """
    Data model representing a clause or text chunk from the document
    that matches the query, with a human-readable explanation.
    """
    text: str = Field(..., description="Text of the matched clause or chunk")
    explanation: Optional[str] = Field(None, description="Why this clause is relevant (rationale)")


class Answer(BaseModel):
    """
    Final answer for a query, including supporting clauses and rationale.
    """
    answer: str = Field(..., description="Direct answer to the question")
    clauses: List[Clause] = Field(..., description="List of supporting clauses")
    decision_rationale: str = Field(..., description="Explanation of how answer was reached")


class QueryResponse(BaseModel):
    """
    Output model for API: structured answers to all user questions.
    """
    answers: List[Answer] = Field(..., description="List of answers, one per question")
