from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question about an uploaded document"
    )

    document_type: str | None = Field(
        default=None,
        description="Optional document type filter"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve"
    )

class QuestionResponse(BaseModel):
    question: str
    answer: str
