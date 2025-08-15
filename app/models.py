from typing import Dict
from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """Response returned after processing an audio file."""

    id: str
    transcript: str
    structured_data: Dict[str, str]


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    question: str


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""

    answer: str
