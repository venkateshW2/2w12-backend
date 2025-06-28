"""
Pydantic models for streaming API
"""

from pydantic import BaseModel
from typing import Optional

class StreamRequest(BaseModel):
    file_id: str
    
class ExtendSessionRequest(BaseModel):
    file_id: str
    extend_minutes: int = 30
    
class ExtendSessionResponse(BaseModel):
    file_id: str
    new_expires_at: str
    extended_by_minutes: int
    success: bool
    message: str
