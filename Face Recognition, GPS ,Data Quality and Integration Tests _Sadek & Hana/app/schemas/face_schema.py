
from pydantic import BaseModel
from typing import Optional


class FaceRecognitionResult(BaseModel):
    person_id: Optional[str] = None
    similarity: Optional[float] = None
    status: str
    detection_score: Optional[float] = None
    faces_detected: int = 0


class FaceRecognitionResponse(BaseModel):
    success: bool
    filename: str
    result: FaceRecognitionResult
