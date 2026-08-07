"""
Pydantic models used to validate API inputs and shape API responses.
"""

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from utils.config import VOTER_ID_PATTERN

_VOTER_ID_RE = re.compile(r"^[A-Z]{2,4}[0-9]{5,}$")


class VoterIdField(BaseModel):
    """Standalone validator used by the /add endpoint's form field."""

    voter_id: str = Field(..., min_length=6, max_length=20)

    @field_validator("voter_id")
    @classmethod
    def validate_format(cls, value: str) -> str:
        value = value.strip().upper()
        if not _VOTER_ID_RE.match(value):
            raise ValueError(
                "voter_id must be 2-4 uppercase letters followed by 5+ digits, e.g. XYZ123456"
            )
        return value


class FaceMatch(BaseModel):
    """A single candidate match against the known dataset."""

    filename: str
    similarity: float = Field(..., ge=0, le=100)
    voter_id: Optional[str] = ""


class FaceResult(BaseModel):
    """One detected face plus whatever matches it produced."""

    input_image: str = Field(..., description="Relative path/URL of the cropped face image")
    voter_id: Optional[str] = Field("", description="Voter ID read from the source document, if any")
    matches: List[FaceMatch] = []


class MatchResponse(BaseModel):
    results: List[FaceResult] = []
    warnings: List[str] = []


class AddEntryResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    voter_id: Optional[str] = None


class UploadPdfResponse(BaseModel):
    results: List[FaceResult] = []
    new_entries: int = 0
    message: str
