"""Error response models for standardized error handling."""
from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ErrorDetail(BaseModel):
    """Detailed error information."""
    message: str = Field(..., description="Human-readable error message")
    code: str = Field(..., description="Machine-readable error code")
    field: Optional[str] = Field(None, description="Field that caused the error, if applicable")


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: ErrorDetail = Field(..., description="Error details")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp of the error"
    )
    path: Optional[str] = Field(None, description="Request path that caused the error")
    request_id: Optional[str] = Field(None, description="Unique request identifier for tracing")

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "message": "Question cannot be empty",
                    "code": "VALIDATION_ERROR",
                    "field": "question"
                },
                "timestamp": "2026-07-28T14:30:00Z",
                "path": "/api/v1/chat",
                "request_id": "req_123456789"
            }
        }


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with additional field errors."""
    errors: list[ErrorDetail] = Field(
        default_factory=list,
        description="List of all validation errors"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "message": "Validation failed",
                    "code": "VALIDATION_ERROR"
                },
                "timestamp": "2026-07-28T14:30:00Z",
                "path": "/api/v1/chat",
                "request_id": "req_123456789",
                "errors": [
                    {
                        "message": "Question must be at least 3 characters",
                        "code": "MIN_LENGTH",
                        "field": "question"
                    }
                ]
            }
        }