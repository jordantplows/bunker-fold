"""Data models for Bunker API."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiError(BaseModel):
    """API error response."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""

    success: bool
    data: Any = None
    error: ApiError | None = None
