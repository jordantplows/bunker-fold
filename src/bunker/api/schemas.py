"""OpenAPI-compatible schemas for biological models."""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ============================================================================
# OpenAI-compatible schemas
# ============================================================================


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(
        default=0, description="Number of tokens in the completion"
    )
    total_tokens: int = Field(..., description="Total tokens used")


class EmbeddingData(BaseModel):
    """Single embedding result."""

    object: Literal["embedding"] = "embedding"
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index in the batch")


class EmbeddingRequest(BaseModel):
    """Request for embedding generation (OpenAI-compatible)."""

    model: str = Field(
        ..., description="Model ID (e.g., 'esm2_t33_650M', 'esm2_t36_3B')"
    )
    input: Union[str, List[str]] = Field(
        ..., description="Protein sequence(s) to embed"
    )
    encoding_format: Literal["float", "base64"] = Field(
        default="float", description="Format of returned embeddings"
    )
    user: Optional[str] = Field(default=None, description="User identifier")


class EmbeddingResponse(BaseModel):
    """Response for embedding generation (OpenAI-compatible)."""

    object: Literal["list"] = "list"
    data: List[EmbeddingData] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used")
    usage: Usage = Field(..., description="Token usage")


# ============================================================================
# Structure prediction schemas
# ============================================================================


class StructurePredictionRequest(BaseModel):
    """Request for structure prediction."""

    model: str = Field(..., description="Model ID (e.g., 'esmfold', 'boltz_1')")
    prompt: Union[str, List[str]] = Field(
        ..., description="Protein sequence(s) to fold"
    )
    temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum structure tokens (if applicable)"
    )
    user: Optional[str] = Field(default=None, description="User identifier")


class StructureData(BaseModel):
    """Structure prediction result."""

    pdb: str = Field(..., description="PDB format structure")
    plddt: Optional[List[float]] = Field(
        default=None, description="Per-residue confidence scores (pLDDT)"
    )
    mean_plddt: Optional[float] = Field(
        default=None, description="Mean confidence score"
    )


class StructurePredictionResponse(BaseModel):
    """Response for structure prediction."""

    object: Literal["structure"] = "structure"
    model: str = Field(..., description="Model used")
    data: List[StructureData] = Field(..., description="Predicted structures")
    usage: Usage = Field(..., description="Token usage")


# ============================================================================
# Model listing schemas
# ============================================================================


class ModelPermission(BaseModel):
    """Model permission information."""

    id: str
    object: Literal["model_permission"] = "model_permission"
    created: int
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = False
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    """Model information card (OpenAI-compatible)."""

    id: str = Field(..., description="Model identifier")
    object: Literal["model"] = "model"
    created: int = Field(..., description="Unix timestamp of model creation")
    owned_by: str = Field(..., description="Organization that owns the model")
    permission: List[ModelPermission] = Field(
        default_factory=list, description="Model permissions"
    )
    root: str = Field(..., description="Root model identifier")
    parent: Optional[str] = Field(default=None, description="Parent model")


class ModelListResponse(BaseModel):
    """List of available models (OpenAI-compatible)."""

    object: Literal["list"] = "list"
    data: List[ModelCard] = Field(..., description="List of models")


# ============================================================================
# Error schemas
# ============================================================================


class ErrorDetail(BaseModel):
    """Error detail information."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: Optional[str] = Field(default=None, description="Parameter that caused error")
    code: Optional[str] = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response (OpenAI-compatible)."""

    error: ErrorDetail = Field(..., description="Error details")
