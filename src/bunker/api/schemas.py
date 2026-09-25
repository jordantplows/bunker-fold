"""OpenAPI-compatible schemas for biological models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from bunker.api.limits import (
    MAX_BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
    MAX_TOTAL_RESIDUES,
)


def validate_sequences(value: str | list[str]) -> str | list[str]:
    """Bound the work a single inference request can trigger."""
    sequences = [value] if isinstance(value, str) else value
    if not sequences or len(sequences) > MAX_BATCH_SIZE:
        raise ValueError(f"Provide between 1 and {MAX_BATCH_SIZE} sequences")
    if any(
        not sequence or len(sequence) > MAX_SEQUENCE_LENGTH for sequence in sequences
    ):
        raise ValueError(f"Each sequence must contain 1-{MAX_SEQUENCE_LENGTH} residues")
    if sum(map(len, sequences)) > MAX_TOTAL_RESIDUES:
        raise ValueError(f"A request may contain at most {MAX_TOTAL_RESIDUES} residues")
    return value


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
    embedding: list[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index in the batch")


class EmbeddingRequest(BaseModel):
    """Request for embedding generation (OpenAI-compatible)."""

    model: str = Field(
        ..., description="Model ID (e.g., 'esm2_t33_650M', 'esm2_t36_3B')"
    )
    input: str | list[str] = Field(..., description="Protein sequence(s) to embed")
    encoding_format: Literal["float", "base64"] = Field(
        default="float", description="Format of returned embeddings"
    )
    user: str | None = Field(default=None, description="User identifier")

    @field_validator("input")
    @classmethod
    def check_input(cls, value: str | list[str]) -> str | list[str]:
        return validate_sequences(value)


class EmbeddingResponse(BaseModel):
    """Response for embedding generation (OpenAI-compatible)."""

    object: Literal["list"] = "list"
    data: list[EmbeddingData] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used")
    usage: Usage = Field(..., description="Token usage")


# ============================================================================
# Structure prediction schemas
# ============================================================================


class StructurePredictionRequest(BaseModel):
    """Request for structure prediction."""

    model: str = Field(..., description="Model ID (e.g., 'esmfold', 'boltz_1')")
    prompt: str | list[str] = Field(..., description="Protein sequence(s) to fold")
    temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum structure tokens (if applicable)"
    )
    user: str | None = Field(default=None, description="User identifier")

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, value: str | list[str]) -> str | list[str]:
        return validate_sequences(value)


class StructureData(BaseModel):
    """Structure prediction result."""

    pdb: str = Field(..., description="PDB format structure")
    plddt: list[float] | None = Field(
        default=None, description="Per-residue confidence scores (pLDDT)"
    )
    mean_plddt: float | None = Field(default=None, description="Mean confidence score")


class StructurePredictionResponse(BaseModel):
    """Response for structure prediction."""

    object: Literal["structure"] = "structure"
    model: str = Field(..., description="Model used")
    data: list[StructureData] = Field(..., description="Predicted structures")
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
    group: str | None = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    """Model information card (OpenAI-compatible)."""

    id: str = Field(..., description="Model identifier")
    object: Literal["model"] = "model"
    created: int = Field(..., description="Unix timestamp of model creation")
    owned_by: str = Field(..., description="Organization that owns the model")
    permission: list[ModelPermission] = Field(
        default_factory=list, description="Model permissions"
    )
    root: str = Field(..., description="Root model identifier")
    parent: str | None = Field(default=None, description="Parent model")


class ModelListResponse(BaseModel):
    """List of available models (OpenAI-compatible)."""

    object: Literal["list"] = "list"
    data: list[ModelCard] = Field(..., description="List of models")


# ============================================================================
# Error schemas
# ============================================================================


class ErrorDetail(BaseModel):
    """Error detail information."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: str | None = Field(default=None, description="Parameter that caused error")
    code: str | None = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response (OpenAI-compatible)."""

    error: ErrorDetail = Field(..., description="Error details")
