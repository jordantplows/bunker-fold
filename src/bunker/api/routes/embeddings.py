"""Embedding endpoints (OpenAI-compatible)."""

from typing import List
from fastapi import APIRouter, HTTPException, Request, status

from bunker.api.schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingData, Usage
from bunker.loader import load

router = APIRouter()


def estimate_tokens(text: str) -> int:
    """Estimate token count for a protein sequence.

    Args:
        text: Protein sequence

    Returns:
        Estimated token count (roughly length of sequence)
    """
    return len(text)


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: Request, body: EmbeddingRequest):
    """Generate embeddings for protein sequences (OpenAI-compatible).

    This endpoint accepts protein sequences and returns dense vector embeddings
    using models like ESM2.

    Args:
        body: Embedding request with model and input sequences

    Returns:
        Embedding response with vectors and usage statistics

    Example:
        ```json
        {
            "model": "esm2_t33_650M",
            "input": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
        }
        ```
    """
    # Normalize input to list
    inputs = [body.input] if isinstance(body.input, str) else body.input

    # Validate model is for embeddings
    from bunker.registry import get_model_info

    try:
        meta = get_model_info(body.model)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    if meta.task != "embedding":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{body.model}' is not an embedding model (task: {meta.task})",
        )

    # Load model (from cache or fresh)
    model_cache = request.app.state.model_cache
    if body.model not in model_cache:
        try:
            model = load(body.model, device="auto")
            model_cache[body.model] = model
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_424_FAILED_DEPENDENCY,
                detail=str(e),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}",
            )
    else:
        model = model_cache[body.model]

    # Generate embeddings
    try:
        embeddings = model.predict(inputs)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}",
        )

    # Format response
    embedding_data = [
        EmbeddingData(
            object="embedding",
            embedding=emb.tolist() if hasattr(emb, "tolist") else emb,
            index=i,
        )
        for i, emb in enumerate(embeddings)
    ]

    # Calculate token usage
    total_tokens = sum(estimate_tokens(inp) for inp in inputs)

    return EmbeddingResponse(
        object="list",
        data=embedding_data,
        model=body.model,
        usage=Usage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
        ),
    )
