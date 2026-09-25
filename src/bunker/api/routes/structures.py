"""Structure prediction endpoints."""

from fastapi import APIRouter, HTTPException, Request, status

from bunker.api.model_cache import ModelTooLargeError, get_or_load_model
from bunker.api.schemas import (
    StructureData,
    StructurePredictionRequest,
    StructurePredictionResponse,
    Usage,
)

router = APIRouter()


def estimate_tokens(text: str) -> int:
    """Estimate token count for a protein sequence."""
    return len(text)


@router.post("/completions", response_model=StructurePredictionResponse)
async def predict_structures(request: Request, body: StructurePredictionRequest):
    """Predict protein structures from sequences.

    This endpoint accepts protein sequences and returns predicted 3D structures
    in PDB format with confidence scores using models like ESMFold or Boltz-1.

    Args:
        body: Structure prediction request with model and sequences

    Returns:
        Structure prediction response with PDB files and confidence scores

    Example:
        ```json
        {
            "model": "esmfold",
            "prompt": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",
            "temperature": 1.0
        }
        ```
    """
    # Normalize input to list
    inputs = [body.prompt] if isinstance(body.prompt, str) else body.prompt

    # Validate model is for structure prediction
    from bunker.registry import get_model_info

    try:
        meta = get_model_info(body.model)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    if meta.task != "structure":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{body.model}' is not a structure prediction model "
                f"(task: {meta.task})"
            ),
        )

    # Load model (from cache or fresh)
    model_cache = request.app.state.model_cache
    try:
        model = get_or_load_model(model_cache, body.model)
    except ModelTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )

    # Predict structures
    try:
        structures = model.predict(inputs, temperature=body.temperature)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Structure prediction failed: {str(e)}",
        )

    # Format response
    structure_data = [
        StructureData(
            pdb=struct["pdb"],
            plddt=struct.get("plddt"),
            mean_plddt=struct.get("mean_plddt"),
        )
        for struct in structures
    ]

    # Calculate token usage
    total_tokens = sum(estimate_tokens(inp) for inp in inputs)

    return StructurePredictionResponse(
        object="structure",
        model=body.model,
        data=structure_data,
        usage=Usage(
            prompt_tokens=total_tokens,
            completion_tokens=total_tokens,  # Approximate
            total_tokens=total_tokens * 2,
        ),
    )
