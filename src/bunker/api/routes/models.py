"""Model listing endpoints (OpenAI-compatible)."""

import time
from fastapi import APIRouter, Request

from bunker.api.schemas import ModelListResponse, ModelCard, ModelPermission
from bunker.registry import get_all_metadata

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request):
    """List all available models (OpenAI-compatible).

    Returns a list of available biological models in OpenAI format.
    """
    metadata = get_all_metadata()

    model_cards = []
    for name, meta in metadata.items():
        # Create model card
        card = ModelCard(
            id=name,
            object="model",
            created=int(time.time()),  # Use current time as placeholder
            owned_by="bunker",
            root=name,
            parent=None,
            permission=[
                ModelPermission(
                    id=f"{name}-permission",
                    created=int(time.time()),
                    allow_sampling=True,
                    allow_view=True,
                )
            ],
        )
        model_cards.append(card)

    return ModelListResponse(object="list", data=model_cards)


@router.get("/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str):
    """Get information about a specific model (OpenAI-compatible).

    Args:
        model_id: Model identifier

    Returns:
        Model information card
    """
    from fastapi import HTTPException, status
    from bunker.registry import get_model_info

    # Verify model exists
    try:
        meta = get_model_info(model_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return ModelCard(
        id=model_id,
        object="model",
        created=int(time.time()),
        owned_by="bunker",
        root=model_id,
        parent=None,
        permission=[
            ModelPermission(
                id=f"{model_id}-permission",
                created=int(time.time()),
                allow_sampling=True,
                allow_view=True,
            )
        ],
    )
