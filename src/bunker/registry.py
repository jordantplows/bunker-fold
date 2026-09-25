"""Model registry for biological foundation models."""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    name: str
    description: str
    task: str  # "embedding", "structure", "design"
    extra: str  # Required pip extra (e.g., "esm", "boltz")
    loader: Callable[..., Any]  # Lazy loader function
    weights_url: Optional[str] = None
    license: str = "Unknown"
    paper_url: Optional[str] = None
    memory_gb: Optional[float] = None  # Estimated GPU memory


# Global registry
_REGISTRY: dict[str, ModelMetadata] = {}


def register_model(
    name: str,
    description: str,
    task: str,
    extra: str,
    loader: Callable[..., Any],
    weights_url: Optional[str] = None,
    license: str = "Unknown",
    paper_url: Optional[str] = None,
    memory_gb: Optional[float] = None,
) -> None:
    """Register a model in the global registry.

    Args:
        name: Unique model identifier
        description: Human-readable description
        task: Task type (embedding, structure, design)
        extra: Required pip extra
        loader: Lazy loader function that returns the model
        weights_url: HuggingFace Hub model ID or URL
        license: Model license
        paper_url: Link to paper
        memory_gb: Estimated GPU memory requirement
    """
    _REGISTRY[name] = ModelMetadata(
        name=name,
        description=description,
        task=task,
        extra=extra,
        loader=loader,
        weights_url=weights_url,
        license=license,
        paper_url=paper_url,
        memory_gb=memory_gb,
    )


def get_model_info(name: str) -> ModelMetadata:
    """Get metadata for a registered model.

    Args:
        name: Model identifier

    Returns:
        Model metadata

    Raises:
        ValueError: If model is not registered
    """
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found. Available models: {available}"
        )
    return _REGISTRY[name]


def list_models(task: Optional[str] = None) -> list[str]:
    """List all registered models.

    Args:
        task: Optional filter by task type

    Returns:
        List of model names
    """
    if task is None:
        return sorted(_REGISTRY.keys())
    return sorted(
        name for name, meta in _REGISTRY.items() if meta.task == task
    )


def get_all_metadata() -> dict[str, ModelMetadata]:
    """Get all model metadata.

    Returns:
        Dictionary mapping model names to metadata
    """
    return _REGISTRY.copy()
