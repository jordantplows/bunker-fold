"""Model loader with lazy imports and device management."""

import os
from pathlib import Path
from typing import Any, Literal, Optional

from bunker.registry import get_model_info


# Cache directory for model weights
def get_cache_dir() -> Path:
    """Get the cache directory for model weights.

    Returns:
        Path to cache directory (defaults to ~/.cache/bunker)
    """
    cache_dir = os.environ.get("BUNKER_CACHE", str(Path.home() / ".cache" / "bunker"))
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(device: str = "auto") -> str:
    """Determine the compute device to use.

    Args:
        device: Device string ("auto", "cuda", "mps", "cpu")

    Returns:
        Resolved device string
    """
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def load(
    name: str,
    device: str = "auto",
    dtype: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Load a biological model by name.

    Args:
        name: Model identifier (e.g., "esm2_t33_650M", "esmfold", "boltz_1")
        device: Device to load model on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("float32", "float16", "bfloat16")
        **kwargs: Additional model-specific arguments

    Returns:
        Loaded model instance

    Raises:
        ValueError: If model is not registered
        ImportError: If required dependencies are not installed

    Example:
        >>> import bunker
        >>> model = bunker.load("esm2_t33_650M", device="cuda")
        >>> embeddings = model.predict(sequence)
    """
    # Get model metadata
    metadata = get_model_info(name)

    # Resolve device
    resolved_device = get_device(device)

    # Call the lazy loader
    try:
        model = metadata.loader(
            name=name,
            device=resolved_device,
            dtype=dtype,
            cache_dir=get_cache_dir(),
            **kwargs,
        )
        return model
    except ImportError as e:
        raise ImportError(
            f"Model '{name}' requires the '{metadata.extra}' extra. "
            f"Install with: pip install 'bunker-fold[{metadata.extra}]'\n"
            f"Original error: {e}"
        ) from e
