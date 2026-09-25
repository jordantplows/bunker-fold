"""Bounded cache for expensive models used by API workers."""

import gc
from collections.abc import MutableMapping
from typing import Any

from bunker.api.limits import MAX_CACHED_MODELS, max_model_memory_gb
from bunker.loader import load
from bunker.registry import get_model_info


class ModelTooLargeError(ValueError):
    """Raised when a model exceeds the configured worker memory budget."""


def get_or_load_model(cache: MutableMapping[str, Any], name: str) -> Any:
    """Load at most one bounded-size model per worker."""
    if name in cache:
        return cache[name]

    metadata = get_model_info(name)
    memory_limit = max_model_memory_gb()
    if metadata.memory_gb and metadata.memory_gb > memory_limit:
        raise ModelTooLargeError(
            f"Model '{name}' requires about {metadata.memory_gb:g} GB; "
            f"the API limit is {memory_limit:g} GB"
        )

    while len(cache) >= MAX_CACHED_MODELS:
        oldest = next(iter(cache))
        del cache[oldest]
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    model = load(name, device="auto")
    cache[name] = model
    return model
