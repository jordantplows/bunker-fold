"""Resource limits for public API requests and model loading."""

import math
import os

MAX_REQUEST_BYTES = 64 * 1024
MAX_SEQUENCE_LENGTH = 1024
MAX_BATCH_SIZE = 4
MAX_TOTAL_RESIDUES = 2048
MAX_CACHED_MODELS = 1
DEFAULT_MAX_MODEL_MEMORY_GB = 24.0


def max_model_memory_gb() -> float:
    """Return the positive worker model budget in GB."""
    value = float(
        os.environ.get("BUNKER_MAX_MODEL_MEMORY_GB", DEFAULT_MAX_MODEL_MEMORY_GB)
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError("BUNKER_MAX_MODEL_MEMORY_GB must be a positive number")
    return value
