"""Bunker: Universal inference package for biological foundation models.

Example:
    >>> import bunker
    >>> model = bunker.load("esm2_t33_650M")
    >>> embeddings = model.predict(sequence)
"""

__version__ = "0.1.0"
__author__ = "Jordan Plows"

from bunker.loader import load
from bunker.registry import list_models, get_model_info

# Register all models
from bunker import models as _models  # noqa: F401

__all__ = [
    "load",
    "list_models",
    "get_model_info",
    "__version__",
]
