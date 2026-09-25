"""Tests for model registry."""

import pytest

from bunker.registry import (
    get_model_info,
    list_models,
    get_all_metadata,
)


def test_list_models():
    """Test listing all models."""
    models = list_models()
    assert len(models) > 0
    assert "esm2_t33_650M" in models
    assert "esmfold" in models


def test_list_models_by_task():
    """Test filtering models by task."""
    embedding_models = list_models(task="embedding")
    assert all("esm2" in m for m in embedding_models)

    structure_models = list_models(task="structure")
    assert "esmfold" in structure_models
    assert "boltz_1" in structure_models


def test_get_model_info():
    """Test getting model metadata."""
    info = get_model_info("esm2_t33_650M")
    assert info.name == "esm2_t33_650M"
    assert info.task == "embedding"
    assert info.extra == "esm"
    assert info.memory_gb == 4.0


def test_get_model_info_invalid():
    """Test getting info for non-existent model."""
    with pytest.raises(ValueError, match="not found"):
        get_model_info("nonexistent_model")


def test_get_all_metadata():
    """Test getting all metadata."""
    metadata = get_all_metadata()
    assert len(metadata) > 0
    assert "esm2_t33_650M" in metadata
    assert metadata["esm2_t33_650M"].task == "embedding"
