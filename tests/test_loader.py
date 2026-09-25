"""Tests for model loader."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from bunker.loader import get_cache_dir, get_device, load


def test_get_cache_dir_default():
    """Test default cache directory."""
    cache_dir = get_cache_dir()
    assert cache_dir.exists()
    assert cache_dir.name == "bunker"


def test_get_cache_dir_custom(monkeypatch):
    """Test custom cache directory via env var."""
    custom_path = "/tmp/custom_bunker_cache"
    monkeypatch.setenv("BUNKER_CACHE", custom_path)
    cache_dir = get_cache_dir()
    assert str(cache_dir) == custom_path


def test_get_device_auto_cpu():
    """Test automatic device selection falls back to CPU."""
    with patch("bunker.loader.torch", None):
        device = get_device("auto")
        assert device == "cpu"


def test_get_device_explicit():
    """Test explicit device selection."""
    assert get_device("cuda") == "cuda"
    assert get_device("cpu") == "cpu"


def test_load_missing_model():
    """Test loading a non-existent model."""
    with pytest.raises(ValueError, match="not found"):
        load("nonexistent_model")


def test_load_missing_dependencies():
    """Test loading model without required dependencies."""
    with patch("bunker.registry.get_model_info") as mock_info:
        mock_meta = MagicMock()
        mock_meta.extra = "esm"
        mock_meta.loader.side_effect = ImportError("No module named 'esm'")
        mock_info.return_value = mock_meta

        with pytest.raises(ImportError, match="requires the 'esm' extra"):
            load("esm2_t33_650M")
