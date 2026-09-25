"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from bunker.api.app import app

TEST_KEY = "t" * 32
client = TestClient(app, headers={"Authorization": f"Bearer {TEST_KEY}"})


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("BUNKER_API_KEY", TEST_KEY)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    # Manually set start_time for test context
    import time

    app.state.start_time = time.time()

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime" in data


def test_list_models():
    """Test model listing endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert any(m["id"] == "esm2_t33_650M" for m in data["data"])


def test_get_model():
    """Test get model info endpoint."""
    response = client.get("/v1/models/esm2_t33_650M")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "esm2_t33_650M"
    assert data["object"] == "model"


def test_get_model_not_found():
    """Test getting non-existent model."""
    response = client.get("/v1/models/nonexistent")
    assert response.status_code in [404, 500]  # May be 500 if wrapped by handler


def test_embeddings_endpoint_schema():
    """Test embeddings endpoint validates schema."""
    # Missing required field
    response = client.post("/v1/embeddings", json={})
    assert response.status_code == 422  # Validation error


def test_embeddings_wrong_task():
    """Test embeddings endpoint rejects non-embedding models."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "esmfold",  # Structure model, not embedding
            "input": "MKTAYIAKQRQISFVK",
        },
    )
    assert response.status_code == 400
    data = response.json()
    error_msg = data.get("error", {}).get("message", "") or data.get("detail", "")
    assert "not an embedding model" in error_msg


def test_structures_endpoint_schema():
    """Test structures endpoint validates schema."""
    # Missing required field
    response = client.post("/v1/completions", json={})
    assert response.status_code == 422  # Validation error


def test_structures_wrong_task():
    """Test structures endpoint rejects non-structure models."""
    response = client.post(
        "/v1/completions",
        json={
            "model": "esm2_t33_650M",  # Embedding model, not structure
            "prompt": "MKTAYIAKQRQISFVK",
        },
    )
    assert response.status_code == 400
    data = response.json()
    error_msg = data.get("error", {}).get("message", "") or data.get("detail", "")
    assert "not a structure prediction model" in error_msg


@pytest.mark.slow
def test_embeddings_full_integration():
    """Integration test for embeddings endpoint (requires esm package)."""
    pytest.importorskip("esm")

    response = client.post(
        "/v1/embeddings",
        json={
            "model": "esm2_t6_8M",  # Smallest model for testing
            "input": "MKTAYIAKQRQISFVK",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert "embedding" in data["data"][0]
    assert len(data["data"][0]["embedding"]) > 0
