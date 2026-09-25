"""Security and resource limits for the HTTP API."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from bunker.api.app import app
from bunker.api.model_cache import ModelTooLargeError, get_or_load_model

TEST_KEY = "t" * 32


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("BUNKER_API_KEY", TEST_KEY)
    with TestClient(app) as test_client:
        yield test_client


def test_server_requires_key_at_startup(monkeypatch):
    monkeypatch.delenv("BUNKER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Set BUNKER_API_KEY"):
        with TestClient(app):
            pass
    monkeypatch.setenv("BUNKER_API_KEY", "too-short")
    with pytest.raises(RuntimeError, match="at least 32 characters"):
        with TestClient(app):
            pass


def test_v1_requires_valid_bearer_key(client):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert (
        client.post(
            "/v1/embeddings", json={"model": "esm2_t6_8M", "input": "MKT"}
        ).status_code
        == 401
    )
    assert (
        client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert (
        client.get(
            "/v1/models", headers={"Authorization": f"Bearer {TEST_KEY}"}
        ).status_code
        == 200
    )


def test_request_and_sequence_limits(client):
    headers = {"Authorization": f"Bearer {TEST_KEY}"}
    for sequence in ("", "M" * 1025):
        response = client.post(
            "/v1/embeddings",
            headers=headers,
            json={"model": "esm2_t6_8M", "input": sequence},
        )
        assert response.status_code == 422
    response = client.post(
        "/v1/completions",
        headers=headers,
        json={"model": "esmfold", "prompt": ["M"] * 5},
    )
    assert response.status_code == 422
    response = client.post(
        "/v1/embeddings",
        headers=headers,
        json={"model": "esm2_t6_8M", "input": "M" * 70000},
    )
    assert response.status_code == 413
    response = client.post(
        "/v1/embeddings",
        headers={**headers, "Content-Type": "application/json"},
        content='{"model":"esm2_t6_8M","input":"' + "M" * 70000 + '"}',
    )
    assert response.status_code == 413
    response = client.post(
        "/v1/embeddings",
        headers=headers,
        content=iter([b" " * 40000, b" " * 40000]),
    )
    assert response.status_code == 413


def test_untrusted_origin_has_no_cors_access(client):
    response = client.options(
        "/v1/embeddings",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") is None


def test_model_cache_eviction_and_memory_budget(client, monkeypatch):
    cache = {}
    with patch("bunker.api.model_cache.load", side_effect=lambda name, device: name):
        assert get_or_load_model(cache, "esm2_t6_8M") == "esm2_t6_8M"
        assert get_or_load_model(cache, "esm2_t12_35M") == "esm2_t12_35M"
        assert list(cache) == ["esm2_t12_35M"]
        with pytest.raises(ModelTooLargeError):
            get_or_load_model(cache, "esm2_t48_15B")
    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": f"Bearer {TEST_KEY}"},
        json={"model": "esm2_t48_15B", "input": "MKT"},
    )
    assert response.status_code == 413
    monkeypatch.setenv("BUNKER_MAX_MODEL_MEMORY_GB", "64")
    with patch("bunker.api.model_cache.load", return_value="large-model"):
        assert get_or_load_model(cache, "esm2_t48_15B") == "large-model"


def test_structure_response_accepts_residue_scores(client):
    class StructureModel:
        def predict(self, sequences, temperature=1.0):
            return [
                {"pdb": "MODEL", "plddt": [80.0] * len(sequence), "mean_plddt": 80.0}
                for sequence in sequences
            ]

    app.state.model_cache["esmfold"] = StructureModel()
    response = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {TEST_KEY}"},
        json={"model": "esmfold", "prompt": "MKT"},
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["plddt"] == [80.0, 80.0, 80.0]
