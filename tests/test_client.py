"""Tests for BunkerClient."""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock

from bunker_fold.client import BunkerClient


@pytest.fixture
def client():
    """Create a test client."""
    return BunkerClient("https://api.bunker.local", api_key="test-key")


def test_client_initialization():
    """Test client initialization."""
    client = BunkerClient("https://api.bunker.local", api_key="test-key")
    assert client.base_url == "https://api.bunker.local"
    assert client.api_key == "test-key"


def test_base_url_trailing_slash_removal():
    """Test that trailing slashes are removed from base URL."""
    client = BunkerClient("https://api.bunker.local/")
    assert client.base_url == "https://api.bunker.local"


@patch("bunker_fold.client.requests.Session.request")
def test_get_request(mock_request, client):
    """Test GET request."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": "test"}
    mock_request.return_value = mock_response

    result = client.get("/test")
    assert result == {"data": "test"}
    mock_request.assert_called_once()


@patch("bunker_fold.client.requests.Session.request")
def test_post_request(mock_request, client):
    """Test POST request."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 1}
    mock_request.return_value = mock_response

    result = client.post("/test", json={"name": "value"})
    assert result == {"id": 1}


def test_context_manager(client):
    """Test context manager support."""
    with BunkerClient("https://api.bunker.local") as c:
        assert isinstance(c, BunkerClient)
