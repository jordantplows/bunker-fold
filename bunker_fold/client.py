"""Bunker API client."""

import requests
from typing import Any

from .models import ApiResponse


class BunkerClient:
    """Client for interacting with the Bunker API."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        """Initialize the Bunker client.

        Args:
            base_url: The base URL of the Bunker API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments to pass to requests

        Returns:
            Parsed response data

        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    def get(self, endpoint: str, **kwargs: Any) -> Any:
        """Make a GET request."""
        return self._request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs: Any) -> Any:
        """Make a POST request."""
        return self._request("POST", endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs: Any) -> Any:
        """Make a PUT request."""
        return self._request("PUT", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs: Any) -> Any:
        """Make a DELETE request."""
        return self._request("DELETE", endpoint, **kwargs)

    def close(self) -> None:
        """Close the session."""
        self.session.close()

    def __enter__(self) -> "BunkerClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
