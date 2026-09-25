"""Authentication and request size enforcement for the API."""

import os
from secrets import compare_digest

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from bunker.api.limits import MAX_REQUEST_BYTES

bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    """Require the configured bearer token for every v1 endpoint."""
    expected = os.environ.get("BUNKER_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BUNKER_API_KEY is not configured",
        )
    if credentials is None or not compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


class RequestSizeLimitMiddleware:
    """Reject oversized bodies before JSON parsing or model inference."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > MAX_REQUEST_BYTES:
                        await self._reject(scope, receive, send)
                        return
                except ValueError:
                    pass

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                await self._reject(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)

        async def replay() -> dict:
            nonlocal body
            chunk, body = body, b""
            return {"type": "http.request", "body": chunk, "more_body": False}

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": "Request body exceeds 64 KiB"},
        )
        await response(scope, receive, send)
