from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


def extract_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.headers.get("x-api-key", "")


def auth_error_response(*, message: str = "Invalid API key", protocol: str = "openai") -> JSONResponse:
    if protocol == "anthropic":
        body = {"error": {"type": "authentication_error", "message": message}}
    else:
        body = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }
    return JSONResponse(status_code=401, content=body)
