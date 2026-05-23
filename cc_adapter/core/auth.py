from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

_admin_password: str = ""
_TOKEN_TTL: int = 86400


def set_password(password: str) -> None:
    global _admin_password
    _admin_password = password


def _password_hash() -> str:
    return hashlib.sha256(_admin_password.encode()).hexdigest()[:16]


def _sign(payload: str) -> str:
    return hmac.new(_admin_password.encode(), payload.encode(), hashlib.sha256).hexdigest()


def generate_token() -> str:
    exp = int(time.time()) + _TOKEN_TTL
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp, "pwh": _password_hash()}).encode()).decode()
    sig = _sign(payload)
    return f"{payload}.{sig}"


def check_api_access(access_key: str, token: str, admin_password: str = "") -> bool:
    """Returns True if the token grants API access."""
    if not access_key:
        return True
    if token == access_key:
        return True
    if admin_password and validate_token(token):
        return True
    return False


def validate_token(token: str) -> bool:
    if not _admin_password:
        return False
    try:
        payload_b64, sig = token.split(".", 1)
        if not hmac.compare_digest(_sign(payload_b64), sig):
            return False
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        if time.time() > payload["exp"]:
            return False
        if payload.get("pwh") != _password_hash():
            return False
        return True
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return False


from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from cc_adapter.core.headers import extract_token, auth_error_response
from cc_adapter.core.runtime import get_client

AUTH_PROTECTED_PATHS = {"/v1/chat/completions", "/v1/messages", "/v1/responses"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path not in AUTH_PROTECTED_PATHS:
            return await call_next(request)

        import structlog

        logger = structlog.get_logger(__name__)

        from cc_adapter.core.config import get_config_or_default

        cfg = get_config_or_default()
        if not cfg.access_key:
            return await call_next(request)

        token = extract_token(request)
        if not check_api_access(cfg.access_key, token, cfg.admin_password or ""):
            protocol = _protocol_from_path(request.url.path)
            logger.warning("auth.failed", reason="invalid_access_key", path=request.url.path)
            return auth_error_response(protocol=protocol)

        client = get_client()
        if client is not None:
            if not client.api_key:
                return auth_error_response(
                    message="CC_ADAPTER_CC_API_KEY is not configured",
                    protocol=_protocol_from_path(request.url.path),
                )
        elif not cfg.cc_api_key:
            return auth_error_response(
                message="CC_ADAPTER_CC_API_KEY is not configured",
                protocol=_protocol_from_path(request.url.path),
            )

        return await call_next(request)


def _protocol_from_path(path: str) -> str:
    if path == "/v1/messages":
        return "anthropic"
    return "openai"
