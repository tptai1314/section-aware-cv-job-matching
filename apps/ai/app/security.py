"""Bao ve cac endpoint noi bo `/internal/*` (plan §15, H26).

AI service chi duoc NestJS (backend) goi qua mang noi bo, khong expose ra ngoai.
Khi `INTERNAL_API_KEY` duoc cau hinh, moi request toi router phai kem header
`X-Internal-Token` khop shared secret; so sanh bang `hmac.compare_digest` de
tranh timing attack. Khi chua cau hinh (dev/test) -> cho phep qua va chi log
canh bao mot lan.
"""

from __future__ import annotations

import hmac
import logging

from starlette.requests import Request

from . import errors
from .config import Settings

TOKEN_HEADER = "x-internal-token"

_warned = False


def require_internal_token(request: Request) -> None:
    """Dependency FastAPI: chặn request thiếu/sai `X-Internal-Token`.

    - Có `INTERNAL_API_KEY`: bắt buộc header khớp (401 nếu sai).
    - Không có khoá + `INTERNAL_API_KEY_REQUIRED=1`: **fail closed** — 503, không
      cho request nào lọt qua (design §8: production bắt buộc đặt khoá).
    - Không có khoá + không bật required: dev/test — cảnh báo 1 lần rồi cho qua.
    """
    global _warned
    settings: Settings = request.app.state.settings
    expected = (settings.internal_api_key or "").strip()
    if not expected:
        if settings.internal_api_key_required:
            raise errors.internal_key_not_configured()
        if not _warned:
            logging.getLogger("app.security").warning(
                "INTERNAL_API_KEY chưa cấu hình: /internal/* đang mở (chỉ dùng cho dev)."
            )
            _warned = True
        return
    provided = request.headers.get("x-internal-token", "")
    if not hmac.compare_digest(provided, expected):
        raise errors.unauthorized()


__all__ = ["require_internal_token"]
