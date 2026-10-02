from __future__ import annotations

import hashlib
import hmac

SIGNATURE_HEADER = "x-hub-signature-256"
SIGNATURE_PREFIX = "sha256="


def compute_signature(secret: str, body: bytes) -> str:
    """Return the ``sha256=<hex>`` value Meta sends in X-Hub-Signature-256."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{SIGNATURE_PREFIX}{digest}"


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    """Constant-time comparison of a webhook payload signature against the app secret."""
    if not header or not secret:
        return False
    candidate = header.strip()
    if not candidate.startswith(SIGNATURE_PREFIX):
        return False
    return hmac.compare_digest(compute_signature(secret, body), candidate)
