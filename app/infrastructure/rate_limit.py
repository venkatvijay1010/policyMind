"""Shared rate-limit configuration for HTTP endpoints."""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

# Endpoint decorators provide reliable enforcement across FastAPI router versions.
# A shared in-memory backend is suitable for local development; production should
# configure SlowAPI with a shared Redis storage URI before running multiple replicas.
limiter = Limiter(
    key_func=get_remote_address,
    headers_enabled=False,
    storage_uri=settings.rate_limit_storage_uri,
)
