from __future__ import annotations

from fastapi import APIRouter, Response

from app import metrics

router = APIRouter(tags=["ops"])


@router.get("/metrics", include_in_schema=False, summary="Prometheus exposition")
async def prometheus_metrics() -> Response:
    return Response(
        content=metrics.render(),
        headers={"Content-Type": metrics.CONTENT_TYPE, "Cache-Control": "no-store"},
    )
