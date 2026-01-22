"""V1 API router for bicleaner-service."""

from fastapi import APIRouter, HTTPException, Request

from ..config import settings
from ..schemas import BatchScoreRequest, BatchScoreResponse
from ..service import ServiceOverloadedError

router = APIRouter(prefix="/v1", tags=["scoring"])


@router.post("/score", response_model=BatchScoreResponse)
async def batch_score(request: Request, body: BatchScoreRequest) -> BatchScoreResponse:
    """Score a batch of source-target translation pairs.

    Returns quality scores (0-1) for each pair:
    - 0.0-0.3: Likely noise/misalignment
    - 0.3-0.5: Low quality, needs review
    - 0.5-0.8: Acceptable quality
    - 0.8-1.0: High quality translation pair

    Features:
    - Backpressure protection (503 when overloaded)
    - Inference timeout (returns error per item if timeout)
    - Queue depth in response metadata

    Raises:
        413: Batch size exceeds limit
        503: Service unavailable (model not loaded or overloaded)
    """
    service = request.app.state.service

    # Check model is loaded
    if service is None or not service._loaded:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: model not loaded"
        )

    # Check batch size limit
    if len(body.requests) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(body.requests)} exceeds limit {settings.max_batch_size}"
        )

    # Score the batch (with backpressure protection)
    try:
        results = await service.score_batch_async(body.requests)
    except ServiceOverloadedError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "5"},  # Suggest retry after 5 seconds
        )

    return BatchScoreResponse(
        results=results,
        metadata={
            "model": settings.model_type,
            "model_path": settings.model_path,
            "count": len(results),
            "queue_depth": service.queue_depth,
        }
    )
