# T003 FastAPI Router POST /v1/score

**Status:** Backlog
**Estimate:** 2 hours

---

## Context

### Current State
- BicleanerService exists (T002)
- Schemas exist (T001)
- No HTTP endpoint

### Desired State
- POST /v1/score endpoint for batch scoring
- Proper error handling (400, 413, 503)
- JSON request/response

---

## Implementation Plan

### Phase 1: Router Setup
- [ ] Create `app/routers/v1.py`
- [ ] Create APIRouter with prefix="/v1"
- [ ] Add tags=["scoring"]

### Phase 2: Score Endpoint
- [ ] Implement POST /v1/score handler
- [ ] Accept BatchScoreRequest body
- [ ] Call BicleanerService.score_batch_async()
- [ ] Return BatchScoreResponse

### Phase 3: Error Handling
- [ ] 400: Invalid request (validation errors)
- [ ] 413: Batch too large (>max_batch_size)
- [ ] 503: Service unavailable (model not loaded)

---

## Technical Approach

**Endpoint pattern:**
```python
router = APIRouter(prefix="/v1", tags=["scoring"])

@router.post("/score", response_model=BatchScoreResponse)
async def batch_score(
    request: BatchScoreRequest,
    service: BicleanerService = Depends(get_service)
) -> BatchScoreResponse:
    if len(request.requests) > settings.max_batch_size:
        raise HTTPException(413, "Batch size exceeds limit")

    results = await service.score_batch_async(request.requests)

    return BatchScoreResponse(
        results=results,
        metadata={"model": settings.model_type, "count": len(results)}
    )
```

**Dependency injection:** Use `Depends(get_service)` to inject BicleanerService singleton.

---

## Acceptance Criteria

- [ ] **Given** valid BatchScoreRequest **When** POST /v1/score **Then** returns scores
- [ ] **Given** batch > max_batch_size **When** POST **Then** returns 413
- [ ] **Given** model not loaded **When** POST **Then** returns 503
- [ ] **Given** invalid JSON **When** POST **Then** returns 400 with details

---

## Affected Components

### Implementation
- `sources/bicleaner-service/app/routers/v1.py` - NEW
- `sources/bicleaner-service/app/routers/__init__.py` - UPDATE

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Endpoint accessible at POST /v1/score
- [ ] OpenAPI docs generated correctly
- [ ] Error responses follow RFC 9457 format
