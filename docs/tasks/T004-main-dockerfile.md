# T004 Main + Dockerfile

**Status:** Backlog
**Estimate:** 2 hours

---

## Context

### Current State
- Project structure exists (T001)
- BicleanerService exists (T002)
- Router exists (T003)
- No application entry point or container

### Desired State
- FastAPI application with lifespan (model loading at startup)
- Dockerfile with CUDA + TensorFlow
- Health check endpoint

---

## Implementation Plan

### Phase 1: FastAPI Application
- [ ] Create `app/main.py`
- [ ] Implement lifespan context manager:
  - Startup: Load BicleanerService model
  - Shutdown: Cleanup executor
- [ ] Include v1 router
- [ ] Add /health endpoint

### Phase 2: Dockerfile
- [ ] Base image: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- [ ] Install Python 3.10+
- [ ] Install TensorFlow with GPU support
- [ ] Install bicleaner-ai from parent directory
- [ ] Configure healthcheck with start_period=120s (model loading)

### Phase 3: Docker Compose Integration
- [ ] Add service to prompsit-api docker-compose.yml (reference only)
- [ ] Configure volumes for model cache
- [ ] GPU reservation

---

## Technical Approach

**Lifespan pattern:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    service = BicleanerService(settings.model_type, settings.model_path)
    service.load()  # 10-30 sec
    app.state.service = service
    yield
    # Shutdown
    service.executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)
app.include_router(v1_router)

@app.get("/health")
async def health():
    if not hasattr(app.state, "service") or app.state.service.model is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "healthy", "model": settings.model_type}
```

**Dockerfile pattern:**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install dependencies
COPY pyproject.toml .
RUN pip install .

# Install bicleaner-ai from parent
COPY --from=bicleaner-ai /app /bicleaner-ai
RUN pip install /bicleaner-ai

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## Acceptance Criteria

- [ ] **Given** docker build **When** Dockerfile built **Then** image created successfully
- [ ] **Given** container start **When** model loads **Then** /health returns healthy
- [ ] **Given** GPU available **When** container runs **Then** TensorFlow uses GPU
- [ ] **Given** /health **When** model not loaded **Then** returns 503

---

## Affected Components

### Implementation
- `sources/bicleaner-service/app/main.py` - NEW
- `sources/bicleaner-service/Dockerfile` - NEW

### Documentation
- `sources/bicleaner-service/README.md` - UPDATE with Docker instructions

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Application starts with uvicorn
- [ ] Model loads at startup (lifespan)
- [ ] Docker image builds successfully
- [ ] Health check works after model load
