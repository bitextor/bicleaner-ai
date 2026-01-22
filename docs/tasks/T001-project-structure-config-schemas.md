# T001 Project Structure + Config/Schemas

**Status:** Backlog
**Estimate:** 2 hours

---

## Context

### Current State
- bicleaner-ai is a CLI tool with no HTTP API
- No microservice structure exists

### Desired State
- `sources/bicleaner-service/` directory with FastAPI project structure
- Pydantic Settings for configuration
- Request/Response schemas for scoring API

---

## Implementation Plan

### Phase 1: Project Structure
- [ ] Create `sources/bicleaner-service/` directory
- [ ] Create `app/` package with `__init__.py`
- [ ] Create `app/routers/` package
- [ ] Create `pyproject.toml` with dependencies

### Phase 2: Configuration
- [ ] Create `app/config.py` with Pydantic Settings:
  - `model_type: str = "xlmr"`
  - `model_path: str = "bitextor/bicleaner-ai-full-en-xx"`
  - `batch_size: int = 32`
  - `max_batch_size: int = 100`
  - `inference_timeout_sec: int = 60`
  - Use `env_prefix="BICLEANER_"`

### Phase 3: Schemas
- [ ] Create `app/schemas.py`:
  - `ScoreItem`: id, source, target
  - `BatchScoreRequest`: requests list
  - `ScoreResult`: id, success, score, error
  - `BatchScoreResponse`: results list, metadata

---

## Technical Approach

**Directory structure:**
```
sources/bicleaner-service/
├── app/
│   ├── __init__.py
│   ├── config.py          # Pydantic Settings
│   ├── schemas.py         # Request/Response models
│   ├── routers/
│   │   └── __init__.py
├── pyproject.toml
└── README.md
```

**Config pattern:** Pydantic BaseSettings with environment variables (BICLEANER_* prefix)

**Schema pattern:** Follow prompsit-api QE schemas structure

---

## Acceptance Criteria

- [ ] **Given** `sources/bicleaner-service/` **When** structure created **Then** follows FastAPI conventions
- [ ] **Given** environment variables **When** Settings loaded **Then** BICLEANER_* prefix works
- [ ] **Given** BatchScoreRequest **When** validated **Then** enforces max_batch_size limit

---

## Affected Components

### Implementation
- `sources/bicleaner-service/` - NEW directory
- `sources/bicleaner-service/app/config.py` - NEW
- `sources/bicleaner-service/app/schemas.py` - NEW
- `sources/bicleaner-service/pyproject.toml` - NEW

### Documentation
- `sources/bicleaner-service/README.md` - NEW

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Project structure follows FastAPI conventions
- [ ] Config loads from environment variables
- [ ] Schemas validate request/response correctly
