# T002 BicleanerService Model Wrapper

**Status:** Backlog
**Estimate:** 3 hours

---

## Context

### Current State
- `src/bicleaner_ai/models.py` has `BaseModel.predict()` for scoring
- `src/bicleaner_ai/util.py` has `get_model()` factory function
- No async wrapper exists

### Desired State
- `BicleanerService` class wrapping existing model
- Singleton pattern for model loading (once at startup)
- ThreadPoolExecutor for async inference (TensorFlow is blocking)

---

> [!WARNING]
> **DRY Check:** Reuse existing functionality
> - Existing: `src/bicleaner_ai/models.py:BaseModel.predict()`
> - Existing: `src/bicleaner_ai/util.py:get_model()`
> - **Recommendation:** REUSE existing model classes (Option 1)
>   - Import `get_model` from bicleaner_ai.util
>   - Call `model.predict(sources, targets)` for scoring
>   - DO NOT reimplement model logic

---

## Implementation Plan

### Phase 1: Service Class
- [ ] Create `app/service.py`
- [ ] Implement `BicleanerService` class:
  - Constructor: model_type, model_path
  - `load()`: Load model using `get_model()`
  - `score_batch()`: Synchronous batch scoring
  - `score_batch_async()`: Async wrapper with ThreadPoolExecutor

### Phase 2: Model Integration
- [ ] Import `get_model` from `bicleaner_ai.util`
- [ ] Call `model.predict(x1, x2, batch_size)` for scoring
- [ ] Handle numpy array to list conversion

### Phase 3: Error Handling
- [ ] Catch model loading errors
- [ ] Handle inference timeouts
- [ ] Return partial results on batch errors

---

## Technical Approach

**Service pattern:**
```python
class BicleanerService:
    def __init__(self, model_type: str, model_path: str):
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def load(self):
        from bicleaner_ai.util import get_model
        self.model = get_model(model_type)(model_path, {})
        self.model.load()  # 10-30 sec at startup

    def score_batch(self, sources: list[str], targets: list[str]) -> list[float]:
        predictions = self.model.predict(sources, targets, batch_size=32)
        return predictions.flatten().tolist()

    async def score_batch_async(self, items: list[dict]) -> list[dict]:
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(self.executor, self._score, items)
        return [{"id": id, "success": True, "score": score} for ...]
```

**Key insight:** TensorFlow operations are blocking, so we use ThreadPoolExecutor to avoid blocking the event loop.

---

## Acceptance Criteria

- [ ] **Given** model_type="xlmr" **When** load() called **Then** model loaded from HuggingFace
- [ ] **Given** source/target pairs **When** score_batch() called **Then** returns scores 0-1
- [ ] **Given** async context **When** score_batch_async() called **Then** doesn't block event loop
- [ ] **Given** invalid pairs **When** scoring **Then** returns error in result item

---

## Affected Components

### Implementation
- `sources/bicleaner-service/app/service.py` - NEW

### Dependencies
- `bicleaner_ai.util.get_model` - REUSE
- `bicleaner_ai.models.BaseModel.predict` - REUSE

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Model loads successfully at startup
- [ ] Scoring returns correct values (matches CLI output)
- [ ] Async wrapper doesn't block FastAPI
