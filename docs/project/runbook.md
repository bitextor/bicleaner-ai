# Runbook

> **SCOPE:** Operational procedures for deploying and maintaining bicleaner-ai microservice. Contains Docker setup, deployment, troubleshooting, monitoring. DO NOT add: architecture (-> architecture.md), training (-> training/), API reference (-> README.md).

## Important: Docker-Only Development

Local development without Docker is **not supported** due to TensorFlow/CUDA package conflicts with host OS. All development, testing, and deployment must be done through Docker containers.

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Docker | >= 20.10 | **Required** for all development |
| Docker Compose | >= 2.0 | Plugin or standalone |
| NVIDIA Driver | >= 520 | For GPU support (optional) |
| NVIDIA Container Toolkit | latest | `nvidia-docker2` package (optional) |

## 2. Quick Start

### 2.1 Auto-Start (Recommended)

```bash
# Windows
scripts\run.bat

# Linux/Mac
./scripts/run.sh
```

Scripts auto-detect GPU availability and start appropriate mode.

### 2.2 Manual Start

```bash
# With GPU
docker compose up -d

# CPU only (no GPU)
docker compose --profile cpu up -d bicleaner-service-cpu
```

### 2.3 Verify Service

```bash
curl http://localhost:8057/health
```

Expected response:
```json
{"status":"healthy","model":"xlmr","model_path":"bitextor/bicleaner-ai-full-en-xx"}
```

## 3. Usage

### 3.1 Score via API

```bash
curl -X POST http://localhost:8057/v1/score \
  -H "Content-Type: application/json" \
  -d '{"requests": [{"id": "1", "source": "Hello", "target": "Hola"}]}'
```

### 3.2 Score File

```bash
# Input: TSV file (source<TAB>target)
# Output: TSV file (source<TAB>target<TAB>score)
python scripts/score_file.py input.tsv output.tsv
```

### 3.3 Run Tests

```bash
# Requires service running
pip install httpx pytest
pytest tests/test_api.py -v
```

## 4. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BICLEANER_MODEL_TYPE` | `xlmr` | Model: `xlmr`, `dec_attention`, `transformer` |
| `BICLEANER_MODEL_PATH` | `bitextor/bicleaner-ai-full-en-xx` | HuggingFace model or local path |
| `BICLEANER_BATCH_SIZE` | `32` (GPU) / `16` (CPU) | Default inference batch size |
| `BICLEANER_MAX_BATCH_SIZE` | `100` (GPU) / `50` (CPU) | Maximum API batch limit |
| `BICLEANER_INFERENCE_TIMEOUT_SEC` | `60` | Request timeout seconds |
| `BICLEANER_HOST` | `0.0.0.0` | Server bind host |
| `BICLEANER_PORT` | `8080` | Internal container port |
| `HF_HOME` | `/cache/huggingface` | HuggingFace cache directory |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | TensorFlow log level (0=all, 3=errors) |
| `CUDA_VISIBLE_DEVICES` | `0` (GPU) / `-1` (CPU) | GPU device selection |

Override in `docker-compose.yml` environment section or pass via `-e` flag.

## 5. Configuration

### 5.1 Port

| External Port | Internal Port | Service |
|---------------|---------------|---------|
| **8057** | 8080 | FastAPI application |

### 5.2 Volume

| Volume | Container Path | Purpose |
|--------|----------------|---------|
| `bicleaner_hf_cache` | `/cache/huggingface` | Persist downloaded models |

### 5.3 GPU/CPU Modes

| Mode | Command | Batch Size | Notes |
|------|---------|------------|-------|
| **GPU** | `docker compose up -d` | 32 | Default, requires NVIDIA |
| **CPU** | `docker compose --profile cpu up -d bicleaner-service-cpu` | 16 | Slower, no GPU required |

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 6. Health Check

### 6.1 Endpoints

| Endpoint | Method | Success | Failure | Description |
|----------|--------|---------|---------|-------------|
| `/health` | GET | 200 | 503 | Model ready check |
| `/docs` | GET | 200 | - | OpenAPI documentation |
| `/v1/score` | POST | 200 | 413/422/503 | Batch scoring |

### 6.2 Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `interval` | 30s | Regular health polling |
| `timeout` | 10s | Fast failure detection |
| `start_period` | 120s | Model loading time (xlmr: 30-60s) |
| `retries` | 3 | Transient failure tolerance |

### 6.3 Startup Verification

1. Check container status: `docker compose ps`
2. Wait for start_period (120s) for model loading
3. Verify health: `curl http://localhost:8057/health`
4. Check GPU in logs: `docker compose logs | grep -i gpu`

## 7. Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `/health` returns 503 | Model not loaded | Wait 120s; check logs for TensorFlow errors |
| `CUDA_ERROR_OUT_OF_MEMORY` | Insufficient GPU memory | Reduce `BICLEANER_BATCH_SIZE`; use `dec_attention` model |
| TensorFlow not using GPU | CUDA/cuDNN mismatch | Dockerfile uses TF 2.15.1 + CUDA 11.8 |
| Container exits immediately | Model path invalid | Check `BICLEANER_MODEL_PATH` accessible |
| Slow inference | CPU fallback | Set `TF_CPP_MIN_LOG_LEVEL=0`, check GPU detection |
| Docker build fails | Package conflicts | Clean build: `docker compose build --no-cache` |
| Port already in use | Conflict | Change port in `docker-compose.yml` or stop conflicting service |
| Local tests fail | OS package conflicts | **Use Docker** - local development not supported |
| GPU not detected | Docker GPU missing | Use `--profile cpu` for CPU-only mode |

## 8. Maintenance

### 8.1 View Logs

```bash
docker compose logs -f bicleaner-service
```

### 8.2 Restart Service

```bash
docker compose restart bicleaner-service
```

### 8.3 Rebuild After Changes

```bash
docker compose down && docker compose build && docker compose up -d
```

### 8.4 Cache Management

```bash
# Clear model cache (forces re-download)
docker volume rm bicleaner_hf_cache
docker compose up -d
```

### 8.5 Version Updates

```bash
git pull && docker compose build && docker compose up -d
```

## 9. CUDA Compatibility

| TensorFlow | CUDA | cuDNN | Notes |
|------------|------|-------|-------|
| **2.15.1** | **11.8** | **8.7** | **Used in Dockerfile** |
| 2.14.x | 11.8 | 8.7 | Compatible |
| 2.13.x | 11.8 | 8.6 | Compatible |

## 10. File Structure

```
bicleaner-ai/
├── Dockerfile              # Container build
├── docker-compose.yml      # Service configuration (GPU + CPU profiles)
├── app/                    # FastAPI service
│   ├── main.py            # Application entry point
│   ├── config.py          # Pydantic settings
│   ├── service.py         # BicleanerService wrapper
│   ├── schemas.py         # Request/Response models
│   └── routers/v1.py      # POST /v1/score endpoint
├── scripts/
│   ├── run.sh             # Linux launcher (auto GPU detection)
│   ├── run.bat            # Windows launcher
│   └── score_file.py      # CLI file scoring tool
└── tests/
    └── test_api.py        # API integration tests
```

## 11. References

| Document | Path | Description |
|----------|------|-------------|
| Dockerfile | [Dockerfile](../../Dockerfile) | Container build |
| Docker Compose | [docker-compose.yml](../../docker-compose.yml) | Service configuration |
| API Tests | [tests/test_api.py](../../tests/test_api.py) | Integration tests |
| Score Script | [scripts/score_file.py](../../scripts/score_file.py) | CLI scoring tool |
| Architecture | [docs/architecture.md](../architecture.md) | Model types, data flow |
| Tech Stack | [docs/tech_stack.md](../tech_stack.md) | Dependencies, CUDA matrix |

---

**Last Updated:** 2026-01-22
