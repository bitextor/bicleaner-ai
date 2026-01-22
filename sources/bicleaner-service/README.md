# Bicleaner Service

HTTP microservice for bicleaner-ai parallel corpus quality scoring.

## Quick Start

```bash
# Build and run with Docker Compose
docker compose build
docker compose up

# Health check (wait for model to load, ~30-60 sec)
curl http://localhost:8080/health

# Score translation pairs
curl -X POST http://localhost:8080/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"id": "1", "source": "Hello world", "target": "Hola mundo"},
      {"id": "2", "source": "How are you?", "target": "Random noise text"}
    ]
  }'
```

## API

### POST /v1/score

Score a batch of source-target translation pairs.

**Request:**
```json
{
  "requests": [
    {"id": "1", "source": "Hello world", "target": "Hola mundo"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {"id": "1", "success": true, "score": 0.95}
  ],
  "metadata": {
    "model": "xlmr",
    "model_path": "bitextor/bicleaner-ai-full-en-xx",
    "count": 1
  }
}
```

**Score Interpretation:**
- 0.0-0.3: Likely noise/misalignment
- 0.3-0.5: Low quality, needs review
- 0.5-0.8: Acceptable quality
- 0.8-1.0: High quality translation pair

### GET /health

Health check endpoint. Returns 503 if model not loaded.

## Configuration

Environment variables (prefix `BICLEANER_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `BICLEANER_MODEL_TYPE` | `xlmr` | Model type |
| `BICLEANER_MODEL_PATH` | `bitextor/bicleaner-ai-full-en-xx` | HuggingFace model |
| `BICLEANER_BATCH_SIZE` | `32` | Inference batch size |
| `BICLEANER_MAX_BATCH_SIZE` | `100` | Max request batch size |
| `BICLEANER_HOST` | `0.0.0.0` | Server host |
| `BICLEANER_PORT` | `8080` | Server port |

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run locally (requires bicleaner-ai installed)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Run tests
pytest tests/
```

## Requirements

- Docker with NVIDIA GPU support
- CUDA 11.8+ / cuDNN 8.6+
- ~4GB GPU memory for xlmr model

## Architecture

```
bicleaner-service/
+-- app/
|   +-- main.py        # FastAPI app + lifespan
|   +-- config.py      # Pydantic Settings
|   +-- service.py     # BicleanerService wrapper
|   +-- schemas.py     # Request/Response models
|   +-- routers/
|       +-- v1.py      # POST /v1/score
+-- Dockerfile         # CUDA + TensorFlow
+-- docker-compose.yml
```
