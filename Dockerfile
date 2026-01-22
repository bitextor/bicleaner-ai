# Bicleaner Service Dockerfile
# Build: docker build -t bicleaner-service .
# Run:   docker run -d -p 8080:8080 --gpus all bicleaner-service
#
# Note: Uses local bicleaner-ai source to avoid glove dependency
# (glove is only needed for dec_attention/lite models, not xlmr)

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# =============================================================================
# Install dependencies (avoiding bicleaner-ai-glove for xlmr model)
# =============================================================================

# Core ML dependencies
RUN pip install --no-cache-dir \
    tensorflow==2.15.1 \
    transformers==4.52.4 \
    "huggingface-hub>=0.30,<1" \
    sentencepiece \
    protobuf==3.20.3 \
    "numpy<2" \
    scikit-learn \
    PyYAML \
    regex \
    toolwrapper

# FastAPI dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "pydantic>=2.0" \
    "pydantic-settings>=2.0"

# =============================================================================
# Copy and install bicleaner-ai from local source
# =============================================================================

# Copy bicleaner-ai source
COPY src/bicleaner_ai /app/bicleaner_ai

# Add to Python path (instead of pip install to avoid glove dep)
ENV PYTHONPATH=/app:$PYTHONPATH

# =============================================================================
# Copy bicleaner-service application
# =============================================================================

COPY app /app/app

# =============================================================================
# Environment and runtime configuration
# =============================================================================

ENV BICLEANER_MODEL_TYPE=xlmr
ENV BICLEANER_MODEL_PATH=bitextor/bicleaner-ai-full-en-xx
ENV BICLEANER_BATCH_SIZE=32
ENV BICLEANER_MAX_BATCH_SIZE=100
ENV BICLEANER_HOST=0.0.0.0
ENV BICLEANER_PORT=8080

# HuggingFace cache
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

# TensorFlow settings
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=0

# Create cache directory
RUN mkdir -p /cache/huggingface && chmod 777 /cache/huggingface

EXPOSE 8080

# Health check with long start period for model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
