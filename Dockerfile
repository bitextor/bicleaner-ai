# Bicleaner Service Dockerfile
#
# GPU build (default):
#   docker build -t bicleaner-service .
#   docker run -d -p 8080:8080 --gpus all bicleaner-service
#
# CPU build:
#   docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.15.0 -t bicleaner-service-cpu .
#   docker run -d -p 8080:8080 bicleaner-service-cpu
#
# Note: Uses local bicleaner-ai source to avoid glove dependency
# (glove is only needed for dec_attention/lite models, not xlmr)

# GPU by default, override with --build-arg BASE_IMAGE for CPU
ARG BASE_IMAGE=tensorflow/tensorflow:2.15.0-gpu
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =============================================================================
# Install dependencies (avoiding bicleaner-ai-glove for xlmr model)
# =============================================================================

# Core ML dependencies (TensorFlow already installed in base image)
RUN pip install --no-cache-dir \
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
# Copy ONLY inference-required bicleaner-ai modules (not training code)
# =============================================================================

# Create package directory
RUN mkdir -p /app/bicleaner_ai

# Copy only the 9 files required for inference
# (excludes 14 training-only files: training.py, noise_generation.py, CLIs, etc.)
COPY src/bicleaner_ai/__init__.py /app/bicleaner_ai/
COPY src/bicleaner_ai/util.py /app/bicleaner_ai/
COPY src/bicleaner_ai/models.py /app/bicleaner_ai/
COPY src/bicleaner_ai/decomposable_attention.py /app/bicleaner_ai/
COPY src/bicleaner_ai/models_util.py /app/bicleaner_ai/
COPY src/bicleaner_ai/metrics.py /app/bicleaner_ai/
COPY src/bicleaner_ai/datagen.py /app/bicleaner_ai/
COPY src/bicleaner_ai/layers.py /app/bicleaner_ai/
COPY src/bicleaner_ai/losses.py /app/bicleaner_ai/

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
