# Tests

> **SCOPE:** Test organization and execution for bicleaner-ai. Contains directory structure, running tests. DO NOT add: testing philosophy (-> testing-strategy.md), implementation details.

## Overview

bicleaner-ai uses **pytest** for testing. Tests cover model training, classification, and utility functions.

## Test Structure

```
tests/
+-- bicleaner_ai_test.py    # Integration tests
|   +-- test_train_lite     # Train lite model (CPU, ~2 min)
|   +-- test_train_full     # Train full model (GPU required, ~5 min)
|   +-- test_classify       # Classification tests
|
+-- unit_test.py            # Unit tests
|   +-- test_noise_generation
|   +-- test_tokenization
|   +-- test_data_loading
|
+-- Test Data Files
|   +-- corpus.en-fr        # Parallel corpus (training)
|   +-- dev.en-fr           # Validation set
|   +-- mono.en-fr          # Monolingual corpus
|   +-- test.en-fr          # Test set
|   +-- wordfreq-fr.gz      # Word frequencies
|
+-- manual/
    +-- results/            # Manual test outputs (gitignored)
```

## Running Tests

### All Tests

```bash
pytest tests/
```

### Unit Tests Only (Fast)

```bash
pytest tests/unit_test.py
```

### Integration Tests

```bash
# Lite model (CPU)
pytest tests/bicleaner_ai_test.py::test_train_lite

# Full model (requires GPU)
pytest tests/bicleaner_ai_test.py::test_train_full
```

### With Verbose Output

```bash
pytest tests/ -v
```

### With Coverage

```bash
pytest tests/ --cov=bicleaner_ai --cov-report=html
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `BICLEANER_AI_THREADS` | Parallel processing | `4` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` or `` (CPU) |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow logs | `3` (errors only) |

## Test Requirements

### CPU Tests
- Python >= 3.8
- `pip install ".[all]"`

### GPU Tests
- CUDA >= 11.2
- TensorFlow with GPU support
- `CUDA_VISIBLE_DEVICES` set

## Quick Navigation

- [Testing Strategy](../docs/reference/guides/testing-strategy.md) - Philosophy and patterns
- [Architecture](../docs/architecture.md) - Model types and data flow
- [Training Guide](../docs/training/README.md) - Training documentation

## Maintenance

**Last Updated:** 2026-01-21

**Update Triggers:**
- New test files added
- Test structure changes
- Test data updates

**Verification:**
- [ ] Directory structure matches actual tests/
- [ ] Test runner commands are current
- [ ] Environment variables documented
