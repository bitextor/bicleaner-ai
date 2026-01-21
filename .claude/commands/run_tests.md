---
description: Run bicleaner-ai test suites
allowed-tools: Read, Bash
---

# Run Tests

Execute pytest test suites for bicleaner-ai.

---

## Quick Start

```bash
pytest tests/
```

---

## Test Options

| Test | Command | Description |
|------|---------|-------------|
| All tests | `pytest tests/` | Unit + integration tests |
| Unit tests | `pytest tests/unit_test.py` | Fast unit tests |
| Lite model training | `pytest tests/bicleaner_ai_test.py::test_train_lite` | Train lite model (CPU) |
| Full model training | `pytest tests/bicleaner_ai_test.py::test_train_full` | Train full model (requires GPU) |
| Verbose output | `pytest tests/ -v` | Detailed test output |

---

## Prerequisites

Install all dependencies before running tests:

```bash
pip install ".[all]"
```

For training tests, ensure CUDA is available (>= 11.2).

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `BICLEANER_AI_THREADS` | Number of threads (default: all cores) |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow log level (0=all, 3=errors only) |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | `pip install ".[all]"` |
| GPU not detected | Check `CUDA_VISIBLE_DEVICES`, TensorFlow CUDA compatibility |
| Slow training tests | Use lite model test or set `CUDA_VISIBLE_DEVICES` |

---

**Last Updated:** 2026-01-21
