# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

> **SCOPE:** Entry point with project overview and navigation ONLY. Contains project summary and links to detailed docs. DO NOT add: architecture details (-> architecture.md), dependencies (-> tech_stack.md), training details (-> docs/training/).

## Navigation

**DAG Structure:** CLAUDE.md -> docs/README.md -> topic docs.

## Project Overview

Bicleaner AI detects noisy sentence pairs in parallel corpora for machine translation. Outputs score (0-1) indicating translation quality.

## Documentation

All documentation accessible through **[docs/README.md](docs/README.md)**.

**Quick links:**
- **Architecture:** [docs/architecture.md](docs/architecture.md) - Model types, data flow, module structure
- **Tech Stack:** [docs/tech_stack.md](docs/tech_stack.md) - Dependencies, CUDA compatibility
- **Training:** [docs/training/README.md](docs/training/README.md) - Model training guides
- **ADRs:** [docs/reference/adrs/](docs/reference/adrs/) - Architecture Decision Records

## Project Highlights

**Unique to bicleaner-ai:**
- **Three model types:** dec_attention (fast), transformer, xlmr (accurate)
- **Synthetic noise generation:** 1:9 positive:negative ratio
- **Optional hardrules:** Requires `--disable_hardrules` without hardrules feature

## Folder Structure

```
src/bicleaner_ai/
+-- CLI: bicleaner_ai_classifier.py, bicleaner_ai_train.py, bicleaner_ai_download.py
+-- Core: classify.py, training.py, models.py
+-- Models: decomposable_attention.py, layers.py
+-- Data: datagen.py, tokenizer.py, noise_generation.py
```

## Technology Stack

**Core:** Python 3.8+ | TensorFlow 2.6.5-2.15.x | transformers 4.52.4 | scikit-learn

See [docs/tech_stack.md](docs/tech_stack.md) for complete dependencies and CUDA compatibility.

## Commands

```bash
# Install
pip install .                    # Basic
pip install ".[all]"             # All features

# Classify
bicleaner-ai-classify --scol 1 --tcol 2 input.tsv output.tsv model

# Train
bicleaner-ai-train -m model -s en -t fr --parallel_train corpus --parallel_valid dev

# Test
pytest tests/
```

## Maintenance

**Update trigger:** After changing navigation or adding new documentation.

**Last Updated:** 2026-01-21
