# Technology Stack

> **SCOPE:** Dependencies and compatibility. Contains versions, CUDA matrix, installation commands. DO NOT add: architecture (-> architecture.md), training (-> training/), usage examples (-> CLAUDE.md).

Bicleaner AI dependencies and compatibility.

## Runtime Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | >= 3.8 | |
| pip | >= 23.0 | |
| CUDA | >= 11.2 | For GPU training and full model inference |

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.6.5 - 2.15.x | ML framework |
| transformers | 4.52.4 | XLMRoberta models |
| huggingface-hub | >= 0.30, < 1 | Model download |
| scikit-learn | >= 0.22.1 | Metrics, preprocessing |
| numpy | < 2 | Compatibility requirement |
| protobuf | == 3.20.3 | Pinned for TF compatibility |
| sentencepiece | latest | Tokenization |
| PyYAML | >= 5.1.2 | Config files |

## Optional Dependencies

### Hardrules (`pip install bicleaner-ai[hardrules]`)

| Package | Version | Purpose |
|---------|---------|---------|
| bicleaner-hardrules | 2.10.8 | Rule-based filtering |
| kenlm | 7-gram support | Language model |
| cyhunspell | 2.0.3 | FastSpell dependency |

### Training (`pip install bicleaner-ai[train]`)

| Package | Purpose |
|---------|---------|
| sacremoses | Tokenization for noise |
| fuzzywuzzy | Fuzzy matching noise |
| python-Levenshtein | Fuzzy matching speed |
| zstandard | Compression |

### Transliteration (`pip install bicleaner-ai[transliterate]`)

| Package | Version | Purpose |
|---------|---------|---------|
| cyrtranslit | 1.1 | Serbo-Croatian languages |

## TensorFlow CUDA Compatibility

| TensorFlow | CUDA | cuDNN |
|------------|------|-------|
| 2.15.x | 12.2 | 8.9 |
| 2.14.x | 11.8 | 8.7 |
| 2.13.x | 11.8 | 8.6 |
| 2.12.x | 11.8 | 8.6 |
| 2.11.x | 11.2 | 8.1 |
| 2.10.x | 11.2 | 8.1 |
| 2.6.5 | 11.2 | 8.1 |

See [TensorFlow GPU support](https://www.tensorflow.org/install/source#gpu) for full matrix.

## Installation Commands

```bash
# Minimal (classify only, requires --disable_hardrules)
pip install bicleaner-ai

# With hardrules (default behavior)
pip install git+https://github.com/MSeal/cython_hunspell@2.0.3
pip install --config-settings="--build-option=--max_order=7" https://github.com/kpu/kenlm/archive/master.zip
pip install bicleaner-ai[hardrules]

# With training support
pip install bicleaner-ai[train]

# All features
pip install bicleaner-ai[all]

# Development
pip install -e ".[all]"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BICLEANER_AI_THREADS` | Number of threads | All cores |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All GPUs |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow log level | 3 (errors only) |

## Build System

| Component | Value |
|-----------|-------|
| Build backend | setuptools >= 61 |
| Package format | pyproject.toml |
| Entry points | Defined in `[project.scripts]` |

## External Services

| Service | Purpose | URL |
|---------|---------|-----|
| HuggingFace Hub | Full model hosting | huggingface.co/bitextor |
| GitHub Releases | Lite model hosting | github.com/bitextor/bicleaner-ai-data |
