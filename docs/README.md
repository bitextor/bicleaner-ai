# Bicleaner AI Documentation

> **SCOPE:** Navigation hub ONLY. Contains links to all documentation. DO NOT add: detailed content (-> specific docs), code examples (-> CLAUDE.md or specific docs).

Navigation hub for bicleaner-ai library documentation.

## Quick Links

| Document | Description |
|----------|-------------|
| [CLAUDE.md](../CLAUDE.md) | Project overview for AI assistants |
| [README.md](../README.md) | User documentation, installation, usage |
| [CHANGELOG.md](../CHANGELOG.md) | Version history |

## Architecture & Design

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Library architecture, model types, data flow |
| [tech_stack.md](tech_stack.md) | Dependencies, frameworks, compatibility |

## Training

| Document | Description |
|----------|-------------|
| [training/README.md](training/README.md) | Training quick start |
| [training/step-by-step.md](training/step-by-step.md) | Detailed training guide |
| [training/multilingual.md](training/multilingual.md) | Multilingual model training |

## Task Management

| Document | Description |
|----------|-------------|
| [tasks/README.md](tasks/README.md) | Task tracking workflow and rules |
| [tasks/kanban_board.md](tasks/kanban_board.md) | Live navigation to active tasks |

## Reference

| Document | Description |
|----------|-------------|
| [reference/README.md](reference/README.md) | Reference hub (ADRs, Guides) |
| [ADR-001: ML Framework](reference/adrs/adr-001-ml-framework.md) | TensorFlow selection rationale |
| [ADR-002: Model Architecture](reference/adrs/adr-002-model-architecture.md) | Three-tier model strategy |
| [Guide 01: Training Patterns](reference/guides/01-model-training-patterns.md) | Training best practices |
| [Testing Strategy](reference/guides/testing-strategy.md) | Testing philosophy and patterns |

## Tests

| Document | Description |
|----------|-------------|
| [tests/README.md](../tests/README.md) | Test organization and execution |

## CLI Tools

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `bicleaner-ai-classify` | `bicleaner_ai_classifier.py` | Score parallel corpus |
| `bicleaner-ai-train` | `bicleaner_ai_train.py` | Train new model |
| `bicleaner-ai-download` | `bicleaner_ai_download.py` | Download pre-trained models |
| `bicleaner-ai-download-hf` | `bicleaner_ai_download_hf.py` | Download from HuggingFace |
| `bicleaner-ai-generate-train` | `bicleaner_ai_generate_train.py` | Generate training data |
| `bicleaner-ai-sample` | `bicleaner_ai_sample.py` | Sample corpus |

## External Resources

- [GitHub Repository](https://github.com/bitextor/bicleaner-ai)
- [HuggingFace Models](https://huggingface.co/bitextor)
- [Lite Models (GitHub Releases)](https://github.com/bitextor/bicleaner-ai-data/releases/latest)
- [Citation Paper](https://aclanthology.org/2022.lrec-1.87/)
