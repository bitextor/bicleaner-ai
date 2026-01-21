# ADR-001: ML Framework Selection

**Status:** Accepted
**Date:** 2022-06-01 (LREC 2022 publication)
**Decision Makers:** Jaume Zaragoza-Bernabeu, Prompsit Team

## Context

Bicleaner AI requires a deep learning framework for training and inference of neural classifiers that detect noisy sentence pairs in parallel corpora. The framework must support:
- Transformer architectures (XLMRoberta)
- Custom layers (Decomposable Attention)
- Keras-style training API
- GPU acceleration
- Model serialization

## Decision

**TensorFlow 2.x** (2.6.5 - 2.15.x) with Keras API.

## Rationale

| Factor | TensorFlow | PyTorch |
|--------|------------|---------|
| Keras API | Native, high-level | Separate (torch.nn) |
| Production deployment | TF Serving, TFLite | TorchServe |
| HuggingFace integration | Full support | Full support |
| Custom layers | Keras Layer subclass | nn.Module subclass |
| **Decision factor** | Keras simplicity for research | More Pythonic |

Key reasons:
1. **Keras API simplicity** - Training pipeline uses Keras Sequence generators
2. **Stable versioning** - TF 2.x maintains backward compatibility
3. **Integration with transformers** - HuggingFace TFAutoModel works seamlessly

## Alternatives Considered

### PyTorch
- **Pros:** More Pythonic, dynamic graphs, research community preference
- **Cons:** Would require rewriting Keras-based training pipeline
- **Verdict:** Not chosen due to existing Keras codebase

### JAX/Flax
- **Pros:** Performance, functional paradigm
- **Cons:** Smaller ecosystem, less mature for NLP
- **Verdict:** Not mature enough at decision time (2021)

## Consequences

### Positive
- Keras Sequence generators for efficient data loading
- Easy custom layer implementation (Decomposable Attention)
- Broad TensorFlow version support (2.6.5 - 2.15.x)

### Negative
- numpy<2 constraint (TensorFlow compatibility)
- protobuf pinned to 3.20.3
- CUDA version dependencies

## References

- [Bicleaner AI Paper (LREC 2022)](https://aclanthology.org/2022.lrec-1.87/)
- [TensorFlow GPU Compatibility](https://www.tensorflow.org/install/source#gpu)
