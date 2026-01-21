# ADR-002: Model Architecture Strategy

**Status:** Accepted
**Date:** 2022-06-01 (LREC 2022 publication)
**Decision Makers:** Jaume Zaragoza-Bernabeu, Prompsit Team

## Context

Bicleaner AI needs classifiers for scoring parallel sentence pairs. Different use cases require different speed/accuracy trade-offs:
- High-throughput processing (millions of sentences)
- High-accuracy filtering (quality over speed)
- Multilingual support (100+ languages)

## Decision

**Three-tier model architecture:**

| Model | Architecture | Speed (CPU) | Speed (GPU) | Use Case |
|-------|--------------|-------------|-------------|----------|
| `dec_attention` | Decomposable Attention | ~600 sent/sec | ~10k sent/sec | Fast processing |
| `transformer` | Transformer encoder | Medium | Medium | Balanced |
| `xlmr` | XLMRoberta fine-tuned | ~1.78 sent/sec | ~200 sent/sec | High accuracy |

## Rationale

### Decomposable Attention (Lite Models)
- Based on [Parikh et al. 2016](https://arxiv.org/abs/1606.01933)
- Custom Keras implementation with attention mechanism
- Trained from scratch with SentencePiece tokenization
- **Trade-off:** Speed over accuracy

### XLMRoberta (Full Models)
- Based on [Conneau et al. 2019](https://arxiv.org/abs/1911.02116)
- Fine-tuned on parallel corpus classification task
- 100+ languages supported out of the box
- **Trade-off:** Accuracy over speed

### 1:9 Positive:Negative Ratio
- Inspired by [WMT20 winner](https://www.statmt.org/wmt20/pdf/2020.wmt-1.105.pdf)
- Synthetic noise generation creates diverse negatives
- Improves classifier discrimination ability

## Alternatives Considered

### Single Model Architecture
- **Pros:** Simpler codebase, one model to maintain
- **Cons:** Cannot satisfy both speed and accuracy requirements
- **Verdict:** Not flexible enough for diverse use cases

### BERT Instead of XLMRoberta
- **Pros:** Smaller model, faster inference
- **Cons:** English-centric, poor multilingual support
- **Verdict:** XLMRoberta better for multilingual corpora

### mBERT
- **Pros:** Multilingual, smaller than XLMRoberta
- **Cons:** Lower accuracy on translation quality tasks
- **Verdict:** XLMRoberta outperforms on benchmarks

## Consequences

### Positive
- Users choose model based on their speed/accuracy needs
- Multilingual support without language-specific training
- Lite models work well on CPU-only environments

### Negative
- Three model types increase maintenance complexity
- Different training procedures per architecture
- Full models require GPU for practical inference

## References

- [Decomposable Attention (Parikh et al.)](https://arxiv.org/abs/1606.01933)
- [XLMRoberta (Conneau et al.)](https://arxiv.org/abs/1911.02116)
- [WMT20 Parallel Corpus Filtering](https://www.statmt.org/wmt20/pdf/2020.wmt-1.105.pdf)
