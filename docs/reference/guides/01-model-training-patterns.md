# Guide 01: Model Training Patterns

> **SCOPE:** Training patterns and best practices for bicleaner-ai models. DO NOT add: full code implementations, API reference (-> manuals/).

## Principle

Bicleaner AI training uses **synthetic noise generation** to create negative samples from parallel corpora. The 1:9 positive:negative ratio is critical for classifier performance.

## Training Patterns

| Do | Don't | When |
|----|-------|------|
| Use `--parallel_train` with clean data | Train on noisy corpus | Always |
| Set `--rand_ratio 3 --womit_ratio 3 --freq_ratio 3` | Use single noise type | Default training |
| Provide `--target_word_freqs` | Skip frequency file | Using freq_ratio > 0 |
| Use `--classifier_type xlmr` for accuracy | Use xlmr on CPU | High-quality filtering |
| Use `--classifier_type dec_attention` for speed | Use dec_attention for quality | High-throughput processing |

## Data Requirements

| Model Type | Parallel Corpus Size | Monolingual | Word Frequencies |
|------------|---------------------|-------------|------------------|
| dec_attention | >= 500k pairs | Required | Required |
| transformer | >= 500k pairs | Required | Required |
| xlmr | >= 200k pairs | Not needed | Optional |

## Noise Generation Strategy

| Noise Type | Flag | Purpose | Requires |
|------------|------|---------|----------|
| Random misalignment | `--rand_ratio` | Shuffle targets | Nothing |
| Word omission | `--womit_ratio` | Remove words | Tokenizer |
| Frequency replacement | `--freq_ratio` | Replace by frequency | `--target_word_freqs` |
| Fuzzy matching | `--fuzzy_ratio` | Similar but wrong | Nothing |

## GPU Configuration

| Scenario | Configuration |
|----------|---------------|
| Single GPU | `--gpu 0` or `CUDA_VISIBLE_DEVICES=0` |
| Multi-GPU | `CUDA_VISIBLE_DEVICES=0,1` (data parallel) |
| CPU only | `CUDA_VISIBLE_DEVICES=""` (dec_attention only) |
| Mixed precision | `--mixed_precision` (reduces memory) |

## Common Issues

| Issue | Solution |
|-------|----------|
| OOM during xlmr training | Reduce `--batch_size`, use `--mixed_precision` |
| Slow training | Reduce `--steps_per_epoch`, use GPU |
| Poor accuracy | Increase corpus size, check noise ratios |
| Tokenizer errors | Provide `--source_tokenizer_command` |

## References

- Training documentation: [docs/training/README.md](../../training/README.md)
- Step-by-step guide: [docs/training/step-by-step.md](../../training/step-by-step.md)
- ADR-002: [Model Architecture](../adrs/adr-002-model-architecture.md)
