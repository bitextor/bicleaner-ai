# Testing Strategy

> **SCOPE:** Universal testing philosophy for bicleaner-ai. Contains test principles, levels, and patterns. DO NOT add: specific test implementations (-> tests/), framework configs (-> pyproject.toml).

## Testing Philosophy

**Core Principle:** Test YOUR code, not frameworks.

For bicleaner-ai (ML library), this means:
- Test classification logic, not TensorFlow internals
- Test training pipeline, not HuggingFace transformers
- Test data generators, not Keras Sequence base class

## Test Levels

| Level | Count | Purpose | Example |
|-------|-------|---------|---------|
| **Integration** | 2-5 | Model training + inference | `test_train_lite`, `test_train_full` |
| **Unit** | 5-15 | Individual functions | `unit_test.py` |

**Note:** bicleaner-ai is a library, not a web service. No E2E tests needed.

## Risk-Based Testing

| Priority | Risk Level | Testing Requirement |
|----------|------------|---------------------|
| >= 15 | Critical | MUST be tested |
| 10-14 | High | Should be tested |
| < 10 | Low | Optional testing |

### Critical Paths for bicleaner-ai

| Component | Risk | Priority |
|-----------|------|----------|
| Model inference (classify.py) | Data corruption | 20 |
| Training pipeline (training.py) | Model quality | 18 |
| Noise generation | Training data quality | 15 |
| Tokenization | Encoding issues | 12 |

## Test Organization

```
tests/
+-- bicleaner_ai_test.py    # Integration tests (training, classification)
+-- unit_test.py            # Unit tests
+-- corpus.en-fr            # Test data: parallel corpus
+-- dev.en-fr               # Test data: validation set
+-- mono.en-fr              # Test data: monolingual
+-- wordfreq-fr.gz          # Test data: word frequencies
+-- manual/                 # Manual test scripts
    +-- results/            # Test outputs (gitignored)
```

## Testing Patterns

### Arrange-Act-Assert

```python
def test_classify():
    # Arrange
    model = load_model(model_path)
    sentences = [("Hello", "Hola")]

    # Act
    scores = model.predict(sentences)

    # Assert
    assert 0 <= scores[0] <= 1
```

### Test Data Builders

Use existing test corpus files:
- `corpus.en-fr` - training data
- `dev.en-fr` - validation data
- `test.en-fr` - test data

### GPU/CPU Testing

| Test | Requirement | Flag |
|------|-------------|------|
| `test_train_lite` | CPU only | Default |
| `test_train_full` | GPU required | `CUDA_VISIBLE_DEVICES` |

## What to Test

| Test | Why |
|------|-----|
| Score range (0-1) | Output contract |
| Model loading | File format compatibility |
| Noise generation ratios | Training data quality |
| Tokenization edge cases | Unicode handling |

## What NOT to Test

| Don't Test | Why |
|------------|-----|
| TensorFlow ops | Framework's responsibility |
| HuggingFace models | External library |
| SentencePiece encoding | External library |

## Common Issues

| Issue | Solution |
|-------|----------|
| Flaky tests | Use fixed random seeds (`--seed`) |
| Slow tests | Reduce corpus size for unit tests |
| GPU tests fail on CI | Skip with `@pytest.mark.skipif` |
| Memory issues | Use smaller batch sizes |

## Coverage Guidelines

| Module | Target |
|--------|--------|
| classify.py | 80% |
| training.py | 70% |
| models.py | 75% |
| noise_generation.py | 70% |

## Maintenance

**Last Updated:** 2026-01-21

**Update Triggers:**
- Test framework changes
- New model architectures added
- Training pipeline changes

**Verification:**
- [ ] All test examples follow pytest syntax
- [ ] Coverage targets reflect current code
- [ ] GPU test requirements documented
