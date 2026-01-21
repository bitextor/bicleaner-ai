# Architecture

> **SCOPE:** Library architecture and design. Contains model types, module structure, data flow. DO NOT add: dependency versions (-> tech_stack.md), training guides (-> training/), CLI usage (-> CLAUDE.md).

Bicleaner AI library architecture and design.

## Overview

Bicleaner AI is a parallel corpus classifier that scores sentence pairs (0-1) indicating whether they are mutual translations.

```
Input (TSV)          Model              Output (TSV + score)
+---------------+    +------------+    +------------------+
| src | tgt     | -> | Classifier | -> | src | tgt | 0.95 |
| Hello | Hola  |    +------------+    | Hello | Hola     |
+---------------+                      +------------------+
```

## Model Types

| Type | Architecture | Speed (CPU) | Speed (GPU) | Use Case |
|------|--------------|-------------|-------------|----------|
| `dec_attention` | Decomposable Attention | ~600 sent/sec | ~10k sent/sec | Fast processing, lite models |
| `transformer` | Transformer | Medium | Medium | Balanced |
| `xlmr` | XLMRoberta | ~1.78 sent/sec | ~200 sent/sec | High accuracy, multilingual |

All models implement `ModelInterface`:
- `get_generator()` - Returns data generator
- `predict()` - Score sentence pairs
- `load()` - Load trained model
- `train()` - Train new model

## Module Structure

```
src/bicleaner_ai/
+-- CLI Tools
|   +-- bicleaner_ai_classifier.py   # Main classification entry point
|   +-- bicleaner_ai_train.py        # Training entry point
|   +-- bicleaner_ai_download.py     # Model download
|   +-- bicleaner_ai_download_hf.py  # HuggingFace download
|   +-- bicleaner_ai_generate_train.py
|   +-- bicleaner_ai_sample.py
|
+-- Core
|   +-- classify.py                  # Classification logic
|   +-- training.py                  # Training pipeline
|   +-- models.py                    # Model factory, ModelInterface
|
+-- Models
|   +-- decomposable_attention.py    # Decomposable Attention (Parikh et al.)
|   +-- layers.py                    # Custom Keras layers
|   +-- losses.py                    # Loss functions
|   +-- metrics.py                   # Evaluation metrics
|
+-- Data
|   +-- datagen.py                   # Keras Sequence generators
|   +-- tokenizer.py                 # SentencePiece integration
|   +-- noise_generation.py          # Synthetic negative samples
|   +-- calibrate.py                 # Output calibration
|
+-- Utils
    +-- util.py                      # Utilities
    +-- word_freqs_*.py              # Word frequency handling
```

## Data Flow

### Classification

```
1. Input TSV (--scol, --tcol)
       |
2. Tokenization (SentencePiece / Model tokenizer)
       |
3. Batch Processing (--batch_size, --block_size)
       |
4. Model Inference
       |
5. Optional: Hardrules filtering
       |
6. Optional: LM filter, Porn filter
       |
7. Output TSV + Score column
```

### Training

```
1. Parallel corpus + validation set
       |
2. Noise generation (synthetic negatives, 1:9 ratio)
   - Random misalignment (--rand_ratio)
   - Word omission (--womit_ratio)
   - Frequency-based replacement (--freq_ratio)
       |
3. Model training (TensorFlow/Keras)
       |
4. Optional: Porn filter (FastText)
       |
5. Optional: LM filter (KenLM)
       |
6. Calibration
       |
7. Model artifacts + metadata.yaml
```

## Key Design Decisions

See [ADR-001: ML Framework](reference/adrs/adr-001-ml-framework.md) and [ADR-002: Model Architecture](reference/adrs/adr-002-model-architecture.md) for detailed rationale.

**Summary:**
- TensorFlow for Keras API simplicity
- Three-tier model strategy (speed vs accuracy trade-off)
- 1:9 positive:negative ratio (WMT20 winner approach)
- Synthetic noise generation (no manual labeling)

## Dependencies

See [tech_stack.md](tech_stack.md) for complete dependency list and CUDA compatibility.

## Configuration

Models store configuration in `metadata.yaml`:
- Source/target languages
- Classifier type
- Training parameters
- Optional filter paths (LM, porn removal)
