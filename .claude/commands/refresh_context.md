---
description: Restore bicleaner-ai project context after memory loss
allowed-tools: Read
---

# Context Refresh (bicleaner-ai)

> **Scope:** Reload minimal project context for bicleaner-ai.

---

## 1) Core Files (Always Load)

- [ ] Read `CLAUDE.md` - project overview, commands, architecture
- [ ] Read `README.md` - user documentation, installation, usage
- [ ] Read `pyproject.toml` - dependencies and entry points

---

## 2) Additional by Focus Area

### Models & Architecture
- [ ] `src/bicleaner_ai/models.py` - model types (dec_attention, transformer, xlmr)
- [ ] `src/bicleaner_ai/classify.py` - classification logic

### Training
- [ ] `docs/training/README.md` - training documentation
- [ ] `src/bicleaner_ai/training.py` - training pipeline
- [ ] `src/bicleaner_ai/noise_generation.py` - synthetic negatives

### Data Processing
- [ ] `src/bicleaner_ai/datagen.py` - Keras generators
- [ ] `src/bicleaner_ai/tokenizer.py` - SentencePiece integration

---

## 3) Output After Refresh

Respond with:
1. "bicleaner-ai context loaded"
2. Project summary: parallel corpus classifier with lite/full models
3. Current task and next steps

---

## Key Project Info

| Item | Value |
|------|-------|
| Package | `bicleaner-ai` |
| Python | >= 3.8 |
| Main CLI | `bicleaner-ai-classify`, `bicleaner-ai-train`, `bicleaner-ai-download` |
| Model types | dec_attention (fast), transformer, xlmr (accurate) |
| Source | `src/bicleaner_ai/` |
| Tests | `tests/` |

---

**Last Updated:** 2026-01-21
