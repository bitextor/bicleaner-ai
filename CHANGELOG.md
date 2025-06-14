# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Bicleaner AI 3.3.0:
### Changed:
- Update Transformers to 4.52
- Add build methods to BCAI XLMR custom layers to load the models correctly.

## Bicleaner AI 3.2.0:
### Added:
- Option to require GPU.

## Bicleaner AI 3.1.0:
### Added:
- `bicleaner-ai-sample` command to sample training data from multiple files/language pairs.

### Changed:
- Restrict `numpy` version to <2.
 
## Bicleaner AI 3.0.1:
### Changed:
 - Update hardrules to 2.10.4.

## Bicleaner AI 3.0.0:
### Added:
- New multilingual full models: `large-en-xx`, `es-xx`, `large-es-xx`.
- Options to overwrite source or target language in model config.
- Generate training data with separated command.
- More logging messages during noise generation.
- Compress noise generation intermediate files.
- Multilingual models training documentation.
- Show a warning when no GPU/TPU has been detected.
- Show a warning if some layers do not correctly.

### Changed:
- Huge improvements in accuracy multilingual full models.
- Updated mutilingual full model `en-xx`.
- Upload multilingual full models to HF.
- Disable hardrules that need lang parameter when using multilingual model if language has not been overwritten.
- Optional `parallel_train`, `parallel_valid` and noise generation in `bicleaner_train`.
- Test classify full with HF model.
- Re-organize documentation.
- Updated Tensorflow.
- Support for Python 3.11.
- Updated Transformers to 4.36.

### Fixed:
- Always English being used in Tokenizer during noise generation.
- Noise generation failing when `freq_noise` was disabled.
- Accidentally deleting generated valid file at the end of training.

## Bicleaner AI 2.3.2:
### Fixed:
- Fix compatibility with Hardrules metadata dict.


## Bicleaner AI 2.3.1:
### Added:
- `bicleaner-ai-download` quiet mode.

### Changed:
- Update hardrules to 2.9.0, now hardrules accepts HF identifiers to load metadata.

## Bicleaner AI 2.3.0:
### Added
- Full models download from HF now accept a local path to store the model instead of using HF cache.

### Changed
- Update HuggingFace Transformers and Hub.
- Update documentation about downloading models and managing HF cache.

### Fixed
- Use a single worker to download from HF, fixes timeout errors during download.


## Bicleaner AI 2.2.2:
### Fixed
- Fix '--rules_config' parameter.

## Bicleaner AI 2.2.1:
### Added

### Changed

### Fixed
- Fix HF downloads in some slow docker instances with increased etag timeout.
- Fix builds in PIP>=23 with new Hardrules and FastSpell.

## Bicleaner AI 2.2.0:
### Added
- Support tokenizing by characters (useful for Chinese).
- CLI option to configure minimum words/tokens to be omited/replaced.

### Changed
- Refactored Tokenizer class.
- Update HF hub.
- Removed external tokenizer option.
- Improved begginers guide.

### Fixed
- Create only one Tokenizer object per process in noise function.

## Bicleaner AI 2.1.0:
### Added

### Changed
* Update Hardrules to 2.8.0
   * Better coverage of Icelandic langid
   * Updated KenLM installation instructions.

### Fixed
* KenLM installation.


## Bicleaner AI 2.0:
### Added
* Upload full models to Hugging Face Hub.
* Automatic download of full models.
* Hide Tensorflow and Transformers logging messages in executable scripts.
* Redirect Keras prediction progress bar to stderr.
* Huge memory improvements during training.
* Speed improvements using pading `longest` instead of `max_length`
* Models are more insensitive to the presence of capital letter at the start of the sentence.
* Improved performance on HBS Cyrillic transliterating in models which had poor training on cyrillic text.
* Basic test suite.
* Allow changing the base model for XLMR. Any XLMRoberta model can be used.
### Changed
* Migrate to `pyproject.toml` and `src/` tree structure, comply with PEP517, PEP518 and PEP621.
* Update to Hardrules 2.6
  * Rules can be parametrized with `--rules_config config.yaml`
  * Some rules have been refactored with better names.
  * `--run_all_rules` mode to run each rule instead of stoppping at first discard
  * Language identification with [FastSpell](https://github.com/mbanon/fastspell)
    * Better Serbo-Croatian and Slovene language detection.
  * Easier installation! Now KenLM comes pre-compiled.
* Now BICLEANER\_AI\_THREADS environment variable controls the number of threads.
* Update HF Transformers.
* Update TensorFlow minimum version.
* Removed `glove-python` dependency and use own custom compilation.
* Improved download scripts, easier to install and use.
* Set inter/intra\_op parallelism to 0 by default.
* Block size by default to 10k, a bit faster.
* Faster noise generation for small datasets with lower block size.
* Model argument can be provided with or without 'metadata.yaml'.
* Add citation info to README.
### Fixed
* Avoid generating empty sentences in omit noise.
* Restore capital letters at the beggining of the sentennce in frequency noise.
* Retrocompatibility with older models.
* Compatibility of `glove` with Python>=3.7.
* Fix loading lite models in other Python versions than 3.8.
* Fix unbound variable `lm_stats`.
* Other minor fixes.

## Bicleaner AI 1.0.1:
* Update hardrules to 1.2: adds score only mode.

## Bicleaner AI 1.0:
* Bicleaner train changes:
  * Separate most of the training logic in the BaseModel class.
  * Re-factor synthetic noise build function.
  * Parallelize synthetic noise generation.
  * Add fuzzy matching noise and neighbour noise.
  * Add Decomposable Attention model.
  * Add Transkformer-like model.
  * Add XLMRoberta model.
* Bicleaner classify changes:
  * Change old classifier by new neural models.
  * Move hardrules into a separate package.
