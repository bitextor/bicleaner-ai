# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]:
### Added
* Upload full models to Hugging Face Hub.
* Automatic download of full models.
* Hide Tensorflow and Transformers logging messages in executable scripts.
* Redirect Keras prediction progress bar to stderr.
* Huge memory improvements during training.
### Changed
* Update to Hardrules 2.3
  * Rules can be parametrized with `--rules_config config.yaml`
  * Some rules have been refactored with better names.
  * `--run_all_rules` mode to run each rule instead of stoppping at first discard
  * Language identification with [FastSpell](https://github.com/mbanon/fastspell)
  * Easier installation! Now KenLM comes pre-compiled.
* Now BICLEANER\_AI\_THREADS environment variable controls the number of threads.
* Update HF Transformers.
* Update TensorFlow minimum version.
* Rename `download-packs.sh` to `bicleaner-ai-download`.
* Set inter/intra\_op parallelism to 0 by default.
* Add citation info to README.
### Fixed
* Avoid generating empty sentences in omit noise.
* Restore capital letters at the beggining of the sentennce in frequency noise.
* Fix loading lite models in other other Python versions than 3.8.
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
