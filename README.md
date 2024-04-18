
# Bicleaner AI

![License](https://img.shields.io/badge/License-GPLv3-blue.svg)

Bicleaner AI (`bicleaner-ai-classify`) is a tool in Python that aims at detecting noisy sentence pairs in a parallel corpus. It
indicates the likelihood of a pair of sentences being mutual translations (with a value near to 1) or not (with a value near to 0).
Sentence pairs considered very noisy are scored with 0.

Although a training tool (`bicleaner-ai-train`) is provided, you may want to use the available ready-to-use language packages.
Please, use `bicleaner-ai-download` to download the latest language packages or visit the [Github releases](https://github.com/bitextor/bicleaner-ai-data/releases/latest) for lite models and [Hugging Face Hub](https://huggingface.co/bitextor) for full models since v2.0.
Visit our [docs](docs/TRAINING.md) for a detailed example on Bicleaner training.

If you find Bicleaner AI useful, please consider [citing us](#citation).

## What is New?
### v3.0.0 Improving Multilinguality!
New improved [multilingual models](#multilingual-models) for zero-shot classification.

<details>
<summary>Previous news</summary>

### v2.0.0, March 10, 2023
Model accuracy improvements and HF integration! See [CHANGELOG](https://github.com/bitextor/bicleaner-ai/blob/v2.0/CHANGELOG.md).

### v1.0.0, June 6 2021
Bicleaner AI is a [Bicleaner](https://github.com/bitextor/bicleaner) fork that uses neural networks.
It comes with two types of models, lite models for fast scoring and full models for high performance.
Lite models use [A Decomposable Attention Model for Natural Language Inference (Parikh et al.)](https://arxiv.org/abs/1606.01933).
Full models use fine-tuned XLMRoberta ([Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)).

The use of XLMRoberta and 1:10 positive to negative ratio were inspired in the winner of WMT20 Parallel Corpus Filtering Task paper ([Filtering noisy parallel corpus using transformers with proxy task learning](https://www.statmt.org/wmt20/pdf/2020.wmt-1.105.pdf)).

</details>

## Installation & Requirements
- Python >= 3.8
- PIP >= 23.0
- CUDA >=11.2 (for training and inference with full models)

Bicleaner AI is written in Python and can be installed using `pip`.
It also requires the [KenLM](https://github.com/kpu/kenlm) Python bindings with support for 7-gram language models.
Hardrules uses [FastSpell](https://github.com/mbanon/fastspell) that requires `cyhunspell` to be installed manually.
You can easily install all the requirements by running the following commands:

```bash
pip install bicleaner-ai git+https://github.com/MSeal/cython_hunspell@2.0.3
pip install --config-settings="--build-option=--max_order=7" https://github.com/kpu/kenlm/archive/master.zip
```

After installation, three binary files (`bicleaner-ai-train`, `bicleaner-ai-classify`, `bicleaner-ai-download`) will be located in your `python/installation/prefix/bin` directory. This is usually `$HOME/.local/bin` or `/usr/local/bin/`.

### TensorFlow
TensorFlow 2 will be installed as a dependency and GPU support is required for training.
`pip` will install latest TensorFlow supported version, but older versions `>=2.6.5` are supported and can be installed if your machine does not meet TensorFlow CUDA requirements.
See [this](https://www.tensorflow.org/install/source#gpu) table for the CUDA and TensorFlow versions compatibility.
In case you want a different TensorFlow version, you can downgrade using:
```bash
pip install tensorflow==2.6.5
```

TensorFlow logging messages are suppressed by default, in case you want to see them you have to explicitly set `TF_CPP_MIN_LOG_LEVEL` environment variable.
For example:
```bash
TF_CPP_MIN_LOG_LEVEL=0 bicleaner-ai-classify
```
**WARNING**: If you are experiencing slow downs because Bicleaner AI is not running in the GPU, you should check those logs to see if TensorFlow is loading all the libraries correctly.


### Optional requirements
For Serbo-Croatian languages, models work better with transliteration. To be able score transliterated text, install optional dependency:
```
pip install bicleaner-ai[transliterate]
```
Note that this won't transliterate the output text, it will be used only for scoring.


## Cleaning
### Getting started
`bicleaner-ai-classify` aims at detecting noisy sentence pairs in a parallel corpus. It
indicates the likelihood of a pair of sentences being mutual translations (with a value near to 1) or not (with a value near to 0). Sentence pairs considered very noisy are scored with 0.

By default, the input file (the parallel corpus to be classified) expects at least four columns, being:

* col1: URL 1
* col2: URL 2
* col3: Source sentence
* col4: Target sentence

but the source and target sentences column index can be customized by using the `--scol` and `--tcol` flags. Urls are not mandatory.

The generated output file will contain the same lines and columns that the original input file had, adding an extra column containing the Bicleaner AI classifier score.

#### Download a model
Bicleaner AI has two types of models, full and lite models.
Full models are recommended, as they provide much higher quality.
If speed is a hard constraint to you, lite models could be an option (take a look at the speed [comparison](#speed)).

See available full models [here](https://huggingface.co/models?other=bicleaner-ai) and available lite models [here](https://github.com/bitextor/bicleaner-ai-data/releases/latest).

You can download the model with:
```
bicleaner-ai-download en fr full
```
This will download `bitextor/bicleaner-ai-full-en-fr` model from HuggingFace and store it at the cache directory.

Or you can download a lite model with:
```
bicleaner-ai-download en fr lite ./bicleaner-models
```
This will download and store the en-fr lite model at `./bicleaner-models/en-fr`.

Since 2.3.0 version, full models also accept a local path to download, instead of the HF cache directory.
In that case, to use the model, provide the local path instead of the HF identifier.

To read more information about how HF cache works, please read the [official documentation](https://huggingface.co/docs/transformers/v4.30.0/en/installation#cache-setup).

#### Classifying
To classify a tab separated file containing English sentences in the first column and French sentences in the second column, use
```bash
bicleaner-ai-classify  \
        --scol 1 --tcol 2
        corpus.en-fr.tsv  \
        corpus.en-fr.classifed.tsv  \
        bitextor/bicleaner-ai-full-en-fr
```
where `--scol` and `--tcol` indicate the location of source and target sentence,
`corpus.en-fr.tsv` the input file,
`corpus.en-fr.classified.tsv` output file and `bitextor/bicleaner-ai-en-fr` is the HuggingFace model name.
Each line of the new file will contain the same content as the input file, adding a column with the score given by the Bicleaner AI classifier.

Note that, to use a lite model, you need to provide model path in your local file system, instead of HuggingFace model name.


#### Multilingual models
There are multilingual full models [available](https://huggingface.co/models?other=bicleaner-ai&search=xx).
They can work with, potentially, any language (currently only paired with English) that XLMR [supports](https://github.com/facebookresearch/fairseq/tree/main/examples/xlmr#introduction).
To see a further explaination on how to train a multilingual model or how our models perform, take a look [here](docs/training/multilingual.md) and [here](docs/training/multilingual.md#performance).

**WARNING**: multilingual models will disable hardrules that expect language parameter.
You can, however, overwrite the language code in the model configuration with `-s`/`--source_lang` or `-t`/`--target_lang` options during classify. For example when scoring English-Icelandic data, use:
```
bicleaner-ai-classify \
    --scol 1 --tcol 2 \
    -t is \
    corpus.en-is.tsv \
    corpus.en-is.classified.tsv \
    bitextor/bicleaner-ai-full-en-xx
```

### Usage

<details>
<summary>Full description of the command-line parameters:</summary>

```
usage: bicleaner-ai-classify [-h] [-s SOURCE_LANG] [-t TARGET_LANG] [-S SOURCE_TOKENIZER_COMMAND] [-T TARGET_TOKENIZER_COMMAND] [--header] [--scol SCOL] [--tcol TCOL] [-b BLOCK_SIZE] [-p PROCESSES] [--batch_size BATCH_SIZE]
                             [--tmp_dir TMP_DIR] [--score_only] [--calibrated] [--raw_output] [--lm_threshold LM_THRESHOLD] [--disable_hardrules] [--disable_lm_filter] [--disable_porn_removal] [--disable_minimal_length]
                             [--run_all_rules] [--rules_config RULES_CONFIG] [--offline] [--auth_token AUTH_TOKEN] [-q] [--debug] [--logfile LOGFILE] [-v]
                             input [output] model

positional arguments:
  input                 Tab-separated files to be classified
  output                Output of the classification (default: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>)
  model                 Path to model directory or HuggingFace Hub model identifier (such as 'bitextor/bicleaner-ai-full-en-fr')

options:
  -h, --help            show this help message and exit

Optional:
  -s SOURCE_LANG, --source_lang SOURCE_LANG
                        Overwrite model config source language (default: None)
  -t TARGET_LANG, --target_lang TARGET_LANG
                        Overwrite model config target language (default: None)
  -S SOURCE_TOKENIZER_COMMAND, --source_tokenizer_command SOURCE_TOKENIZER_COMMAND
                        Source language (SL) tokenizer full command (default: None)
  -T TARGET_TOKENIZER_COMMAND, --target_tokenizer_command TARGET_TOKENIZER_COMMAND
                        Target language (TL) tokenizer full command (default: None)
  --header              Input file will be expected to have a header, and the output will have a header as well (default: False)
  --scol SCOL           Source sentence column (starting in 1). The name of the field is expected instead of the position if --header is set (default: 3)
  --tcol TCOL           Target sentence column (starting in 1). The name of the field is expected instead of the position if --header is set (default: 4)
  -b BLOCK_SIZE, --block_size BLOCK_SIZE
                        Sentence pairs per block (default: 10000)
  -p PROCESSES, --processes PROCESSES
                        Option no longer available, please set BICLEANER_AI_THREADS environment variable (default: None)
  --batch_size BATCH_SIZE
                        Sentence pairs per block (default: 32)
  --tmp_dir TMP_DIR     Temporary directory where creating the temporary files of this program (default: /tmp)
  --score_only          Only output one column which is the bicleaner score (default: False)
  --calibrated          Output calibrated scores (default: False)
  --raw_output          Return raw output without computing positive class probability. (default: False)
  --lm_threshold LM_THRESHOLD
                        Threshold for language model fluency scoring. All TUs whose LM fluency score falls below the threshold will are removed (classifier score set to 0), unless the option --keep_lm_result set. (default: 0.5)
  --disable_hardrules   Disables the bicleaner_hardrules filtering (only bicleaner_classify is applied) (default: False)
  --disable_lm_filter   Disables LM filtering (default: False)
  --disable_porn_removal
                        Don't apply porn removal (default: False)
  --disable_minimal_length
                        Don't apply minimal length rule (default: False)
  --run_all_rules       Run all rules of Hardrules instead of stopping at first discard (default: False)
  --rules_config RULES_CONFIG
                        Hardrules configuration file (default: None)
  --offline             Don't try to download the model, instead try directly to load from local storage (default: False)
  --auth_token AUTH_TOKEN
                        Auth token for the Hugging Face Hub (default: None)

Logging:
  -q, --quiet           Silent logging mode (default: False)
  --debug               Debug logging mode (default: False)
  --logfile LOGFILE     Store log to a file (default: <_io.TextIOWrapper name='<stderr>' mode='w' encoding='utf-8'>)
  -v, --version         Show version of the package and exit
```

</details>

## Training models
Bicleaner AI provides a command-line tool to train your own model, in case available models do not fit your needs.
Please go to our [training documentation](docs/training) for a quick start and further details.

## Setting the number of threads
To set the maximum number of threads/processes to be used during training or classifying, `--processes` option is no longer available.
You will need to set `BICLEANER_AI_THREADS` environment variable to the desired value.
For example:
```
BICLEANER_AI_THREADS=12 bicleaner-ai-classify ...
```
If the variable is not set, the program will use all the available CPU cores.

## Speed
A comparison of the speed in number of sentences per second between different types of models and hardware:

| model | speed CPUx1 | speed GPUx1 |
| ----- | ----------- | ----------- |
| full | 1.78 rows/sec | 200 rows/sec |
| lite | 600 rows/sec | 10,000 rows/sec |

* CPU: Intel Core i9-9960X single core (lite model batch 16, full model batch 1)
* GPU: Nvidia V100 (lite model batch 2048, full model batch 16)

## Citation

> J. Zaragoza-Bernabeu, M. Bañón, G. Ramírez-Sánchez, S. Ortiz-Rojas, \
> "[Bicleaner AI: Bicleaner Goes Neural](https://aclanthology.org/2022.lrec-1.87/)", \
> in *Proceedings of the 13th Language Resources and Evaluation Conference*. \
> Marseille, France: Language Resources and Evaluation Conference, June 2022

```latex
@inproceedings{zaragoza-bernabeu-etal-2022-bicleaner,
    title = {"Bicleaner {AI}: Bicleaner Goes Neural"},
    author = {"Zaragoza-Bernabeu, Jaume  and
      Ram{\'\i}rez-S{\'a}nchez, Gema  and
      Ba{\~n}{\'o}n, Marta  and
      Ortiz Rojas, Sergio"},
    booktitle = {"Proceedings of the Thirteenth Language Resources and Evaluation Conference"},
    month = jun,
    year = {"2022"},
    address = {"Marseille, France"},
    publisher = {"European Language Resources Association"},
    url = {"https://aclanthology.org/2022.lrec-1.87"},
    pages = {"824--831"},
    abstract = {"This paper describes the experiments carried out during the development of the latest version of Bicleaner, named Bicleaner AI, a tool that aims at detecting noisy sentences in parallel corpora. The tool, which now implements a new neural classifier, uses state-of-the-art techniques based on pre-trained transformer-based language models fine-tuned on a binary classification task. After that, parallel corpus filtering is performed, discarding the sentences that have lower probability of being mutual translations. Our experiments, based on the training of neural machine translation (NMT) with corpora filtered using Bicleaner AI for two different scenarios, show significant improvements in translation quality compared to the previous version of the tool which implemented a classifier based on Extremely Randomized Trees."},
}
```

___

![Connecting Europe Facility](https://www.paracrawl.eu/images/logo_en_cef273x39.png)

All documents and software contained in this repository reflect only the authors' view. The Innovation and Networks Executive Agency of the European Union is not responsible for any use that may be made of the information it contains.
