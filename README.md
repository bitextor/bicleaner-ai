
# Bicleaner AI

![License](https://img.shields.io/badge/License-GPLv3-blue.svg)

Bicleaner AI (`bicleaner-ai-classify`) is a tool in Python that aims at detecting noisy sentence pairs in a parallel corpus. It
indicates the likelihood of a pair of sentences being mutual translations (with a value near to 1) or not (with a value near to 0).
Sentence pairs considered very noisy are scored with 0.

Although a training tool (`bicleaner-ai-train`) is provided, you may want to use the available ready-to-use language packages.
Please, use `bicleaner-ai-download` to download the latest language packages or visit the [Github releases](https://github.com/bitextor/bicleaner-ai-data/releases/latest) for lite models and [Hugging Face Hub](https://huggingface.co/bitextor) for full models since v2.0.
Visit our [Wiki](https://github.com/bitextor/bicleaner-ai/wiki/How-to-train-your-Bicleaner-AI) for a detailed example on Bicleaner training.

## Citation
If you find Bicleaner AI useful, please consider citing the following paper:

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

## What is New?
Bicleaner AI is a [Bicleaner](https://github.com/bitextor/bicleaner) fork that uses neural networks.
It comes with two types of models, lite models for fast scoring and full models for high performance.
Lite models use [A Decomposable Attention Model for Natural Language Inference (Parikh et al.)](https://arxiv.org/abs/1606.01933).
Full models use fine-tuned XLMRoberta ([Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)).

The use of XLMRoberta and 1:10 positive to negative ratio were inspired in the winner of WMT20 Parallel Corpus Filtering Task paper ([Filtering noisy parallel corpus using transformers with proxy task learning](https://www.statmt.org/wmt20/pdf/2020.wmt-1.105.pdf)).

## Installation & Requirements
- Python >= 3.7
- TensorFlow >= 2.6.5
- CUDA 11.2 (for training and inference with full models)

Bicleaner AI is written in Python and can be installed using `pip`.
It also requires the [KenLM](https://github.com/kpu/kenlm) Python bindings with support for 7-gram language models.
You can easily install it by running the following commands:

```bash
pip install bicleaner-ai
pip install --config-settings="--build-option=--max_order=7" https://github.com/kpu/kenlm/archive/master.zip
```

Hardrules uses [FastSpell](https://github.com/mbanon/fastspell) that requires `python-dev` and `libhunspell-dev`:
```bash
sudo apt install python-dev libhunspell-dev
```

Hunspell dictionaries used by default are automatically installed.
If you need to change default configuration for language identification, see https://github.com/mbanon/fastspell#configuration.

After installation, three binary files (`bicleaner-ai-train`, `bicleaner-ai-classify`, `bicleaner-ai-download`) will be located in your `python/installation/prefix/bin` directory. This is usually `$HOME/.local/bin` or `/usr/local/bin/`.

### TensorFlow
TensorFlow 2 will be installed as a dependency and [GPU support](https://www.tensorflow.org/install/gpu) is required for training.
`pip` will install latest TensorFlow but older versions `>=2.6.5` are supported and can be installed if your machine does not meet TensorFlow CUDA requirements.
See [this](https://www.tensorflow.org/install/source#gpu) table for the CUDA and TensorFlow versions compatibility.
But current allowed versions only supoport **CUDA 11.2**.
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
### Getting Started
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
Each line of the new file will contain the same content as the input file, adding a column with the score given by the Bicleaner classifier.

Note that, to use a lite model, you need to provide model path in your local file system, instead of HuggingFace model name.


### Parameters
The complete list of parameters is:

* positional arguments:
  * `input`: Tab-separated files to be classified (default line format: `URL1 URL2 SOURCE_SENTENCE TARGET_SENTENCE [EXTRA_COLUMNS]`, tab-separated). When input is -, reads standard input.
  * `output`: Output of the classification (default: standard output). When output is -, writes standard output.
  * `metadata`: Training metadata (YAML file), generated by `bicleaner-ai-train` or downloaded as a part of a language pack.
  There's a script that can download and unpack it for you, use:
  ```bash
  $ bicleaner-ai-download en cs lite ./models
  ```
  to download English-Czech lite language pack to the ./models directory and unpack it.
* optional arguments:
  * `-h, --help`: show this help message and exit
* Optional:
  * `-S SOURCE_TOKENIZER_COMMAND`: Source language tokenizer full command (including flags if needed). If not given, Sacremoses tokenizer is used (with `escape=False` option).
  * `-T TARGET_TOKENIZER_COMMAND`: Target language tokenizer full command (including flags if needed). If not given, Sacremoses tokenizer is used (with `escape=False` option).
  * `--scol SCOL`: Source sentence column (starting in 1). If `--header` is set, the expected value will be the name of the field (default: 3 if `--header` is not set else src_text)
  * `--tcol TCOL`: Target sentence column (starting in 1). If `--header` is set, the expected value will be the name of the field (default: 4 if `--header` is not set else trg_text)
  * `--tmp_dir TMP_DIR`: Temporary directory where creating the temporary files of this program (default: default system temp dir, defined by the environment variable TMPDIR in Unix)
  * `-b BLOCK_SIZE, --block_size BLOCK_SIZE`: Sentence pairs per block (default: 10000)
  * `--lm_threshold LM_THRESHOLD`: Threshold for language model fluency scoring. All sentence pairs whose LM fluency score falls below the threshold are removed (classifier score set to 0), unless the option --keep_lm_result is set. (default: 0.5)
  * `--score_only`: Only output one column which is the bicleaner score (default: False)
  * `--calibrated`: Output calibrated scores (default: False)
  * `--raw_output`: Return raw output without computing positive class probability. (default: False)
  * `--disable_hardrules`: Disables the bicleaner_hardrules filtering (only bicleaner_classify is applied) (default: False)
  * `--disable_lm_filter`: Disables LM filtering.
  * `--disable_porn_removal`: Disables porn removal.
  * `--disable_minimal_length` : Don't apply minimal length rule (default: False).

* Logging:
  * `-q, --quiet`: Silent logging mode (default: False)
  * `--debug`: Debug logging mode (default: False)
  * `--logfile LOGFILE`: Store log to a file (default: \<\_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'\>)
  * `-v, --version`: show version of this script and exit

## Training classifiers

In case you need to train a new classifier (i.e. because it is not available in the language packs provided at [bicleaner-ai-data](https://github.com/bitextor/bicleaner-ai-data/releases/latest)), you can use `bicleaner-ai-train`.
`bicleaner-ai-train` is a Python tool that allows you to train a classifier which predicts
whether a pair of sentences are mutual translations or not and discards too noisy sentence pairs. Visit our [Wiki](https://github.com/bitextor/bicleaner-ai/wiki/How-to-train-your-Bicleaner-AI) for a detailed example on Bicleaner AI training.

### Requirements

In order to train a new classifier, you must provide:
* A clean parallel corpus (500k pairs of sentences is the recommended size).
* Monolingual corpus for the source and the target language (not necessary for `xlmr` classifier).
* Gzipped lists of monolingual word frequencies. You can check their format by downloading any of the available language packs.
   * The SL list of word frequencies with one entry per line. Each entry must contain the following 2 fields, split by space, in this order: word frequency (number of times a word appears in text), SL word.
   * The TL list of word frequencies with one entry per line. Each entry must contain the following 2 fields, split by space, in this order: word frequency (number of times a word appears in text), TL word.
   * These lists can easily be obtained from a monolingual corpus and a command line in bash:
```bash
$ cat monolingual.SL \
    | sacremoses -l SL tokenize -x \
    | awk '{print tolower($0)}' \
    | tr ' ' '\n' \
    | LC_ALL=C sort | uniq -c \
    | LC_ALL=C sort -nr \ \
    | grep -v '[[:space:]]*1' \
    | gzip > wordfreq-SL.gz
$ cat monolingual.TL \
    | sacremoses -l TL tokenize -x \
    | awk '{print tolower($0)}' \
    | tr ' ' '\n' \
    | LC_ALL=C sort | uniq -c \
    | LC_ALL=C sort -nr \ \
    | grep -v '[[:space:]]*1' \
    | gzip > wordfreq-TL.gz

```
Optionally, if you want the classifier to include a porn filter, you must also provide:
* File with training dataset for porn removal classifier. Each sentence must contain at the beginning the `__label__negative` or `__label__positive` according to FastText convention. It should be lowercased and tokenized.

### Parameters
It can be used as follows.

```bash
bicleaner-ai-train [-h]
    -m MODEL_DIR
    -s SOURCE_LANG
    -t TARGET_LANG
    [--mono_train MONO_TRAIN]
    --parallel_train PARALLEL_TRAIN
    --parallel_dev PARALLEL_DEV
    [-S SOURCE_TOKENIZER_COMMAND]
    [-T TARGET_TOKENIZER_COMMAND]
    [-F TARGET_WORD_FREQS]
    [--block_size BLOCK_SIZE]
    [-p PROCESSES]
    [-g GPU]
    [--mixed_precision]
    [--save_train_data SAVE_TRAIN_DATA]
    [--distilled]
    [--seed SEED]
    [--classifier_type {dec_attention,transformer,xlmr}]
    [--batch_size BATCH_SIZE]
    [--steps_per_epoch STEPS_PER_EPOCH]
    [--epochs EPOCHS]
    [--patience PATIENCE]
    [--pos_ratio POS_RATIO]
    [--rand_ratio RAND_RATIO]
    [--womit_ratio WOMIT_RATIO]
    [--freq_ratio FREQ_RATIO]
    [--fuzzy_ratio FUZZY_RATIO]
    [--neighbour_mix NEIGHBOUR_MIX]
    [--porn_removal_train PORN_REMOVAL_TRAIN]
    [--porn_removal_test PORN_REMOVAL_TEST]
    [--porn_removal_file PORN_REMOVAL_FILE]
    [--porn_removal_side {sl,tl}]
    [--noisy_examples_file_sl NOISY_EXAMPLES_FILE_SL]
    [--noisy_examples_file_tl NOISY_EXAMPLES_FILE_TL]
    [--lm_dev_size LM_DEV_SIZE]
    [--lm_file_sl LM_FILE_SL]
    [--lm_file_tl LM_FILE_TL]
    [--lm_training_file_sl LM_TRAINING_FILE_SL]
    [--lm_training_file_tl LM_TRAINING_FILE_TL]
    [--lm_clean_examples_file_sl LM_CLEAN_EXAMPLES_FILE_SL]
    [--lm_clean_examples_file_tl LM_CLEAN_EXAMPLES_FILE_TL]
    [-q]
    [--debug]
    [--logfile LOGFILE]
```

* positional arguments:
  * `input`: Tab-separated bilingual input file (default: Standard input)(line format: SOURCE_SENTENCE TARGET_SENTENCE, tab-separated)
* optional arguments:
  * `-h, --help`: show this help message and exit
* Mandatory:
  * `-m MODEL_DIR, --model_dir MODEL_DIR`: Model directory, metadata, classifier and SentencePiece models will be saved in the same directory (default: None)
  * `-s SOURCE_LANG, --source_lang SOURCE_LANG`: Source language (default: None)
  * `-t TARGET_LANG, --target_lang TARGET_LANG`: Target language (default: None)
  * `--mono_train MONO_TRAIN`: File containing monolingual sentences of both languages shuffled together, used to train SentencePiece embeddings. Not required for XLMR. (default: None)
  * `--parallel_train PARALLEL_TRAIN`: TSV file containing parallel sentences to train the classifier (default: None)
  * `--parallel_dev PARALLEL_DEV`: TSV file containing parallel sentences for development (default: None)

* Options:
  * `-S SOURCE_TOKENIZER_COMMAND, --source_tokenizer_command SOURCE_TOKENIZER_COMMAND`: Source language tokenizer full command (default: None)
  * `-T TARGET_TOKENIZER_COMMAND, --target_tokenizer_command TARGET_TOKENIZER_COMMAND`: Target language tokenizer full command (default: None)
  * `-F TARGET_WORD_FREQS, --target_word_freqs TARGET_WORD_FREQS`: R language gzipped list of word frequencies (needed for frequence based noise) (default: None)
  * `--block_size BLOCK_SIZE`: Sentence pairs per block when apliying multiprocessing in the noise function (default: 10000)
  * `-p PROCESSES, --processes PROCESSES`: Number of process to use (default: 71)
  * `-g GPU, --gpu GPU`: Which GPU use, starting from 0. Will set the CUDA_VISIBLE_DEVICES. (default: None)
  * `--mixed_precision`: Use mixed precision float16 for training (default: False)
  * `--save_train_data SAVE_TRAIN_DATA`: Save the generated dataset into a file. If the file already exists the training dataset will be loaded from there. (default: None)
  * `--distilled`: Enable Knowledge Distillation training. It needs pre-built training set with raw scores from a teacher model. (default: False)
  * `--seed`: SEED           Seed for random number generation. By default, no seeed is used. (default: None)
  * `--classifier_type {dec_attention,transformer,xlmr}`: Neural network architecture of the classifier (default: dec_attention)
  * `--batch_size BATCH_SIZE`: Batch size during classifier training. If None, default architecture value will be used. (default: None)
  * `--steps_per_epoch STEPS_PER_EPOCH`: Number of batch updates per epoch during training. If None, default architecture value will be used or the full dataset size. (default: None)
  * `--epochs EPOCHS`: Number of epochs for training. If None, default architecture value will be used. (default: None)
  * `--patience PATIENCE`: Stop training when validation has stopped improving after PATIENCE number of epochs (default: None)
  * `--pos_ratio POS_RATIO`: Ratio of positive samples used to oversample on validation and test sets (default: 1)
  * `--rand_ratio RAND_RATIO`: Ratio of negative samples misaligned randomly (default: 3)
  * `--womit_ratio WOMIT_RATIO`: Ratio of negative samples misaligned by randomly omitting words (default: 3)
  * `--freq_ratio FREQ_RATIO`: Ratio of negative samples misaligned by replacing words by frequence (needs --target_word_freq) (default: 3)
  * `--fuzzy_ratio FUZZY_RATIO`: Ratio of negative samples misaligned by fuzzy matching (default: 0)
  * `--neighbour_mix NEIGHBOUR_MIX`: If use negative samples misaligned by neighbourhood (default: False)
  * `--porn_removal_train PORN_REMOVAL_TRAIN`: File with training dataset for FastText classifier. Each sentence must contain at the beginning the '__label__negative' or '__label__positive' according to FastText convention. It should be lowercased and tokenized. (default: None)
  * `--porn_removal_test PORN_REMOVAL_TEST`: Test set to compute precision and accuracy of the porn removal classifier (default: None)
  * `--porn_removal_file PORN_REMOVAL_FILE`: Porn removal classifier output file (default: porn_removal.bin)
  * `--porn_removal_side {sl,tl}`: Whether the porn removal should be applied at the source or at the target language. (default: sl)
  * `--noisy_examples_file_sl NOISY_EXAMPLES_FILE_SL`: File with noisy text in the SL. These are used to estimate the perplexity of noisy text. (default: None)
  * `--noisy_examples_file_tl NOISY_EXAMPLES_FILE_TL`: File with noisy text in the TL. These are used to estimate the perplexity of noisy text. (default: None)
  * `--lm_dev_size LM_DEV_SIZE`: Number of sentences to be removed from clean text before training LMs. These are used to estimate the perplexity of clean text. (default: 2000)
  * `--lm_file_sl LM_FILE_SL`: SL language model output file. (default: None)
  * `--lm_file_tl LM_FILE_TL`: TL language model output file. (default: None)
  * `--lm_training_file_sl LM_TRAINING_FILE_SL`: SL text from which the SL LM is trained. If this parameter is not specified, SL LM is trained from the SL side of the input file, after removing --lm_dev_size sentences. (default: None)
  * `--lm_training_file_tl LM_TRAINING_FILE_TL`: TL text from which the TL LM is trained. If this parameter is not specified, TL LM is trained from the TL side of the input file, after removing --lm_dev_size sentences. (default: None)
  * `--lm_clean_examples_file_sl LM_CLEAN_EXAMPLES_FILE_SL`: File with clean text in the SL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_sl and both files must not have common sentences. This option replaces --lm_dev_size. (default: None)
  * `--lm_clean_examples_file_tl LM_CLEAN_EXAMPLES_FILE_TL`: File with clean text in the TL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_tl and both files must not have common sentences. This option replaces --lm_dev_size. (default: None)
* Logging:
  * `-q, --quiet`: Silent logging mode (default: False)
  * `--debug`: Debug logging mode (default: False)
  * `--logfile LOGFILE`: Store log to a file (default: \<\_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>)

### Example

```bash
bicleaner-ai-train \
          --parallel_train corpus.en-cs.train \
          --parallel_dev corpus.en-cs.dev \
          --mono_train mono.en-cs \
          -m models/en-cs \
          -s en \
          -t cs \
          -F wordfreqs-cs.gz \
          --lm_file_sl models/en-cs/lm.en  --lm_file_tl models/en-cs/lm.cs \
          --porn_removal_train porn-removal.txt.en  --porn_removal_file models/en-cs/porn-model.en \
```

This will train a lite classifier for English-Czech using the corpus `corpus.en-cs.train`, the `corpus.en-cs.dev` as development set and the monolingual corpus `mono.en-cs` to train the vocabulary embeddings.
All the model files created during training, the language model files, the porn removal file, and the `metadata.yaml` will be stored in the model directory `models/en-cs`.

To train full models you would need to use `--classifier_type xlmr` and `--mono_train` is not needed.

### Synthetic noise
By default the training will use `rand_ratio`, `womit_ratio` and `freq_ratio` options with a value of 3.
Both `womit_ratio` and `freq_ratio` will use Sacremoses tokenizer by default.
So, for languages that are not supported by this tokenizer or are poorly supported, `source_tokenizer_command` and/or `target_tokenizer_command` should be provided.
Also note that, if a tokenizer command is used, the word frequencies need to be tokenized in the same way to allow noise based on frequency work correctly.

If no tokenization is available for your languages, you can disable these noise option that use tokenization and use fuzzy mathing noise: `--womit_ratio 0 --freq_ratio 0 --fuzzy_ratio 6`.

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
___

![Connecting Europe Facility](https://www.paracrawl.eu/images/logo_en_cef273x39.png)

All documents and software contained in this repository reflect only the authors' view. The Innovation and Networks Executive Agency of the European Union is not responsible for any use that may be made of the information it contains.
