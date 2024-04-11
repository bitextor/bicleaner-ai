# Training

In case you need to train a new model (i.e. because it is not available in the language packs provided at [bicleaner-ai-data](https://github.com/bitextor/bicleaner-ai-data/releases/latest)), you can use `bicleaner-ai-train`.
`bicleaner-ai-train` is a Python tool that allows you to train a classifier which predicts
whether a pair of sentences are mutual translations or not and discards too noisy sentence pairs.

Here we will show a quick start guide, but you can take a look at the [step by step guide](step-by-step.md) for a detailed example.
Also, you can find a more advanced example that shows how to [train a multilingual model](multilingual.md).

## Quick start

### Requirements
In order to train a new model, you must provide:
* A clean parallel corpus. Recommended size of at least 500k sentence pairs for lite models and 200k for full models.
* Monolingual corpus for the source and the target language (not necessary for `xlmr` classifier).
* Gzipped list of monolingual word frequencies for the target language. You can check their format by downloading any of the available language packs.
   * The TL list of word frequencies with one entry per line. Each entry must contain the following 2 fields, split by space, in this order: word frequency (number of times a word appears in text), TL word.
   * Typically the target language is the one that is paired with English or the language that is more high resource, like `en-??`. But it can also work the other way around (e.g. `??-en''). Just make sure your target word frequencies are for the language that you set as `--target\_language`.
   * These lists can easily be obtained from a monolingual corpus and a command line in bash:
```bash
$ cat monolingual.TL \
    | sacremoses -l TL tokenize -x \
    | awk '{print tolower($0)}' \
    | tr ' ' '\n' \
    | LC_ALL=C sort | uniq -c \
    | LC_ALL=C sort -nr \ \
    | grep -v '[[:space:]]*1' \
    | gzip > wordfreq-TL.gz
```
Optionally, if you want the model to include a porn filter, you must also provide:
* File with training dataset for porn removal filter. Each sentence must contain at the beginning the `\_\_label\_\_negative` or `\_\_label\_\_positive` according to FastText convention. It should be lowercased and tokenized.

### Example

```bash
bicleaner-ai-train \
    --parallel_train corpus.en-cs \
    --parallel_dev dev.en-cs \
    --mono_train mono.en-cs \
    -m models/en-cs \
    -s en \
    -t cs \
    -F wordfreqs-cs.gz \
    --lm_file_sl models/en-cs/lm.en  --lm_file_tl models/en-cs/lm.cs \
    --porn_removal_train porn-removal.txt.en  --porn_removal_file models/en-cs/porn-model.en
```

This will train a lite model for English-Czech using the corpus `corpus.en-cs.train`, the `corpus.en-cs.dev` as development set and the monolingual corpus `mono.en-cs` to train the vocabulary embeddings.
All the model files created during training, the language model files, the porn removal file, and the `metadata.yaml` will be stored in the model directory `models/en-cs`.

To train full models you would need to use `--classifier\_type xlmr` and `--mono\_train` is not needed.

Note that, despite `-F`/`--target\_freq\_words` is not 100% mandatory, it is still required if you do not disable frequency based noise with `--freq\_ratio 0`, which is enabled by default.

## Usage
<details>
<summary>Full description of the parameter list:</summary>

```
usage: bicleaner-ai-train [-h] -m MODEL_DIR -s SOURCE_LANG -t TARGET_LANG [--mono_train MONO_TRAIN] --parallel_train PARALLEL_TRAIN --parallel_valid PARALLEL_VALID [--model_name MODEL_NAME] [--base_model BASE_MODEL] [-g GPU]
                          [--mixed_precision] [--distilled] [--seed SEED] [--generated_train GENERATED_TRAIN] [--generated_valid GENERATED_VALID] [--classifier_type {dec_attention,transformer,xlmr}] [--batch_size BATCH_SIZE]
                          [--steps_per_epoch STEPS_PER_EPOCH] [--epochs EPOCHS] [--patience PATIENCE] [-F TARGET_WORD_FREQS] [--target_tokenizer_type {word,char}] [--block_size BLOCK_SIZE] [-p PROCESSES] [--pos_ratio POS_RATIO]
                          [--rand_ratio RAND_RATIO] [--womit_ratio WOMIT_RATIO] [--freq_ratio FREQ_RATIO] [--fuzzy_ratio FUZZY_RATIO] [--neighbour_mix NEIGHBOUR_MIX] [--min_omit_words MIN_OMIT_WORDS] [--min_freq_words MIN_FREQ_WORDS]
                          [--porn_removal_train PORN_REMOVAL_TRAIN] [--porn_removal_test PORN_REMOVAL_TEST] [--porn_removal_file PORN_REMOVAL_FILE] [--porn_removal_side {sl,tl}] [--noisy_examples_file_sl NOISY_EXAMPLES_FILE_SL]
                          [--noisy_examples_file_tl NOISY_EXAMPLES_FILE_TL] [--lm_dev_size LM_DEV_SIZE] [--lm_file_sl LM_FILE_SL] [--lm_file_tl LM_FILE_TL] [--lm_training_file_sl LM_TRAINING_FILE_SL]
                          [--lm_training_file_tl LM_TRAINING_FILE_TL] [--lm_clean_examples_file_sl LM_CLEAN_EXAMPLES_FILE_SL] [--lm_clean_examples_file_tl LM_CLEAN_EXAMPLES_FILE_TL] [-q] [--debug] [--logfile LOGFILE] [-v]

options:
  -h, --help            show this help message and exit

Mandatory:
  -m MODEL_DIR, --model_dir MODEL_DIR
                        Model directory, metadata, classifier and SentencePiece vocabulary will be saved in the same directory (default: None)
  -s SOURCE_LANG, --source_lang SOURCE_LANG
                        Source language (default: None)
  -t TARGET_LANG, --target_lang TARGET_LANG
                        Target language (default: None)
  --mono_train MONO_TRAIN
                        File containing monolingual sentences of both languages shuffled together, used to train SentencePiece embeddings. Not required for XLMR. (default: None)
  --parallel_train PARALLEL_TRAIN
                        TSV file containing parallel sentences to train the classifier (default: None)
  --parallel_valid PARALLEL_VALID
                        TSV file containing parallel sentences for validation (default: None)

Options:
  --model_name MODEL_NAME
                        The name of the model. For the XLMR models it will be used as the name in Hugging Face Hub. (default: None)
  --base_model BASE_MODEL
                        The name of the base model to start from. Only used in XLMR classifiers, must be an XLMR instance. (default: None)
  -g GPU, --gpu GPU     Which GPU use, starting from 0. Will set the CUDA_VISIBLE_DEVICES. (default: None)
  --mixed_precision     Use mixed precision float16 for training (default: False)
  --distilled           Enable Knowledge Distillation training. It needs pre-built training set with raw scores from a teacher model. (default: False)
  --seed SEED           Seed for random number generation. By default, no seeed is used. (default: None)
  --generated_train GENERATED_TRAIN
                        Generated training dataset. If the file already exists the training dataset will be loaded from here. (default: None)
  --generated_valid GENERATED_VALID
                        Generated validation dataset. If the file already exists the validation dataset will be loaded from here. (default: None)
  --classifier_type {dec_attention,transformer,xlmr}
                        Neural network architecture for the classifier (default: dec_attention)
  --batch_size BATCH_SIZE
                        Batch size during classifier training. If None, default architecture value will be used. (default: None)
  --steps_per_epoch STEPS_PER_EPOCH
                        Number of batch updates per epoch during training. If None, default architecture value will be used or the full dataset size. (default: None)
  --epochs EPOCHS       Number of epochs for training. If None, default architecture value will be used. (default: None)
  --patience PATIENCE   Stop training when validation has stopped improving after PATIENCE number of epochs (default: None)
  -F TARGET_WORD_FREQS, --target_word_freqs TARGET_WORD_FREQS
                        R language gzipped list of word frequencies (needed for frequence based noise) (default: None)
  --target_tokenizer_type {word,char}
                        Type of tokenization for noise generation. (default: word)
  --block_size BLOCK_SIZE
                        Sentence pairs per block when apliying multiprocessing in the noise function (default: 2000)
  -p PROCESSES, --processes PROCESSES
                        Option no longer available, please set BICLEANER_AI_THREADS environment variable (default: None)
  --pos_ratio POS_RATIO
                        Ratio of positive samples used to oversample on validation and test sets (default: 1)
  --rand_ratio RAND_RATIO
                        Ratio of negative samples misaligned randomly (default: 3)
  --womit_ratio WOMIT_RATIO
                        Ratio of negative samples misaligned by randomly omitting words (default: 3)
  --freq_ratio FREQ_RATIO
                        Ratio of negative samples misaligned by replacing words by frequence (needs --target_word_freq) (default: 3)
  --fuzzy_ratio FUZZY_RATIO
                        Ratio of negative samples misaligned by fuzzy matching (default: 0)
  --neighbour_mix NEIGHBOUR_MIX
                        If use negative samples misaligned by neighbourhood (default: False)
  --min_omit_words MIN_OMIT_WORDS
                        Minimum words to omit per sentence in omit noise (default: 1)
  --min_freq_words MIN_FREQ_WORDS
                        Minimum words to replace per sentence in freq noise (default: 1)
  --porn_removal_train PORN_REMOVAL_TRAIN
                        File with training dataset for porn filter. Each sentence must contain at the beginning the '__label__negative' or '__label__positive' according to FastText convention. It should be lowercased and tokenized.
                        (default: None)
  --porn_removal_test PORN_REMOVAL_TEST
                        Test set to compute precision and accuracy of the porn removal filter (default: None)
  --porn_removal_file PORN_REMOVAL_FILE
                        Porn removal output file (default: porn_removal.bin)
  --porn_removal_side {sl,tl}
                        Whether the porn removal should be applied at the source or at the target language. (default: sl)
  --noisy_examples_file_sl NOISY_EXAMPLES_FILE_SL
                        File with noisy text in the SL. These are used to estimate the perplexity of noisy text. (default: None)
  --noisy_examples_file_tl NOISY_EXAMPLES_FILE_TL
                        File with noisy text in the TL. These are used to estimate the perplexity of noisy text. (default: None)
  --lm_dev_size LM_DEV_SIZE
                        Number of sentences to be removed from clean text before training LMs. These are used to estimate the perplexity of clean text. (default: 2000)
  --lm_file_sl LM_FILE_SL
                        SL language model output file. (default: None)
  --lm_file_tl LM_FILE_TL
                        TL language model output file. (default: None)
  --lm_training_file_sl LM_TRAINING_FILE_SL
                        SL text from which the SL LM is trained. If this parameter is not specified, SL LM is trained from the SL side of the input file, after removing --lm_dev_size sentences. (default: None)
  --lm_training_file_tl LM_TRAINING_FILE_TL
                        TL text from which the TL LM is trained. If this parameter is not specified, TL LM is trained from the TL side of the input file, after removing --lm_dev_size sentences. (default: None)
  --lm_clean_examples_file_sl LM_CLEAN_EXAMPLES_FILE_SL
                        File with clean text in the SL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_sl and both files must not have common sentences. This option replaces
                        --lm_dev_size. (default: None)
  --lm_clean_examples_file_tl LM_CLEAN_EXAMPLES_FILE_TL
                        File with clean text in the TL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_tl and both files must not have common sentences. This option replaces
                        --lm_dev_size. (default: None)

Logging:
  -q, --quiet           Silent logging mode (default: False)
  --debug               Debug logging mode (default: False)
  --logfile LOGFILE     Store log to a file (default: <_io.TextIOWrapper name='<stderr>' mode='w' encoding='utf-8'>)
  -v, --version         Show version of the package and exit
```
</details>


## Store generated training data
Bicleaner AI generates its own training (as well as validation) data from the parallel data provided, as explained [below](#synthetic-noise).
Using `bicleaner-ai-train` as shown in the above [exampe](#example), will generate training data for you.
But, if you want to save it for further inspection, or to run again the training in case it fails, you can provide `--generated\_train FILE` and `--generated\_valid FILE`.
If provided files are empty or do not exist, Bicleaner AI will generate the training and save it in those files.
If the files already exist, generation will be skipped and the data provided will be used.

Additionally, the data generation procedure can be run separatedly if needed, using `bicleaner-ai-generate-train` command.
Which has the same parameters as `bicleaner-ai-train` for data generation.

Here there is an example of data generation and further training:
```
bicleaner-ai-generate-train \
    -s en \
    -t cs \
    -F wordfreqs-cs.gz \
    corpus.en-cs train.en-cs

bicleaner-ai-generate-train \
    -s en \
    -t cs \
    -F wordfreqs-cs.gz \
    dev.en-cs valid.en-cs

bicleaner-ai-train \
    --generated_train train.en-cs \
    --generated_valid valid.en-cs \
    --mono_train mono.en-cs \
    -m models/en-cs \
    -s en \
    -t cs \
    --lm_file_sl models/en-cs/lm.en  --lm_file_tl models/en-cs/lm.cs \
    --porn_removal_train porn-removal.txt.en  --porn_removal_file models/en-cs/porn-model.en
```


## Synthetic noise
Bicleaner AI scoring consists of a binary classifier.
That needs positive and negative samples for training.
To achieve this, provided parallel data will be used as positive samples, and negative samples will be generated from the parallel sentences applying synthetic noise.
For each parallel sentence, several negative samples will be generated from it.
Tyipically generating a 10 to 1, or 9 to 1 ratio.
Having many counterparts for each parallel sentence, is very important for the classifier.
That way, it will be able to learn that, the target sentence for example, even if it looks similar to a possible translation of the source sentence, it should be classified as not parallel if it has missing parts.

For a description of each synthetic noise generation parameters, see: `rand\_ratio`. `womit\_ratio`, `freq\_ratio`, `fuzzy\_ratio` and `neighbour\_mix` in [](#usage).

By default the training will use `rand\_ratio`, `womit\_ratio` and `freq\_ratio` options with a value of 3.
Both `womit\_ratio` and `freq\_ratio` will use Sacremoses tokenizer by default.
So, for languages that are not supported by this tokenizer or are poorly supported, `source\_tokenizer\_command` and/or `target\_tokenizer\_command` should be provided.
Also note that, if a tokenizer command is used, the word frequencies need to be tokenized in the same way to allow noise based on frequency work correctly.

If no tokenization is available for your languages, you can disable these noise option that use tokenization and use fuzzy mathing noise: `--womit\_ratio 0 --freq\_ratio 0 --fuzzy\_ratio 6`.

`neighbour\_mix` will need your parallel sentences to appear in the same order as if they would appear in a document. Just like this:
```
I had a bad dream last night.	Tuve una pesadilla anoche.
You were training a machine translation model.	Tú estabas entrenando un modelo de traducción automática.
But you were not cleaning your data.	Pero no habías limpiado tus datos.
```
This format is quite rare in publicly available parallel corpora, so it is disabled by default.
