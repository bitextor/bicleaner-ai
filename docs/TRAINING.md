# How To Train Your Bicleaner AI

## Intro

In this article we'll develop an example to illustrate the recommended way to train Bicleaner AI from scratch.
Of course you can follow your own way, but let us unveil our secrets on Bicleaner AI training
(trust us, we have done this a zillion times before).

If after reading this guide you are still having questions or needing clarification, please don't hesitate to open a new [issue](https://github.com/bitextor/bicleaner-ai/issues).

Let's assume you'd like to train a Bicleaner for English-Icelandic (en-is).

## What can you train
Bicleaner AI has mainly two types of models, lite models and full models.
Lite models (`dec_attention`) are small and fast, well suited if you don't have GPU's for cleaning or if you want to clean a very large corpus.
Full models (`xlmr`) are large and slow, recommended if you can afford GPU's, because they are much more precise.

## What you will need

* GPU's are needed for training, especially for full models.
* As much as monolingual for both languages as you can, about 5M sentences each is nice (needed only for lite models).
* Word frequencies file for the target language, in this case Icelandic (optional).
* A porn-annotated monolingual dataset (`__label__negative`/`__label__positive` sentences) for English (or Icelandic) (optional).
* A training corpus.
* A development set made of very clean parallel sentences.

### What is recommended
* For the training corpus a size of 600K sentences that come from different corpora and are as clean as possible.
* For the development set about 2K sentences of vey clean corpus that don't come from the same sources as the training corpus.
* The word frequencies is recommended in order to use the frequency based synthetic noise.

## What you will get
* An English-Icelandic classifier.
* A fluency filter for English.
* A fluency filter for Icelandic.
* A monolingual model of porn removal for English (or Icelandic).
* A yaml file with metadata.

If you already have all the ingredients (parallel corpus and monolingual data) beforehand, you won't need to do anything else before running the training command. If not, don't worry: below we'll show you how to get them all.

# Data preparation

## Starting point: a parallel corpus

Good news: You can build everything needed to train Bicleaner from a single parallel corpus.

* If you don't have monolingual data you can use each of the sides of your parallel corpus.

```bash
cut -f1 corpus.en-is > mono.en
cut -f2 corpus.en-is > mono.is
```

* If you have TMXs, you can convert them to plain text by using [tmxt](https://github.com/sortiz/tmxt):

```bash
python3.7 tmxt/tmxt.py --codelist en,is smallcorpus.en-is.tmx smallcorpus.en-is.txt
```

* If your corpora happens to be pre-tokenized (it happens sometimes when downloading from Opus), you need to detokenize:

```bash
cut -f1 smallcorpus.en-is.txt > smallcorpus.en-is.en
cut -f2 smallcorpus.en-is.txt > smallcorpus.en-is.is
sacremoses -l en < smallcorpus.en-is.en > smallcorpus.en-is.detok.en
sacremoses -l is < smallcorpus.en-is.is > smallcorpus.en-is.detok.is
paste smallcorpus.en-is.is  smallcorpus.en-is.detok.is > smallcorpus.en-is
```

* If you do not have enough sentences in your source or target languages, you can try translating from another language by using [Apertium](https://github.com/apertium). For example, if you want to translate an English-Swedish corpus for English-Icelandic:

```bash
cut -f1 corpus.en-sv > corpus.en-sv.en
cut -f2 corpus.en-sv > corpus.en-sv.sv
cat corpus.en-sv.sv | apertium-destxt -i | apertium -f none -u swe-isl | apertium-retxt > corpus.en-is.is
paste corpus.en-sv.en corpus.en-is.is > corpus.en-is
```

## Word frequency file

A word frequency file is recommended in order to use synthetic word frequency based noise when training. The format for this file is:

```
freq1 word1
freq2 word2
...
```
To build it, you just need to count the numbers of times a given words appears in a corpus. For this, you need a big monolingual corpus for each source language and target language.

```bash
$ cat mono.is \
    | sacremoses -l is tokenize -x \
    | awk '{print tolower($0)}' \
    | tr ' ' '\n' \
    | LC_ALL=C sort | uniq -c \
    | LC_ALL=C sort -nr \
    | grep -v "[[:space:]]*1" \
    | gzip > wordfreq-is.gz
```
Remember to tokenize with the same method you use in the rest of the process!

## Porn-annotated monolingual dataset

An optional feature of Bicleaner since version 0.14 is filtering out sentences containing porn. If you don't want to remove these kind of sentences, this dataset is not required.
If you want to use the porn removal and one of your languages is English, you can borrow the training file included in the [language packs](https://github.com/bitextor/bicleaner-data/releases) and skip this step.

In order to train this feature, an annotated dataset for porn must be provided, containing around 200K sentences. Each sentence must contain at the beginning the `__label__negative` or `__label__positive` according to [FastText](https://fasttext.cc/) convention. It should be lowercased and tokenized.

More elaborated strategies can be choosen but, for a naive approach, sentences containing "porny" words from the English side of your corpus can be selected by simply using `grep` (around 200K sentences is enough to train):
```
cat mono.en | grep -i "pornword1" | grep -i "pornword2" | ... | grep -i "pornwordn"  \
                 | awk '{if (toupper($0) != tolower($0)) print tolower($0);}'  > positive.lower.en.txt
```
In the same fashion, "safe" negative examples can be extracted using inverse `grep`:
```
cat mono.en | grep -iv "pornword1" | grep -iv "pornword2" | ... | grep -iv "pornwordn" \
                 | awk '{if (toupper($0) != tolower($0)) print tolower($0);}'  > negative.lower.en.txt
```
(a small `awk` filter is added to avoid having sentences not containing alphabetic characters)

Once you have obtained the positive and negative porn sentences, they need to be tokenized, and the label added:

```
cat positive.lower.en.txt | sacremoses -l en tokenize -x  \
                          | LC_ALL=C sort | uniq  \
                          | awk '{print "__label__positive "$0}'| shuf -n 200000 > positive.dedup.lower.tok.en.txt
cat negative.lower.en.txt | sacremoses -l en tokenize -x  \
                          | LC_ALL=C sort | uniq  \
                          | awk '{print "__label__negative "$0}' | shuf -n 200000 > negative.dedup.lower.tok.en.txt
```
Finally, they just need to be joined in a single file:

```
cat positive.dedup.lower.tok.en.txt negative.dedup.lower.tok.en.txt | shuf >  porn-annotated.txt.en
```
## Training corpus
If you have a super clean parallel corpus, containing around 500K parallel sentences, you can skip this part. 
If not, you can build a cleaner corpus from a not-so-clean parallel corpus by using [Bifixer](https://github.com/bitextor/bifixer) and the Bicleaner Hardrules.

First, apply Bifixer:
```bash
python3.7 bifixer/bifixer/bifixer.py --scol 1 --tcol 2 --ignore_duplicates corpus.en-is corpus.en-is.bifixed en is
```
Then, apply the hardrules:

```bash
bicleaner-hardrules corpus.en-is.bifixed corpus.en-is.annotated \
   -s en -t is --scol 1 --tcol 2 \
   --annotated_output --disable_mininal_length \
```

If any of your source or target languages is easily mistaken with other similar languages (for example, Norwegian and Danish, Galician and Portuguese...), you may need to use the `--disable_lang_ident` when running Hardrules. You can detect if this is happening by running:

```bash
cat corpus.en-is.annotated | awk -F'\t' '{print $4}' | sort | uniq -c | sort -nr
```

If language-related annotations are high (`c_different_language`,  `c_reliable_long_language(right, targetlang)`and/or `c_reliable_long_language(left, sourcelang)`): you are probably experiencing this issue (so you _really want_ to use the `--disable_lang_ident` flag also for training)

Once you have an annotated version of your corpus, you can get the cleaner parallel sentences and use these as a training corpus (100K sentences is a good number):

```bash
cat corpus.en-is.annotated  | grep "keep$" |  shuf -n 100000 | cut -f1,2 > trainingcorpus.en-is
```


# Train Bicleaner AI

## Lite model
To train a lite model you will need to concatenate the monolingual data of both languages into a single file.
Keep in mind that the vocabulary is trained from this data, so it is recommended to use a similar amount of data for wach of the sentences.
If you have very different sizes you should `head` the large one and then concatenate:

```bash
head -5000000 mono.en | cat - mono.is > mono.en-is
```

```bash
bicleaner-ai-train \
    --classifier_type dec_attention \
    -m models/en-is -s en -t is \
    -F models/en-is/wordfreq-is.gz \
    --mono_train mono.en-is
    --parallel_train corpus.en-is \
    --parallel_dev dev.en-is \
    --lm_file_sl model.en-is.en --lm_file_tl model.en-is.is \
    --porn_removal_train porn-annotated.txt.en \
    --porn_removal_file porn-model.en
```
Remember to check in the [Readme](https://github.com/bitextor/bicleaner/#parameters-1) all the available options and choose those that are the most useful for you.

### Full model
For the full model no monolingual data is needed.
The default batch size for the full model is 16 to fit a single GPU with 11GB of memory, but at least 4 GPU's and a batch size of 128 is advised in order to get a proper stable training.

```bash
bicleaner-ai-train \
    --classifier_type xlmr \
    --batch_size 128 \
    --steps_per_epoch 2000 \
    --epochs 15 --patience 4 \
    -m models/en-is -s en -t is \
    -F models/en-is/wordfreq-is.gz \
    --parallel_train corpus.en-is \
    --parallel_dev dev.en-is \
    --lm_file_sl model.en-is.en --lm_file_tl model.en-is.is \
    --porn_removal_train porn-annotated.txt.en \
    --porn_removal_file porn-model.en
```

#  Bicleaning a corpus:
At this point, you probably want to try your freshly trained Bicleaner to clean an actual corpus. Just run:

```bash
bicleaner-ai-classify testcorpus.en-is testcorpus.en-is.classified models/en-is/metadata.yaml --scol 1 --tcol 2
```

After running Bicleaner, you'll have a new file (`testcorpus.en-is.classified`), having the same content as the input file (`testcorpus.en-is`) plus an extra column. This new column contains the scores given by the classifier to each pair of parallel sentences. If the score is `0`, it means that the sentence was discarded by the Hardrules filter or the language model. If the score is above 0, it means that it made it to the classifier, and the closer to 1 the better is the  sentence. For most languages (and distributed language packs), we consider a sentence to be very likely a good sentence when its score is above `0.5` .

# Software

## Bicleaner AI
### Installation
Bicleaner works with Python3.6+ and can be installed with `pip`:
```bash
python3.7 -m pip install bicleaner-ai
```

It also requires [KenLM](https://github.com/kpu/kenlm) with support for 7-gram language models:
```bash
git clone https://github.com/kpu/kenlm
cd kenlm
python3.7 -m pip install . --install-option="--max_order 7"
cd build
cmake .. -DKENLM_MAX_ORDER=7 -DCMAKE_INSTALL_PREFIX:PATH=/your/prefix/path
make -j all install
```

## tmxt
[tmxt](https://github.com/sortiz/tmxt) is a tool that extract plain text parallel corpora from TMX files.

### Installation

```bash
git clone http://github.com/sortiz/tmxt
python3.7 -m pip install -r tmxt/requirements.txt
```
Two tools are available: `tmxplore.py` (that determines the language codes available inside a TMX file) and  `tmxt.py`, that transforms the TMX to a tab-separated text file.

## Apertium

[Apertium](https://github.com/apertium) is a platform for developing rule-based machine translation systems.  It can be useful to translate to a given language when you do  not have enough parallel text.

### Installation

In Ubuntu and other Debian-like operating systems:
```bash
wget http://apertium.projectjj.com/apt/install-nightly.sh
sudo bash install-nightly.sh
sudo apt-get update
sudo apt-get install apertium-LANGUAGE-PAIR
```
(choose your apropiate `apertium-LANGUAGE-PAIR` from the list under  `apt search apertium`)

For other systems, please read Apertium [documentation](http://wiki.apertium.org/wiki/Installation).

## Bifixer

[Bifixer](https://github.com/bitextor/bifixer/) is a tool that fixes bitexts and tags near-duplicates for removal. It's useful to fix errors in our training corpus.

### Installation

```bash
git clone https://github.com/bitextor/bifixer.git
python3.7 -m pip install -r bifixer/requirements.txt
```
