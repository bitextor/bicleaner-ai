# Training a Multilingual Model

Multilingual models can be a very effective approach to deal with language pairs that we have __very little__ amount of Bicleaner AI training data or __none__ at all,
exploiting the capabilities of __full models__(XLMRoberta models can perform really well on zero-shot crosslingual classification).

Note that the languages expected to work under a crosslingual zero-shot scenario, are the ones that XLMR has been pre-trained on.
Other languages can work, such as Maltese, but this one has strong influences from Arabic and Italian,
so others that have few resemblance to any of the pre-train languages, might not work.
See Appendix A of the [XLMR paper](https://arxiv.org/pdf/1911.02116.pdf) to see the 100 supported languages.

If you are new to Bicleaner AI training, please read the [training documentation](#README.md) before proceeding with this guide.

In this guide, we will train a multilingual model with several languages, all paired with English.
You can, of course, use a different combination, but for now, only multiple languages as a target and one as a source has been proven to work.
Other setups where there are multiple languages in both sides, have not been deeply explored and may not work.

## Generate training data
Since `bicleaner-ai-train` cannot generate training data for multiple languages at the same time, you will need to use `bicleaner-ai-generate-train` for each language pair included.
Assuming you have the needed parallel data and word frequency lists for each language, you can generate training like this:

```
# Generate training data
for $L in fr es ru ar vi fa nb tr
do
    bicleaner-ai-generate-train \
        -s en -t $L \
        -F wordfreqs-$L.gz \
        corpus.en-$L train.en-$L
done

# Generate validation data
for $L in fr es ru ar vi fa nb tr
do
    bicleaner-ai-generate-train \
        -s en -t $L \
        -F wordfreqs-$L.gz \
        dev.en-$L valid.en-$L
done
```

## Mix training data
In a multilingual training you will probably have much more data in total than in a single language pair scenario.
So it is recommended to mix your data to favor language diversity and avoid language pairs that have more data, become the largest part of your training.

Furtheremore, when sampling each language pair, is important to keep each positive sample with its negative samples together.
To do that, you can use our command
```
bicleaner-ai-sample-train 10000 train.en-?? train.en-xx
bicleaner-ai-sample-train 300 valid.en-?? valid.en-xx
```
where the first argument is the maximum number of positive samples per file.


## Training
At the end, just run the same training command as always.
```
bicleaner-ai-train \
    --generated_train train.en-xx \
    --generated_valid valid.en-xx \
    --classifier_type xlmr \
    -m models/en-xx \
    -s en \
    -t xx \
    --porn_removal_train porn-removal.txt.en \
    --porn_removal_file models/en-xx/porn-model.en
```
This time omitting the language model parameter for fluency filter, given that it is not supported for multilingual setups.
On the other side, we keep the porn removal filter, as it works only on the source side and our source language is English.
If you want to keep the porn removal filter and change English with another language, porn removal training data will need to be in that language.

It is important to use `xx` or `xxx` as language parameter, so `bicleaner-ai-classify` will know it is a multilingual model.
To see an example usage of the classify command, go to the [README](../README.md#multilingual-models).

## Performance
Here is a comparison of an experiment with our `en-xx` models, measuring performance on the English-Icelandic validation set:
| model | Matthews Corr. Coef. |
| ------ | :-----: |
| `bitextor/bicleaner-ai-full-en-is` | 85.6 |
| `bitextor/bicleaner-ai-full-en-xx` | 87.4 |
| `bitextor/bicleaner-ai-full-large-en-xx` | 92.4 |

Despite not having any English-Icelandic data in the Bicleaner AI training, multilingual models can perform reasonably well on zero-shot classification.

**NOTE**: this an example of a multilingual model not containing English-Icelandic data, our publicly available model does contain English-Icelandic though.
