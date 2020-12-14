#!/usr/bin/env python
from multiprocessing import cpu_count

import joblib
from sklearn import metrics
from tempfile import TemporaryFile, NamedTemporaryFile
from timeit import default_timer

import argparse
import logging
import numpy as np
import os
import random
import sys
import json

#Allows to load modules while inside or outside the package  
try:
    from .model import Model
    from .word_freqs_zipf import WordZipfFreqDist
    from .word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from .util import no_escaping, check_dir, check_positive, check_positive_or_zero, logging_setup
    from .training import build_noisy_set, precision_recall, write_metadata, train_fluency_filter, train_porn_removal
    from .tokenizer import Tokenizer
except (SystemError, ImportError):
    from model import Model
    from word_freqs_zipf import WordZipfFreqDist
    from word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from util import no_escaping, check_dir, check_positive, check_positive_or_zero, logging_setup
    from training import build_noisy_set, precision_recall, write_metadata, train_fluency_filter, train_porn_removal
    from tokenizer import Tokenizer

logging_level = 0
    
# Argument parsing
def initialization():

    global logging_level
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    parser.add_argument('bilingual',  nargs='?', type=argparse.FileType('r'), default=sys.stdin,  help="Tab-separated bilingual input file")

    groupM = parser.add_argument_group("Mandatory")
    groupM.add_argument('-m', '--model_dir', type=check_dir, required=True, help="Model directory, metadata, classifier and Sentenceiece models will be saved in the same directory")
    groupM.add_argument('-s', '--source_lang', required=True, help="Source language")
    groupM.add_argument('-t', '--target_lang', required=True, help="Target language")
    groupM.add_argument('-f', '--source_word_freqs', type=argparse.FileType('r'), default=None, required=True, help="L language gzipped list of word frequencies")
    groupM.add_argument('-F', '--target_word_freqs', type=argparse.FileType('r'), default=None, required=True, help="R language gzipped list of word frequencies")
    groupM.add_argument('--mono_train', type=argparse.FileType('r'), default=None, required=True, help="File containing monolingual sentences of both languages shuffled together")

    groupO = parser.add_argument_group('Options')
    groupO.add_argument('-S', '--source_tokenizer_command', help="Source language tokenizer full command")
    groupO.add_argument('-T', '--target_tokenizer_command', help="Target language tokenizer full command")
    groupO.add_argument('-b', '--block_size', type=check_positive, default=10000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=check_positive, default=max(1, cpu_count()-1), help="Number of process to use")
    groupO.add_argument('--wrong_examples_file', type=argparse.FileType('r'), default=None, help="File with wrong examples extracted to replace the synthetic examples from method used by default")
    groupO.add_argument('--disable_lang_ident', default=False, action='store_true', help="Don't apply features that use language detecting")
    groupO.add_argument('--disable_relative_paths', action='store_true', help="Don't use relative paths if they are in the same directory of model_file")
    groupO.add_argument('--seed', default=None, type=int, help="Seed for random number generation: by default, no seeed is used")

    #For LM filtering
    groupO.add_argument('--noisy_examples_file_sl', type=str, help="File with noisy text in the SL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--noisy_examples_file_tl', type=str, help="File with noisy text in the TL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--lm_dev_size', type=check_positive_or_zero, default=2000, help="Number of sentences to be removed from clean text before training LMs. These are used to estimate the perplexity of clean text.")
    groupO.add_argument('--lm_file_sl', type=str, help="SL language model output file.")
    groupO.add_argument('--lm_file_tl', type=str, help="TL language model output file.")
    groupO.add_argument('--lm_training_file_sl', type=str, help="SL text from which the SL LM is trained. If this parameter is not specified, SL LM is trained from the SL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_training_file_tl', type=str, help="TL text from which the TL LM is trained. If this parameter is not specified, TL LM is trained from the TL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_clean_examples_file_sl', type=str, help="File with clean text in the SL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_sl and both files must not have common sentences. This option replaces --lm_dev_size.")
    groupO.add_argument('--lm_clean_examples_file_tl', type=str, help="File with clean text in the TL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_tl and both files must not have common sentences. This option replaces --lm_dev_size.")

    groupO.add_argument('--porn_removal_train', type=argparse.FileType('r'), help="File with training dataset for FastText classifier. Each sentence must contain at the beginning the '__label__negative' or '__label__positive' according to FastText convention. It should be lowercased and tokenized.")
    groupO.add_argument('--porn_removal_test', type=argparse.FileType('r'), help="Test set to compute precision and accuracy of the porn removal classifier")
    groupO.add_argument('--porn_removal_file', type=str, help="Porn removal classifier output file")
    groupO.add_argument('--porn_removal_side', choices=['sl','tl'], default="sl", help="Whether the porn removal should be applied at the source or at the target language.")

    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")

    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    args.metadata = open(args.model_dir + '/metadata.yaml', 'w+')

    # Logging
    logging_setup(args)
    logging_level = logging.getLogger().level

    return args

def train_classifier():
    return

# Main loop of the program
def perform_training(args):
    time_start = default_timer()
    logging.debug("Starting process")

    #Read input to a named temporary file
    count_input_lines = 0
    input = NamedTemporaryFile(mode="w",delete=False)
    for line in args.bilingual:
        input.write(line)
        count_input_lines = count_input_lines +1
    input.close()

    # if count_input_lines < 10000:
    #     logging.error("Training corpus must be at least 10K sentences long (was {}).".format(count_input_lines))
    #     sys.exit(1)

    # Load word frequencies
    if args.source_word_freqs:
        args.sl_word_freqs = WordZipfFreqDist(args.source_word_freqs)
    if args.target_word_freqs:
        args.tl_word_freqs = WordZipfFreqDistDoubleLinked(args.target_word_freqs)
    else:
        args.tl_word_freqs = None

    # Train porn removal classifier
    train_porn_removal(args)

    stats=None
    with open(input.name) as input_f:
        args.bilingual=input_f
        stats=train_fluency_filter(args)
        input_f.seek(0)

        # Shuffle and get length ratio
        noisy_target_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)
        total_size, length_ratio, good_sentences, wrong_sentences = build_noisy_set(args.bilingual, count_input_lines//2, count_input_lines//2, args.wrong_examples_file, args.tl_word_freqs, noisy_target_tokenizer)
        noisy_target_tokenizer.close()
    os.remove(input.name)

    args.length_ratio = length_ratio

    logging.info("Start training.")

    # Use 90% of the input to train and 10% for test
    if args.wrong_examples_file is not None:
        good_examples = int(count_input_lines*0.9)
        good_examples_test = int(count_input_lines*0.1)
        wrong_examples = 0
        with args.examples_file as file:
            wrong_examples = sum(1 for line in file)
        wrong_examples_test = min(good_examples_test, int(wrong_examples*0.1))
    else:
        good_examples = int(count_input_lines//2*0.9)
        good_examples_test = int(count_input_lines//2*0.1)
        wrong_examples = good_examples
        wrong_examples_test = good_examples_test

    model = Model(args.model_dir)

    # Load spm and embeddings if already trained
    try:
        model.load_spm()
        model.load_embed()
    except:
        model.train_vocab(args.mono_train, args.processes)

    # Read sentences from file
    train_sentences = [[], [], []]
    dev_sentences = [[], [], []]
    i = 0
    for good_line in good_sentences:
        parts = good_line.split('\t')
        if i < good_examples:
            train_sentences[0].append(parts[0])
            train_sentences[1].append(parts[1])
            train_sentences[2].append(1)
        else:
            dev_sentences[0].append(parts[0])
            dev_sentences[1].append(parts[1])
            dev_sentences[2].append(1)
        i += 1
    i = 0
    for wrong_line in wrong_sentences:
        parts = wrong_line.split('\t')
        if i < wrong_examples:
            train_sentences[0].append(parts[0])
            train_sentences[1].append(parts[1])
            train_sentences[2].append(0)
        else:
            dev_sentences[0].append(parts[0])
            dev_sentences[1].append(parts[1])
            dev_sentences[2].append(0)

    model.train(train_sentences, dev_sentences)

    logging.info("End training.")

    # Compute histogram for dev predictions
    labels = dev_sentences[2]
    prediction = model.predict(dev_sentences[0], dev_sentences[1])

    pos = 0
    good = []
    wrong = []
    labels = dev_sentences[2]
    for pred in prediction:
        if labels[pos] == 1:
            good.append(pred[0])
        else:
            wrong.append(pred[0])
        pos += 1

    hgood  = np.histogram(good,  bins = np.arange(0, 1.1, 0.1))[0].tolist()
    hwrong = np.histogram(wrong, bins = np.arange(0, 1.1, 0.1))[0].tolist()

    write_metadata(args, hgood, hwrong, stats)
    args.metadata.close()

    # Stats
    logging.info("Finished.")
    elapsed_time = default_timer() - time_start
    logging.info("Elapsed time {:.2f}s.".format(elapsed_time))

# Main function: setup logging and calling the main loop
def main(args):

    # Filtering
    perform_training(args)

if __name__ == '__main__':
    args = initialization()
    main(args)
