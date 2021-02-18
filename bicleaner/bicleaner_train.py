#!/usr/bin/env python
from sklearn.metrics import f1_score, precision_score
from tempfile import TemporaryFile, NamedTemporaryFile
from multiprocessing import cpu_count
from timeit import default_timer
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
import random
import sys
import shutil

#Allows to load modules while inside or outside the package  
try:
    from .models import DecomposableAttention
    from .word_freqs_zipf import WordZipfFreqDist
    from .word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from .util import no_escaping, check_dir, check_positive, check_positive_or_zero, logging_setup
    from .training import build_noise, load_tuple_sentences, write_metadata, train_fluency_filter, train_porn_removal
    from .tokenizer import Tokenizer
except (SystemError, ImportError):
    from models import DecomposableAttention
    from word_freqs_zipf import WordZipfFreqDist
    from word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from util import no_escaping, check_dir, check_positive, check_positive_or_zero, logging_setup
    from training import build_noise, load_tuple_sentences, write_metadata, train_fluency_filter, train_porn_removal
    from tokenizer import Tokenizer

logging_level = 0
    
# Argument parsing
def initialization():

    global logging_level
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    groupM = parser.add_argument_group("Mandatory")
    groupM.add_argument('-m', '--model_dir', type=check_dir, required=True, help="Model directory, metadata, classifier and Sentenceiece models will be saved in the same directory")
    groupM.add_argument('-s', '--source_lang', required=True, help="Source language")
    groupM.add_argument('-t', '--target_lang', required=True, help="Target language")
    groupM.add_argument('-f', '--source_word_freqs', type=argparse.FileType('r'), default=None, required=True, help="L language gzipped list of word frequencies")
    groupM.add_argument('-F', '--target_word_freqs', type=argparse.FileType('r'), default=None, required=True, help="R language gzipped list of word frequencies")
    groupM.add_argument('--mono_train', type=argparse.FileType('r'), default=None, required=True, help="File containing monolingual sentences of both languages shuffled together to train embeddings")
    groupM.add_argument('--parallel_train', type=argparse.FileType('r'), default=None, required=True, help="TSV file containing parallel sentences to train the classifier")
    groupM.add_argument('--parallel_test', type=argparse.FileType('r'), default=None, required=True, help="TSV file containing parallel sentences to test the classifier")

    groupO = parser.add_argument_group('Options')
    groupO.add_argument('-S', '--source_tokenizer_command', help="Source language tokenizer full command")
    groupO.add_argument('-T', '--target_tokenizer_command', help="Target language tokenizer full command")
    groupO.add_argument('-b', '--block_size', type=check_positive, default=10000, help="Sentence pairs per block when apliying multiprocessing in the noise function")
    groupO.add_argument('-p', '--processes', type=check_positive, default=max(1, cpu_count()-1), help="Number of process to use")
    groupO.add_argument('-g', '--gpu', type=check_positive_or_zero, help="Which GPU use")
    groupO.add_argument('--save_train_data', type=str, default=None, help="Save the generated dataset into a file. If the file already exists the training dataset will be loaded from there.")
    groupO.add_argument('--wrong_examples_file', type=argparse.FileType('r'), default=None, help="File with wrong examples extracted to replace the synthetic examples from method used by default")
    groupO.add_argument('--disable_lang_ident', default=False, action='store_true', help="Don't apply features that use language detecting")
    groupO.add_argument('--disable_relative_paths', action='store_true', help="Don't use relative paths if they are in the same directory of model_file")
    groupO.add_argument('--seed', default=None, type=int, help="Seed for random number generation: by default, no seeed is used")

    # Noise options
    groupO.add_argument('--pos_ratio', default=1, type=int, help="Ratio of positive samples used to oversample on validation and test sets")
    groupO.add_argument('--rand_ratio', default=3, type=int, help="Ratio of negative samples misaligned randomly")
    groupO.add_argument('--womit_ratio', default=3, type=int, help="Ratio of negative samples misaligned by randomly omitting words")
    groupO.add_argument('--freq_ratio', default=3, type=int, help="Ratio of negative samples misaligned by replacing words by frequence")
    groupO.add_argument('--fuzzy_ratio', default=0, type=int, help="Ratio of negative samples misaligned by fuzzy matching")
    groupO.add_argument('--neighbour_mix', default=False, type=bool, help="If use negative samples misaligned by neighbourhood")

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
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        tf.random.seed = args.seed

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tf.config.threading.set_intra_op_parallelism_threads(args.processes)
    tf.config.threading.set_inter_op_parallelism_threads(args.processes)

    args.metadata = open(args.model_dir + '/metadata.yaml', 'w+')

    # Logging
    logging_setup(args)
    logging_level = logging.getLogger().level

    return args

# Main loop of the program
def perform_training(args):
    time_start = default_timer()
    logging.debug("Starting process")

    # Load word frequencies
    if args.source_word_freqs:
        args.sl_word_freqs = WordZipfFreqDist(args.source_word_freqs)
    if args.target_word_freqs:
        args.tl_word_freqs = WordZipfFreqDistDoubleLinked(args.target_word_freqs)
    else:
        args.tl_word_freqs = None

    # Train porn removal classifier
    train_porn_removal(args)

    if (args.save_train_data is None
            or not os.path.isfile(args.save_train_data)
            or os.stat(args.save_train_data).st_size == 0):
        logging.info("Building training set.")
        train_sentences = build_noise(args.parallel_train, args)
        if args.save_train_data is not None:
            shutil.copyfile(train_sentences, args.save_train_data)
    else:
        train_sentences = args.save_train_data
        logging.info("Using pre-built training set: " + train_sentences)
    logging.info("Building development set.")
    test_sentences = build_noise(args.parallel_test, args)
    dev_sentences = test_sentences
    logging.debug(f"Training sentences file: {train_sentences}")
    logging.debug(f"Development sentences file: {dev_sentences}")

    logging.info("Start training.")

    model = DecomposableAttention(args.model_dir)
    # Load spm and embeddings if already trained
    try:
        model.load_spm()
        model.load_embed()
    except:
        model.train_vocab(args.mono_train, args.processes)

    y_true, y_pred = model.train(train_sentences, dev_sentences)

    if args.save_train_data is not None and train_sentences != args.save_train_data:
        os.unlink(train_sentences)
    os.unlink(dev_sentences)
    logging.info("End training.")

    # Compute histogram for test predictions
    pos = 0
    good = []
    wrong = []
    for pred in y_pred:
        if y_true[pos] == 1:
            good.append(pred[0])
        else:
            wrong.append(pred[0])
        pos += 1

    hgood  = np.histogram(good,  bins = np.arange(0, 1.1, 0.1))[0].tolist()
    hwrong = np.histogram(wrong, bins = np.arange(0, 1.1, 0.1))[0].tolist()

    write_metadata(args, hgood, hwrong, None)
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
