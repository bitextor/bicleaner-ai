#!/usr/bin/env python
import os
# Suppress Tenssorflow logging messages unless log level is explictly set
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tempfile import TemporaryFile, NamedTemporaryFile, gettempdir
from multiprocessing import cpu_count
from timeit import default_timer
import tensorflow as tf
import numpy as np
import argparse
import logging
import random
import sys
import shutil

#Allows to load modules while inside or outside the package  
try:
    from .word_freqs_zipf import WordZipfFreqDist
    from .word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from .util import *
    from .training import build_noise, write_metadata
    from .tokenizer import Tokenizer
except (SystemError, ImportError):
    from word_freqs_zipf import WordZipfFreqDist
    from word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from util import *
    from training import build_noise, write_metadata
    from tokenizer import Tokenizer

logging_level = 0

# Argument parsing
def initialization():
    global logging_level

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    groupM = parser.add_argument_group("Mandatory")
    groupM.add_argument('-m', '--model_dir', type=check_dir, required=True, help="Model directory, metadata, classifier and SentencePiece models will be saved in the same directory")
    groupM.add_argument('-s', '--source_lang', required=True, help="Source language")
    groupM.add_argument('-t', '--target_lang', required=True, help="Target language")
    groupM.add_argument('--mono_train', type=argparse.FileType('r'), default=None, required=False, help="File containing monolingual sentences of both languages shuffled together, used to train SentencePiece embeddings. Not required for XLMR.")
    groupM.add_argument('--parallel_train', type=argparse.FileType('r'), default=None, required=True, help="TSV file containing parallel sentences to train the classifier")
    groupM.add_argument('--parallel_dev', type=argparse.FileType('r'), default=None, required=True, help="TSV file containing parallel sentences for development")

    groupO = parser.add_argument_group('Options')
    groupO.add_argument('-S', '--source_tokenizer_command', help="Source language tokenizer full command")
    groupO.add_argument('-T', '--target_tokenizer_command', help="Target language tokenizer full command")
    #groupO.add_argument('-f', '--source_word_freqs', type=argparse.FileType('r'), default=None, required=False, help="L language gzipped list of word frequencies")
    groupO.add_argument('-F', '--target_word_freqs', type=argparse.FileType('r'), default=None, required=False, help="R language gzipped list of word frequencies (needed for frequence based noise)")
    groupO.add_argument('--block_size', type=check_positive, default=10000, help="Sentence pairs per block when apliying multiprocessing in the noise function")
    groupO.add_argument('-p', '--processes', type=check_positive, default=max(1, cpu_count()-1), help="Number of process to use")
    groupO.add_argument('-g', '--gpu', type=check_positive_or_zero, help="Which GPU use, starting from 0. Will set the CUDA_VISIBLE_DEVICES.")
    groupO.add_argument('--mixed_precision', action='store_true', default=False, help="Use mixed precision float16 for training")
    groupO.add_argument('--save_train_data', type=str, default=None, help="Save the generated dataset into a file. If the file already exists the training dataset will be loaded from there.")
    groupO.add_argument('--distilled', action='store_true', help='Enable Knowledge Distillation training. It needs pre-built training set with raw scores from a teacher model.')
    groupO.add_argument('--seed', default=None, type=int, help="Seed for random number generation. By default, no seeed is used.")

    # Classifier training options
    groupO.add_argument('--classifier_type', choices=model_classes.keys(), default="dec_attention", help="Neural network architecture of the classifier")
    groupO.add_argument('--batch_size', type=check_positive, default=None, help="Batch size during classifier training. If None, default architecture value will be used.")
    groupO.add_argument('--steps_per_epoch', type=check_positive, default=None, help="Number of batch updates per epoch during training. If None, default architecture value will be used or the full dataset size.")
    groupO.add_argument('--epochs', type=check_positive, default=None, help="Number of epochs for training. If None, default architecture value will be used.")
    groupO.add_argument('--patience', type=check_positive, default=None, help="Stop training when validation has stopped improving after PATIENCE number of epochs")

    # Negative sampling options
    groupO.add_argument('--pos_ratio', default=1, type=int, help="Ratio of positive samples used to oversample on validation and test sets")
    groupO.add_argument('--rand_ratio', default=3, type=int, help="Ratio of negative samples misaligned randomly")
    groupO.add_argument('--womit_ratio', default=3, type=int, help="Ratio of negative samples misaligned by randomly omitting words")
    groupO.add_argument('--freq_ratio', default=3, type=int, help="Ratio of negative samples misaligned by replacing words by frequence (needs --target_word_freq)")
    groupO.add_argument('--fuzzy_ratio', default=0, type=int, help="Ratio of negative samples misaligned by fuzzy matching")
    groupO.add_argument('--neighbour_mix', default=False, type=bool, help="If use negative samples misaligned by neighbourhood")

    # Porn removal training options
    groupO.add_argument('--porn_removal_train', type=argparse.FileType('r'), help="File with training dataset for FastText classifier. Each sentence must contain at the beginning the '__label__negative' or '__label__positive' according to FastText convention. It should be lowercased and tokenized.")
    groupO.add_argument('--porn_removal_test', type=argparse.FileType('r'), help="Test set to compute precision and accuracy of the porn removal classifier")
    groupO.add_argument('--porn_removal_file', type=str, default="porn_removal.bin", help="Porn removal classifier output file")
    groupO.add_argument('--porn_removal_side', choices=['sl','tl'], default="sl", help="Whether the porn removal should be applied at the source or at the target language.")

    # LM fluency filter training options
    groupO.add_argument('--noisy_examples_file_sl', type=str, help="File with noisy text in the SL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--noisy_examples_file_tl', type=str, help="File with noisy text in the TL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--lm_dev_size', type=check_positive_or_zero, default=2000, help="Number of sentences to be removed from clean text before training LMs. These are used to estimate the perplexity of clean text.")
    groupO.add_argument('--lm_file_sl', type=str, help="SL language model output file.")
    groupO.add_argument('--lm_file_tl', type=str, help="TL language model output file.")
    groupO.add_argument('--lm_training_file_sl', type=str, help="SL text from which the SL LM is trained. If this parameter is not specified, SL LM is trained from the SL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_training_file_tl', type=str, help="TL text from which the TL LM is trained. If this parameter is not specified, TL LM is trained from the TL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_clean_examples_file_sl', type=str, help="File with clean text in the SL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_sl and both files must not have common sentences. This option replaces --lm_dev_size.")
    groupO.add_argument('--lm_clean_examples_file_tl', type=str, help="File with clean text in the TL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_tl and both files must not have common sentences. This option replaces --lm_dev_size.")

    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")

    args = parser.parse_args()

    if args.freq_ratio > 0 and args.target_word_freqs is None:
        raise Exception("Frequence based noise needs target language word frequencies")
    if args.mono_train is None and args.classifier_type != 'xlmr':
        raise Exception("Argument --mono_train not found, required when not training XLMR classifier")

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        tf.random.seed = args.seed

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        import psutil
        cpus = psutil.cpu_count(logical=False)
        # Set number of threads for CPU training
        tf.config.threading.set_intra_op_parallelism_threads(min(cpus, args.processes))
        tf.config.threading.set_inter_op_parallelism_threads(min(2, args.processes))

    if args.mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')

    # Remove trailing / in model dir
    args.model_dir.rstrip('/')

    # If the model files are basenames, prepend model path
    if args.lm_file_sl and args.lm_file_sl.count('/') == 0:
        args.lm_file_sl = args.model_dir + '/' + args.lm_file_sl
    if args.lm_file_tl and args.lm_file_tl.count('/') == 0:
        args.lm_file_tl = args.model_dir + '/' + args.lm_file_tl
    if args.porn_removal_file and args.porn_removal_file.count('/') == 0:
        args.porn_removal_file = args.model_dir + '/' + args.porn_removal_file

    args.metadata = open(args.model_dir + '/metadata.yaml', 'w+')

    # Logging
    logging_setup(args)
    logging_level = logging.getLogger().level
    if logging_level < logging.INFO:
        tf.get_logger().setLevel('INFO')
    else:
        tf.get_logger().setLevel('CRITICAL')

    return args

# Main loop of the program
def perform_training(args):
    time_start = default_timer()
    logging.debug("Starting process")

    # Load word frequencies
    #if args.source_word_freqs:
    #    args.sl_word_freqs = WordZipfFreqDist(args.source_word_freqs)
    if args.target_word_freqs:
        args.tl_word_freqs = WordZipfFreqDistDoubleLinked(args.target_word_freqs)
    else:
        args.tl_word_freqs = None

    # Train porn removal classifier
    if args.porn_removal_file is not None and args.porn_removal_train is not None:
        from hardrules.training import train_porn_removal
        train_porn_removal(args)

    if (args.save_train_data is None
            or not os.path.isfile(args.save_train_data)
            or os.stat(args.save_train_data).st_size == 0):
        logging.info("Building training set")
        train_sentences = build_noise(args.parallel_train, args)
        if args.save_train_data is not None:
            shutil.copyfile(train_sentences, args.save_train_data)
    else:
        train_sentences = args.save_train_data
        logging.info("Using pre-built training set: " + train_sentences)
    logging.info("Building development set")
    test_sentences = build_noise(args.parallel_dev, args)
    dev_sentences = test_sentences
    logging.debug(f"Training sentences file: {train_sentences}")
    logging.debug(f"Development sentences file: {dev_sentences}")

    # Train LM fluency filter
    if args.lm_file_sl and args.lm_file_tl:
        from hardrules.training import train_fluency_filter
        args.parallel_train.seek(0)
        args.input = args.parallel_train
        lm_stats = train_fluency_filter(args)
    args.parallel_train.close()
    args.parallel_dev.close()

    logging.info("Start training")

    model_settings = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch
    }
    # Avoid overriding settings with None
    model_settings = {k:v for k,v in model_settings.items() if v is not None }
    classifier = get_model(args.classifier_type)(
                    args.model_dir,
                    model_settings,
                    distilled=args.distilled)
    if args.classifier_type in ['dec_attention', 'transformer']:
        # Load spm and embeddings if already trained
        try:
            classifier.load_spm()
            classifier.load_embed()
        except:
            classifier.train_vocab(args.mono_train, args.processes)

    y_true, y_pred = classifier.train(train_sentences, dev_sentences)

    if args.save_train_data is not None and train_sentences != args.save_train_data:
        os.unlink(train_sentences)
    os.unlink(dev_sentences)
    logging.info("End training")

    write_metadata(args, classifier, y_true, y_pred, lm_stats)
    args.metadata.close()

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Elapsed time {:.2f}s".format(elapsed_time))

# Main function: setup logging and calling the main loop
def main(args):

    # Filtering
    perform_training(args)

if __name__ == '__main__':
    args = initialization()
    main(args)
