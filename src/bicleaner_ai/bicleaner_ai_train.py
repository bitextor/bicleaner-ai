#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress Tenssorflow logging messages unless log level is explictly set
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set Tensorflow max threads before initialization
if 'BICLEANER_AI_THREADS' in os.environ:
    threads = int(os.environ["BICLEANER_AI_THREADS"])
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
from tempfile import TemporaryFile, NamedTemporaryFile, gettempdir
from multiprocessing import cpu_count
from timeit import default_timer
import sentencepiece as spm
import tensorflow as tf
import numpy as np
import argparse
import logging
import random
import sys
import shutil

#Allows to load modules while inside or outside the package  
try:
    from . import __version__
    from .util import *
    from .training import write_metadata
    from .noise_generation import add_noise_options, setup_noise, build_noise
except (SystemError, ImportError):
    from bicleaner_ai import __version__
    from util import *
    from training import build_noise, write_metadata
    from noise_generation import add_noise_options, setup_noise, build_noise

logging_level = 0

# Argument parsing
def get_arguments(argv = None):
    global logging_level

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    groupM = parser.add_argument_group("Mandatory")
    groupM.add_argument('-m', '--model_dir', type=check_dir, required=True, help="Model directory, metadata, classifier and SentencePiece vocabulary will be saved in the same directory")
    groupM.add_argument('-s', '--source_lang', required=True, help="Source language")
    groupM.add_argument('-t', '--target_lang', required=True, help="Target language")
    groupM.add_argument('--mono_train', type=argparse.FileType('r'), default=None, required=False, help="File containing monolingual sentences of both languages shuffled together, used to train SentencePiece embeddings. Not required for XLMR.")
    groupM.add_argument('--parallel_train', type=argparse.FileType('r'), default=None, required='--generated_train' not in sys.argv, help="TSV file containing parallel sentences to train the classifier")
    groupM.add_argument('--parallel_valid', type=argparse.FileType('r'), default=None, required='--generated_valid' not in sys.argv, help="TSV file containing parallel sentences for validation")

    groupO = parser.add_argument_group('Options')
    groupO.add_argument('--model_name', type=str, default=None, help='The name of the model. For the XLMR models it will be used as the name in Hugging Face Hub.')
    groupO.add_argument('--base_model', type=str, default=None, help='The name of the base model to start from. Only used in XLMR classifiers, must be an XLMR instance.')
    groupO.add_argument('-g', '--gpu', type=check_positive_or_zero, help="Which GPU use, starting from 0. Will set the CUDA_VISIBLE_DEVICES.")
    groupO.add_argument('--mixed_precision', action='store_true', default=False, help="Use mixed precision float16 for training")
    groupO.add_argument('--distilled', action='store_true', help='Enable Knowledge Distillation training. It needs pre-built training set with raw scores from a teacher model.')
    groupO.add_argument('--seed', default=None, type=int, help="Seed for random number generation. By default, no seeed is used.")
    groupO.add_argument('--generated_train', type=str, default=None, help="Generated training dataset. If the file already exists the training dataset will be loaded from here.")
    groupO.add_argument('--generated_valid', type=str, default=None, help="Generated validation dataset. If the file already exists the validation dataset will be loaded from here.")

    # Model training options
    groupO.add_argument('--classifier_type', choices=model_classes.keys(), default="dec_attention", help="Neural network architecture for the classifier")
    groupO.add_argument('--batch_size', type=check_positive, default=None, help="Batch size during classifier training. If None, default architecture value will be used.")
    groupO.add_argument('--steps_per_epoch', type=check_positive, default=None, help="Number of batch updates per epoch during training. If None, default architecture value will be used or the full dataset size.")
    groupO.add_argument('--epochs', type=check_positive, default=None, help="Number of epochs for training. If None, default architecture value will be used.")
    groupO.add_argument('--patience', type=check_positive, default=None, help="Stop training when validation has stopped improving after PATIENCE number of epochs")

    add_noise_options(groupO)

    # Porn removal training options
    groupO.add_argument('--porn_removal_train', type=argparse.FileType('r'), help="File with training dataset for porn filter. Each sentence must contain at the beginning the '__label__negative' or '__label__positive' according to FastText convention. It should be lowercased and tokenized.")
    groupO.add_argument('--porn_removal_test', type=argparse.FileType('r'), help="Test set to compute precision and accuracy of the porn removal filter")
    groupO.add_argument('--porn_removal_file', type=str, default="porn_removal.bin", help="Porn removal output file")
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
    groupL.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="Show version of the package and exit")

    return parser.parse_args(argv)

def initialization(args):
    # initialization of noise generation related arguments
    setup_noise(args)

    if args.mono_train is None and args.classifier_type != 'xlmr':
        logging.error("Argument --mono_train not found, required when not training XLMR classifier")
        sys.exit(1)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        tf.random.seed = args.seed
        spm.set_random_generator_seed(args.seed)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

    # Logging
    logging_setup(args)
    logging_level = logging.getLogger().level
    if logging_level < logging.INFO:
        tf.get_logger().setLevel('INFO')
    else:
        tf.get_logger().setLevel('CRITICAL')

    logging.debug(args)

# Main loop of the program
def perform_training(args):
    time_start = default_timer()
    logging.debug("Starting process")

    # Train porn removal classifier
    if args.porn_removal_file is not None and args.porn_removal_train is not None:
        from hardrules.training import train_porn_removal
        train_porn_removal(args)

    # If generated_train is not provided or empty build new train set
    # otherwise use the prebuilt training set
    if (args.generated_train is None
            or not os.path.isfile(args.generated_train)
            or os.stat(args.generated_train).st_size == 0):
        logging.info("Building training set")
        train_sentences = build_noise(args.parallel_train, args)
        if args.generated_train:
            shutil.copyfile(train_sentences, args.generated_train)
    else:
        train_sentences = args.generated_train
        logging.info("Using pre-built training set: " + train_sentences)

    # Same for valid set
    if (args.generated_valid is None
            or not os.path.isfile(args.generated_valid)
            or os.stat(args.generated_valid).st_size == 0):
        logging.info("Building validation set")
        valid_sentences = build_noise(args.parallel_valid, args)
        if args.generated_valid:
            shutil.copyfile(valid_sentences, args.generated_valid)
    else:
        valid_sentences = args.generated_valid
        logging.info("Using pre-built validation set: " + valid_sentences)
    test_sentences = valid_sentences

    logging.debug(f"Training sentences file: {train_sentences}")
    logging.debug(f"Validation sentences file: {valid_sentences}")

    # Train LM fluency filter
    if args.lm_file_sl and args.lm_file_tl:
        from hardrules.training import train_fluency_filter
        args.parallel_train.seek(0)
        args.input = args.parallel_train
        args.source_tokenizer_command = None
        args.target_tokenizer_command = None
        lm_stats = train_fluency_filter(args)
    else:
        lm_stats = None

    if args.parallel_train:
        args.parallel_train.close()
    if args.parallel_valid:
        args.parallel_valid.close()

    # Define the model name
    if args.model_name is None:
        model_name = 'bitextor/bicleaner-ai'
        if args.classifier_type in ['dec_attention', 'transformer']:
            model_name += f'-lite-{args.source_lang}-{args.target_lang}'
        else:
            model_name += f'-full-{args.source_lang}-{args.target_lang}'
    else:
        model_name = args.model_name

    model_settings = {
        "model_name": model_name,
        "base_model": args.base_model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "vocab_size": args.vocab_size if 'vocab_size' in args else None,
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

    y_true, y_pred = classifier.train(train_sentences, valid_sentences)

    if args.generated_train and train_sentences != args.generated_train:
        os.unlink(train_sentences)
    if args.generated_valid and valid_sentences != args.generated_valid:
        os.unlink(valid_sentences)
    logging.info("End training")

    args.metadata = open(args.model_dir + '/metadata.yaml', 'w+')
    write_metadata(args, classifier, y_true, y_pred, lm_stats)
    args.metadata.close()

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Elapsed time {:.2f}s".format(elapsed_time))

# Main function: setup logging and calling the main loop
def main():
    args = get_arguments()
    initialization(args)
    perform_training(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
