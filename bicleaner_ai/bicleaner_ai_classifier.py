#!/usr/bin/env python
import os
# Suppress Tenssorflow logging messages unless log level is explictly set
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import logging
import traceback

from timeit import default_timer

#Allows to load modules while inside or outside the package
try:
    from .classify import classify, argument_parser, load_metadata
    from .util import logging_setup
    from .tokenizer import Tokenizer
except (ImportError, SystemError):
    from classify import classify, argument_parser, load_metadata
    from util import logging_setup
    from tokenizer import Tokenizer

logging_level = 0

# All the scripts should have an initialization according with the usage. Template:
def initialization():
    global logging_level

    # Validating & parsing arguments
    parser, groupO, _ = argument_parser()
    args = parser.parse_args()

    # Set up logging
    logging_setup(args)
    logging_level = logging.getLogger().level
    import tensorflow as tf

    # Set number of processes to be used by TensorFlow
    tf.config.threading.set_intra_op_parallelism_threads(args.processes)
    tf.config.threading.set_inter_op_parallelism_threads(args.processes)

    # Load metadata YAML
    args = load_metadata(args, parser)

    return args

# Filtering input texts
def perform_classification(args):
    if not args.disable_hardrules and not args.disable_lm_filter:
        # Don't force lm modules to be loaded when lm_filter is disabled
        from hardrules.bicleaner_hardrules import load_lm_filter
        lm_filter = load_lm_filter(args.source_lang, args.target_lang, args.metadata_yaml, args.source_tokenizer_command, args.target_tokenizer_command)
    else:
        lm_filter = None

    if not args.disable_hardrules and not args.disable_porn_removal:
        if args.metadata_yaml['porn_removal_side'] == 'tl':
            porn_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)
        else:
            porn_tokenizer = Tokenizer(args.source_tokenizer_command, args.source_lang)
    else:
        porn_tokenizer = None

    time_start = default_timer()
    logging.info("Starting process")

    # Score sentences
    nline = classify(args, args.input, args.output, lm_filter, porn_tokenizer)

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Total: {0} rows".format(nline))
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))
    logging.info("Troughput: {0} rows/s".format(int((nline*1.0)/elapsed_time)))

def main(args):
    perform_classification(args)
    logging.info("Program finished")

if __name__ == '__main__':
    try:
        logging_setup()
        args = initialization() # Parsing parameters
        main(args)  # Running main program
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
