#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress Tenssorflow logging messages unless log level is explictly set
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set Tensorflow max threads before initialization
if 'BICLEANER_AI_THREADS' in os.environ:
    threads = int(os.environ["BICLEANER_AI_THREADS"])
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
import sys
import logging
import traceback

from timeit import default_timer
from multiprocessing import cpu_count
import regex

#Allows to load modules while inside or outside the package
try:
    from .classify import classify, argument_parser, load_metadata
    from .util import logging_setup
except (ImportError, SystemError):
    from classify import classify, argument_parser, load_metadata
    from util import logging_setup

logging_level = 0

# Argument parsing
def initialization(argv = None):
    global logging_level

    # Validating & parsing arguments
    parser, groupO, _ = argument_parser()
    args = parser.parse_args(argv)

    # Set up logging
    logging_setup(args)
    logging_level = logging.getLogger().level

    # Warn about args.processes deprecation
    if args.processes is not None:
        logging.warning("--processes option is not available anymore, please use BICLEANER_AI_THREADS environment variable instead.")

    # Set the number of processes from the environment variable
    # or instead use all cores
    if "BICLEANER_AI_THREADS" in os.environ and os.environ["BICLEANER_AI_THREADS"]:
        args.processes = int(os.environ["BICLEANER_AI_THREADS"])
    else:
        args.processes = max(1, cpu_count()-1)

    # Try to use HF if not a valid model path
    hub_not_found = False
    if not args.offline and not os.path.exists(args.model):
        logging.debug("Model path does not exist, looking in HuggingFace...")
        from huggingface_hub import snapshot_download, model_info
        from huggingface_hub.utils import RepositoryNotFoundError, HFValidationError
        from requests.exceptions import HTTPError
        try:
            # Check if it exists at the HF Hub
            model_info(args.model, token=args.auth_token)
        except (RepositoryNotFoundError, HTTPError, HFValidationError):
            hub_not_found = True
            args.metadata = args.model
        else:
            # Check if already downloaded
            try:
                # Use local files to avoid triggering the download code
                cache_path = snapshot_download(args.model, local_files_only=True)
            except FileNotFoundError:
                # Not downloaded
                # Download all the model files from the hub
                logging.info(f"Downloading the model {args.model}")
                cache_path = snapshot_download(args.model,
                                               use_auth_token=args.auth_token)
            # Set metadata path to the cache location of the model
            args.metadata = cache_path
    else:
        args.metadata = args.model

    # Add config file suffix if not present
    if not regex.match(r'.*\.ya?ml$', args.metadata):
        args.metadata = args.metadata + '/metadata.yaml'

    if not os.path.isfile(args.metadata):
        if hub_not_found:
            logging.error(
                    f"Model {args.model} not found at HF Hub. If the model is private use --auth_token option.")
        raise FileNotFoundError(f"model {args.metadata} no such file")

    # Load metadata YAML
    args = load_metadata(args, parser)

    logging.debug(args)

    return args

# Filtering input texts
def perform_classification(args):
    time_start = default_timer()
    logging.info("Starting process")

    # Score sentences
    nline = classify(args, args.input, args.output)

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Total: {0} rows".format(nline))
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))
    logging.info("Troughput: {0} rows/s".format(int((nline*1.0)/elapsed_time)))

def main():
    args = initialization() # Parsing parameters and loading models
    perform_classification(args) # Main loop
    logging.info("Program finished")

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
