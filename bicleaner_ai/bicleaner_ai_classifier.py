#!/usr/bin/env python
import os
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

#Allows to load modules while inside or outside the package
try:
    from .classify import classify, argument_parser, load_metadata
    from .util import logging_setup
except (ImportError, SystemError):
    from classify import classify, argument_parser, load_metadata
    from util import logging_setup

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

    # Warn about args.processes deprecation
    if args.processes is not None:
        logging.warging("--processes option is not available anymore, please use BICLEANER_AI_THREADS environment variable instead.")

    # Set the number of processes from the environment variable
    # or instead use all cores
    if "BICLEANER_AI_THREADS" in os.environ and os.environ["BICLEANER_AI_THREADS"]:
        args.processes = int(os.environ["BICLEANER_AI_THREADS"])
    else:
        args.processes = max(1, cpu_count()-1)

    # Try to download the model if not a valid path
    hub_not_found = False
    if not args.offline:
        from huggingface_hub import snapshot_download, model_info
        from huggingface_hub.utils import RepositoryNotFoundError
        from requests.exceptions import HTTPError
        try:
            # Check if it exists at the HF Hub
            model_info(args.model, token=args.auth_token)
        except RepositoryNotFoundError:
            hub_not_found = True
            args.metadata = args.model + '/metadata.yaml'
        else:
            logging.info(f"Downloading the model {args.model}")
            # Download all the model files from the hub
            cache_path = snapshot_download(args.model,
                                           use_auth_token=args.auth_token)
            # Set metadata path to the cache location of the model
            args.metadata = cache_path + '/metadata.yaml'
    else:
        args.metadata = args.model + '/metadata.yaml'

    if not os.path.isfile(args.metadata):
        if hub_not_found:
            logging.error(
                    f"Model {args.model} not found at HF Hub")
        raise FileNotFoundError(f"model {args.model} no such file")

    # Load metadata YAML
    args = load_metadata(args, parser)

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
