#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress Tenssorflow logging messages unless log level is explictly set
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeit import default_timer
import argparse
import logging
import sys

try:
    from .util import logging_setup
    from .noise_generation import build_noise, setup_noise, add_noise_options
except (SystemError, ImportError):
    from util import logging_setup
    from noise_generation import build_noise, setup_noise, add_noise_options

def get_arguments(argv = None):
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate synthetic data for Bicleaner AI training.")

    groupM = parser.add_argument_group("Mandatory")
    groupM.add_argument('-s', '--source_lang', required=True, help="Source language")
    groupM.add_argument('-t', '--target_lang', required=True, help="Target language")
    groupM.add_argument('input', type=argparse.FileType('rt'), default=None,
                        help="TSV file containing parallel sentences used as positive samples.")
    groupM.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help="Generated synthetic data for ready training")

    groupO = parser.add_argument_group('Optional')
    add_noise_options(groupO)

    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")

    return parser.parse_args()


def main():
    args = get_arguments()
    logging_setup(args)
    setup_noise(args)
    logging.debug(args)

    time_start = default_timer()

    # Generate synthetic noise
    logging.info(f"Generating synthetic noise for {args.source_lang}-{args.target_lang}")
    filename = build_noise(args.input, args)

    # Save in output file
    with open(filename) as f:
        for line in f:
            print(line, end='')

    os.unlink(filename)

    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))

if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
