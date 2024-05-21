import os
import argparse
import logging
import random
import sys

try:
    from .util import logging_setup
except (SystemError, ImportError):
    from util import logging_setup

def get_arguments(argv = None):
    parser = argparse.ArgumentParser(
            prog=os.path.basename(sys.argv[0]),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Randomly select training samples from multiple training files."
                " Keep each group of negative and the positive samples together.")
    parser.add_argument("num_samples", type=int, help="Number of sample groups to extract per file.")
    parser.add_argument("files", type=argparse.FileType('rt'), nargs='*',
                        help="Files to sample from.")

    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
    return parser.parse_args()

def main():
    args = get_arguments()
    logging_setup(args)

    for infile in args.files:
        # Count the number of positive samples
        num_pos = 0
        for line in infile:
            parts = line.strip().split('\t')
            if parts[2] == '1':
                num_pos += 1

        logging.info(f"File '{infile.name}' extract {num_pos} sample groups")

        # Extract a random sample of each file of at most 'num_samples'
        # in a way that for each positive sample we also keep the negative samples
        # associated to it. That is the samples that come after it in the file
        infile.seek(0)
        # if the file is smaller than the requested amount, just print the full content
        if args.num_samples >= num_pos:
            print(f.read(), end='')
            continue

        to_sample = set(random.sample(range(1,num_pos), args.num_samples))
        pos_id = 0
        for line in infile:
            parts = line.strip().split('\t')
            if parts[2] == '1':
                pos_id += 1

            if pos_id in to_sample:
                print(line.strip())

        infile.close()

if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
