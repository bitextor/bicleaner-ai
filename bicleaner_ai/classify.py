from hardrules.hardrules import wrong_tu
from multiprocessing import cpu_count
from tempfile import gettempdir
import tensorflow as tf
import numpy as np
import traceback
import argparse
import fasttext
import logging
import yaml
import sys
import os
import gc

#Allows to load modules while inside or outside the package
try:
    from .util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup, get_model
except (ImportError, SystemError):
    from util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup, get_model

__author__ = "Jaume Zaragoza"
__version__ = "Version 1.0 # 14/06/2021 #"
__version__ = "Version 1.0.1 # 16/06/2021 #"


# Create an argument parser and add all the arguments
def argument_parser():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    # Mandatory parameters
    ## Input file. Try to open it to check if it exists
    parser.add_argument('input', type=argparse.FileType('rt'), default=None, help="Tab-separated files to be classified")
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="Output of the classification")
    parser.add_argument('metadata', type=argparse.FileType('r'), default=None, help="Training metadata (YAML file)")

    # Options group
    groupO = parser.add_argument_group('Optional')
    groupO.add_argument("-S", "--source_tokenizer_command", type=str, help="Source language (SL) tokenizer full command")
    groupO.add_argument("-T", "--target_tokenizer_command", type=str, help="Target language (TL) tokenizer full command")

    groupO.add_argument("--scol", default=3, type=check_positive, help ="Source sentence column (starting in 1)")
    groupO.add_argument("--tcol", default=4, type=check_positive, help ="Target sentence column (starting in 1)")
    groupO.add_argument('-b', '--block_size', type=int, default=1000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count()-1), help="Number of processes to use")
    groupO.add_argument('--batch_size', type=int, default=32, help="Sentence pairs per block")

    groupO.add_argument('--tmp_dir', default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-d', '--discarded_tus', type=argparse.FileType('w'), default=None, help="TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file.")
    groupO.add_argument('--score_only',action='store_true', help="Only output one column which is the bicleaner score", default=False)
    groupO.add_argument('--calibrated',action='store_true', help="Output calibrated scores", default=False)
    groupO.add_argument('--raw_output',action='store_true', help="Return raw output without computing positive class probability.", default=False)
    groupO.add_argument('--lm_threshold',type=check_positive_between_zero_and_one, default=0.5, help="Threshold for language model fluency scoring. All TUs whose LM fluency score falls below the threshold will are removed (classifier score set to 0), unless the option --keep_lm_result set.")

    groupO.add_argument('--disable_hardrules',action = 'store_true', help = "Disables the bicleaner_hardrules filtering (only bicleaner_classify is applied)")
    groupO.add_argument('--disable_lm_filter', action = 'store_true', help = "Disables LM filtering")
    groupO.add_argument('--disable_porn_removal', default=False, action='store_true', help="Don't apply porn removal")
    groupO.add_argument('--disable_minimal_length', default=False, action='store_true', help="Don't apply minimal length rule")

    # Logging group
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
    groupL.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="show version of this script and exit")

    return parser, groupO, groupL


# Load metadata, classifier, lm_filter and porn_removal
def load_metadata(args, parser):
    try:
        # Load YAML
        metadata_yaml = yaml.safe_load(args.metadata)
        yamlpath = os.path.dirname(os.path.abspath(args.metadata.name))
        metadata_yaml["yamlpath"] = yamlpath

        # Read language pair and tokenizers
        args.source_lang=metadata_yaml["source_lang"]
        args.target_lang=metadata_yaml["target_lang"]
        if "source_tokenizer_command" in metadata_yaml:
            args.source_tokenizer_command=metadata_yaml["source_tokenizer_command"]
        if "target_tokenizer_command" in metadata_yaml:
            args.target_tokenizer_command=metadata_yaml["target_tokenizer_command"]

        # Load classifier
        if "calibration_params" in metadata_yaml["classifier_settings"]:
            cal_params = metadata_yaml["classifier_settings"]["calibration_params"]
            if args.calibrated:
                logging.info(f"Enabling calibrated output with parameters: {cal_params}")
        else:
            cal_params = None
        args.clf = get_model(metadata_yaml["classifier_type"])(yamlpath,
                                                metadata_yaml["classifier_settings"])
        args.clf.load()

        if "disable_lang_ident" in metadata_yaml:
            args.disable_lang_ident = metadata_yaml["disable_lang_ident"]
        else:
            args.disable_lang_ident = False

        # Try loading metadata for LM filtering
        if not args.disable_lm_filter:
            if not ("source_lm" in metadata_yaml and "target_lm" in metadata_yaml):
                args.disable_lm_filter = True
                logging.warning("LM filter not present in metadata, disabling.")
        else:
            logging.info("LM filtering disabled")

        # Try loading porn_removal model
        if not args.disable_porn_removal:
            if not ("porn_removal_file" in metadata_yaml and "porn_removal_side" in metadata_yaml):
                args.porn_removal = None
                args.disable_porn_removal = True
                logging.warning("Porn removal not present in metadata, disabling")
            else:
                try:
                    args.porn_removal = fasttext.load_model(os.path.join(yamlpath, metadata_yaml['porn_removal_file']))
                except:
                    args.porn_removal = fasttext.load_model(args.metadata_yaml['porn_removal_file'])
        else:
            args.porn_removal = None
            logging.info("Porn removal disabled")


        logging.debug("YAML")
        logging.debug(metadata_yaml)
        args.metadata_yaml = metadata_yaml
        parser.set_defaults(**metadata_yaml)
    except:
        logging.error("Error loading metadata")
        traceback.print_exc()
        sys.exit(1)

    # Ensure that directory exists; if not, create it
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed")
    return args


# Classify sentences from input and place them at output
# that can be either files or stdin/stdout
def classify(args, input, output, lm_filter, porn_tokenizer):
    nline = 0
    buf_sent = []
    buf_sent_sl = []
    buf_sent_tl = []
    buf_score = []

    # Read from input file/stdin
    for line in input:
        nline += 1
        parts = line.split("\t")

        # Parse fields and buffer sentences
        sl_sentence=None
        tl_sentence=None
        if len(parts) >= max(args.scol, args.tcol):
            sl_sentence=parts[args.scol -1].strip()
            tl_sentence=parts[args.tcol -1].strip()
        else:
            logging.error("ERROR: scol ({}) or tcol ({}) indexes above column number ({}) on line {}".format(args.scol, args.tcol, len(parts), nline))

        buf_sent.append(line)

        # Buffer sentences that are not empty and pass hardrules
        # buffer all sentences in raw mode
        if args.raw_output or (sl_sentence and tl_sentence and (args.disable_hardrules or wrong_tu(sl_sentence, tl_sentence, args, lm_filter, args.porn_removal, porn_tokenizer)== False)):
            buf_score.append(1)
            buf_sent_sl.append(sl_sentence)
            buf_sent_tl.append(tl_sentence)
        else:
            buf_score.append(0)

        # Score batch and empty buffers
        if (nline % args.block_size) == 0:
            classify_batch(args, output, buf_sent, buf_sent_sl, buf_sent_tl, buf_score)
            buf_sent = []
            buf_sent_sl = []
            buf_sent_tl = []
            buf_score = []

        # Avoid memory not beeing freed too late
        if (nline % 1e6) == 0:
            gc.collect()
            tf.keras.backend.clear_session()

    # Score remaining sentences
    if len(buf_sent) > 0:
        classify_batch(args, output, buf_sent, buf_sent_sl, buf_sent_tl, buf_score)

    return nline

# Score a batch of sentences
def classify_batch(args, output, buf_sent, buf_sent_sl, buf_sent_tl, buf_score):
    # Classify predictions
    if len(buf_sent_tl) > 0 and len(buf_sent_sl) > 0:
        predictions = args.clf.predict(buf_sent_sl, buf_sent_tl,
                                       args.batch_size,
                                       args.calibrated,
                                       args.raw_output)
    else:
        predictions = []
    p = iter(predictions)

    # Print sentences and scores to output
    for score, sent in zip(buf_score, buf_sent):
        if score == 1:
            clf_score = next(p)
            # Print 2 scores if raw output is enabled
            if args.raw_output and len(clf_score) == 2:
                outscore = f"{clf_score[0]:.3f}\t{clf_score[1]:.3f}"
            else:
                outscore = f"{clf_score[0]:.3f}"

            if args.score_only:
                output.write(outscore)
            else:
                output.write(sent.strip())
                output.write("\t")
                output.write(outscore)
            output.write("\n")
        else:
            if args.score_only:
                output.write("0")
            else:
                output.write(sent.rstrip("\n"))
                output.write("\t0")
            output.write("\n")
