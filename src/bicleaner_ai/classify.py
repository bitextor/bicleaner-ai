from hardrules.hardrules import Hardrules
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
    from . import __version__
    from .util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup, get_model
except (ImportError, SystemError):
    from bicleaner_ai import __version__
    from util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup, get_model

HBS_CYR = ('hbs', 'sr', 'me', 'cnr')


# Create an argument parser and add all the arguments
def argument_parser():
    header = "--header" in sys.argv
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    # Mandatory parameters
    ## Input file. Try to open it to check if it exists
    parser.add_argument('input', type=argparse.FileType('rt'), default=None, help="Tab-separated files to be classified")
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="Output of the classification")
    parser.add_argument('model', type=str, default=None, help="Path to model directory or HuggingFace Hub model identifier (such as 'bitextor/bicleaner-ai-full-en-fr')")

    # Options group
    groupO = parser.add_argument_group('Optional')
    groupO.add_argument("-S", "--source_tokenizer_command", type=str, help="Source language (SL) tokenizer full command")
    groupO.add_argument("-T", "--target_tokenizer_command", type=str, help="Target language (TL) tokenizer full command")

    groupO.add_argument("--header", action='store_true', help ="Input file will be expected to have a header, and the output will have a header as well")
    groupO.add_argument("--scol", default=3 if not header else "src_text", type=check_positive if not header else str, help ="Source sentence column (starting in 1). The name of the field is expected instead of the position if --header is set")
    groupO.add_argument("--tcol", default=4 if not header else "trg_text", type=check_positive if not header else str, help ="Target sentence column (starting in 1). The name of the field is expected instead of the position if --header is set")
    groupO.add_argument('-b', '--block_size', type=int, default=10000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', default=None, help="Option no longer available, please set BICLEANER_AI_THREADS environment variable")
    groupO.add_argument('--batch_size', type=int, default=32, help="Sentence pairs per block")

    groupO.add_argument('--tmp_dir', default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('--score_only',action='store_true', help="Only output one column which is the bicleaner score", default=False)
    groupO.add_argument('--calibrated',action='store_true', help="Output calibrated scores", default=False)
    groupO.add_argument('--raw_output',action='store_true', help="Return raw output without computing positive class probability.", default=False)
    groupO.add_argument('--lm_threshold',type=check_positive_between_zero_and_one, default=0.5, help="Threshold for language model fluency scoring. All TUs whose LM fluency score falls below the threshold will are removed (classifier score set to 0), unless the option --keep_lm_result set.")

    groupO.add_argument('--disable_hardrules',action = 'store_true', help = "Disables the bicleaner_hardrules filtering (only bicleaner_classify is applied)")
    groupO.add_argument('--disable_lm_filter', action = 'store_true', help = "Disables LM filtering")
    groupO.add_argument('--disable_porn_removal', default=False, action='store_true', help="Don't apply porn removal")
    groupO.add_argument('--disable_minimal_length', default=False, action='store_true', help="Don't apply minimal length rule")
    groupO.add_argument('--run_all_rules', default=False, action='store_true', help="Run all rules of Hardrules instead of stopping at first discard")
    groupO.add_argument('--rules_config', type=argparse.FileType('r'), default=None, help="Hardrules configuration file")

    # HuggingFace Hub options
    groupO.add_argument('--offline', default=False, action='store_true', help="Don't try to download the model, instead try directly to load from local storage")
    groupO.add_argument('--auth_token', default=None, type=str, help="Auth token for the Hugging Face Hub")

    # Logging group
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
    groupL.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="Show version of the package and exit")

    return parser, groupO, groupL


# Load metadata, classifier, lm_filter and porn_removal
def load_metadata(args, parser):
    metadata_file = open(args.metadata)
    try:
        # Load YAML
        metadata_yaml = yaml.safe_load(metadata_file)
        yamlpath = os.path.dirname(os.path.abspath(args.metadata))
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

        args.translit = False
        if args.target_lang in HBS_CYR or args.source_lang in HBS_CYR:
            try:
                global to_latin
                from cyrtranslit import to_latin
                args.translit = True
            except (ModuleNotFoundError, NameError):
                logging.warning("You might want to install 'cyrtranslit'"
                                " to transliterate before scoring."
                                "This improves accuracy and"
                                " does not change output text.")
        # Load rules config
        if args.rules_config:
            yaml_file = args.rules_config
            args.rules_config = yaml.safe_load(args.rules_config)
            yaml_file.close()


        logging.debug("YAML")
        logging.debug(metadata_yaml)
        args.metadata_yaml = metadata_yaml
        parser.set_defaults(**metadata_yaml)
    except:
        logging.error("Error loading metadata")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if not metadata_file.closed:
            metadata_file.close()

    # Ensure that directory exists; if not, create it
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed")
    return args


def transliterate(args, source, target):
    ''' Transliterate the text in a list of
        source sentences or target sentences '''
    if args.source_lang in HBS_CYR:
        new_source = to_latin(source)
    else:
        new_source = source

    if args.target_lang in HBS_CYR:
        new_target = to_latin(target)
    else:
        new_target = target

    return new_source, new_target


# Classify sentences from input and place them at output
# that can be either files or stdin/stdout
def classify(args, input, output):
    nline = 0
    buf_sent = []
    buf_sent_sl = []
    buf_sent_tl = []
    buf_score = []
    # Don't load hardrules objects if disabled
    if args.disable_hardrules:
        hardrules = None
    else:
        hardrules = Hardrules(args)

    # Process input and output headers
    if args.header:
        args.header = False # We only need to execute the following code once
        header = next(input).strip().split("\t")

        # Transform fields to idxs
        if args.scol not in header:
            raise Exception(f"The provided --scol '{args.scol}' is not in the input header")
        if args.tcol not in header:
            raise Exception(f"The provided --tcol '{args.tcol}' is not in the input header")

        args.scol = int(header.index(args.scol)) + 1
        args.tcol = int(header.index(args.tcol)) + 1

        output_header = header

        if args.score_only:
            output_header = ["bicleaner_ai_score"]
        else:
            output_header.append("bicleaner_ai_score")

        # Write the output header once
        output.write('\t'.join(output_header) + '\n')

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

            # Transliterate if needed the sentences before scoring
            # this does not change the output
            if args.translit:
                sl_sentence, tl_sentence = transliterate(args, sl_sentence, tl_sentence)
        else:
            logging.error("ERROR: scol ({}) or tcol ({}) indexes above column number ({}) on line {}".format(args.scol, args.tcol, len(parts), nline))

        buf_sent.append(line)

        # Buffer sentences that are not empty and pass hardrules
        # buffer all sentences in raw mode
        if args.raw_output or (sl_sentence and tl_sentence \
                and (args.disable_hardrules or hardrules.wrong_tu(sl_sentence, tl_sentence) == False)):
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
    if logging.getLogger().level <= logging.DEBUG:
        verbose = 1
    else:
        verbose = 0

    # Classify predictions
    if len(buf_sent_tl) > 0 and len(buf_sent_sl) > 0:
        predictions = args.clf.predict(buf_sent_sl, buf_sent_tl,
                                       args.batch_size,
                                       args.calibrated,
                                       args.raw_output,
                                       verbose=verbose)
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
