#!/usr/bin/env python

import os
import argparse
import logging
import re
import regex
import sys
from os import path
import typing
import random

from tempfile import TemporaryFile
from toolwrapper import ToolWrapper

try:
    from .models import DecomposableAttention, Transformer, BCXLMRoberta
except (SystemError, ImportError):
    from models import DecomposableAttention, Transformer, BCXLMRoberta

# variables used by the no_escaping function
replacements = {"&amp;":  "&",
                "&#124;": "|",
                "&lt;":   "<",
                "&gt;":   ">",
                "&apos":  "'",
                "&quot;": '"',
                "&#91;":  "[",
                "&#93;":  "]"}

substrs = sorted(replacements, key=len, reverse=True)
nrregexp = re.compile('|'.join(map(re.escape, substrs)))

regex_alpha = regex.compile("^[[:alpha:]]+$")

# Return model class according to its cli alias
model_classes = {
    "dec_attention": DecomposableAttention,
    "transformer": Transformer,
    "xlmr": BCXLMRoberta,
}
def get_model(model_type):
    return model_classes[model_type]

# Back-replacements of strings mischanged by the Moses tokenizer
def no_escaping(text):
    global nrregexp, replacements
    return nrregexp.sub(lambda match: replacements[match.group(0)], text)

# Check if the argument is a directory
def check_dir(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

# Check if the argument of a program (argparse) is positive or zero
def check_positive_between_zero_and_one(value):
    ivalue = float(value)
    if ivalue < 0 or ivalue > 1:
        raise argparse.ArgumentTypeError("%s is an invalid float value between 0 and 1" % value)
    return ivalue

# Check if the argument of a program (argparse) is positive or zero
def check_positive_or_zero(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

# Check if the argument of a program (argparse) is strictly positive
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

# Check if the argument of a program (argparse) is strictly positive
def check_if_folder(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError("%s is not a directory" % path)
    return path

# Logging config
def logging_setup(args = None):
    
    logger = logging.getLogger()
    logger.handlers = [] # Removing default handler to avoid duplication of log messages
    logger.setLevel(logging.ERROR)
    
    h = logging.StreamHandler(sys.stderr)
    if args != None:
       h = logging.StreamHandler(args.logfile)
      
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)

    #logger.setLevel(logging.INFO)
    
    if args != None:
        if not args.quiet:
            logger.setLevel(logging.INFO)
        if args.debug:
            logger.setLevel(logging.DEBUG)

    logging_level = logging.getLogger().level
    if logging_level <= logging.WARNING and logging_level != logging.DEBUG:
        logging.getLogger("ToolWrapper").setLevel(logging.WARNING)

    if logging.getLogger().level != logging.DEBUG:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')


def check_gpu(require_gpu: bool):
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU') + tf.config.list_physical_devices('TPU')
    if not devices:
        if require_gpu:
            # Exit with a 75 EX_TEMPFAIL, which indicates a temporary error. GPUs can
            # become unavailable in the cloud, and 75 indicates that the task can be
            # retried.
            logging.error("No GPU or TPU detected and --require_gpu was specified. Exiting with a EX_TEMPFAIL (75)")
            sys.exit(75)
        else:
            logging.warning("No GPU or TPU was detected. Running on CPU will be slow.")


def shuffle_file(input: typing.TextIO, output: typing.TextIO):
    offsets=[]
    with TemporaryFile("w+") as temp:
        count = 0
        for line in input:
            offsets.append(count)
            count += len(bytearray(line, "UTF-8"))
            temp.write(line)
        temp.flush()
        
        random.shuffle(offsets)
        
        for offset in offsets:
            temp.seek(offset)
            output.write(temp.readline())
