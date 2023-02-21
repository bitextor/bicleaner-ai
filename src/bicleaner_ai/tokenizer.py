from sacremoses import MosesTokenizer, MosesDetokenizer
from toolwrapper import ToolWrapper
from subprocess import run, PIPE
import logging
import sys
import os

try:
    from .util import no_escaping
except (SystemError, ImportError):
    from util import  no_escaping


class Tokenizer:
    def __init__(self, l="en"):
        self.tokenizer = MosesTokenizer(lang=l)
        self.detokenizer = MosesDetokenizer(lang=l)
        self.cmd = None

    def tokenize(self, text):
        if isinstance(text, list):
            return [self.tokenizer.tokenize(line, escape=False) for line in text]
        else:
            return self.tokenizer.tokenize(text, escape=False)

    def detokenize(self, text):
        elif self.detokenizer is not None:
            return self.detokenizer.detokenize(text)
        else:
            return ' '.join(text)
