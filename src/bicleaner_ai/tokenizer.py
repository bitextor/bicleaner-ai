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
    def get_tokenizer(cls, tok_type, l='eng'):
        if tok_type == 'char':
            return CharTokenizer()
        else:
            return WordTokenizer(l)

    def tokenize(self, text):
        pass

    def detokenize(self, text):
        pass


class CharTokenizer(Tokenizer):
    ''' Separate each character with a space
        for languages like Chinese this can be a solution
    '''

    def tokenize(self, text):
        return [c for c in text.rstrip('\n')]

    def detokenize(self, text):
        return ''.join(text)


class WordTokenizer(Tokenizer):
    ''' Sacremoses tokenizer '''

    def __init__(self, l="en"):
        self.tokenizer = MosesTokenizer(lang=l)
        self.detokenizer = MosesDetokenizer(lang=l)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text, escape=False)

    def detokenize(self, text):
        return self.detokenizer.detokenize(text)
