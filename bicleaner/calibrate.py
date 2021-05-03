from multiprocessing import cpu_count
import numpy as np
import argparse
import yaml
import sys
import os

try:
    from .word_freqs_zipf import WordZipfFreqDist
    from .word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from .training import build_noise
    from .models import calibrate_output
    from .util import get_model,check_positive
except (ImportError, SystemError):
    from word_freqs_zipf import WordZipfFreqDist
    from word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from training import build_noise
    from models import calibrate_output
    from util import get_model,check_positive

parser = argparse.ArgumentParser()
parser.add_argument('metadata')
parser.add_argument('dev_file')
parser.add_argument('-b','--batch_size', default=32, type=int)
parser.add_argument('-p', '--processes', type=check_positive, default=max(1, cpu_count()-1), help="Number of process to use")
parser.add_argument('--block_size', type=check_positive, default=10000, help="Sentence pairs per block when apliying multiprocessing in the noise function")
parser.add_argument('-S', '--source_tokenizer_command', help="Source language tokenizer full command")
parser.add_argument('-T', '--target_tokenizer_command', help="Target language tokenizer full command")
parser.add_argument('-s', '--source_lang', required=True, help="Source language")
parser.add_argument('-t', '--target_lang', required=True, help="Target language")
# Negative sampling options
parser.add_argument('--pos_ratio', default=1, type=int)
parser.add_argument('--rand_ratio', default=3, type=int)
parser.add_argument('--womit_ratio', default=3, type=int)
parser.add_argument('--freq_ratio', default=3, type=int)
parser.add_argument('--fuzzy_ratio', default=0, type=int)
parser.add_argument('--neighbour_mix', default=False, type=bool)
args = parser.parse_args()

with open(args.metadata) as f:
    meta = yaml.safe_load(f)
print(meta)
clf = get_model(meta["classifier_type"])(os.path.dirname(args.metadata))
clf.load()

path = os.path.dirname(args.metadata)
# Load word frequencies
if "source_word_freqs" in meta:
    args.sl_word_freqs = WordZipfFreqDist(path + '/' + meta["source_word_freqs"])
if "target_word_freqs" in meta:
    args.tl_word_freqs = WordZipfFreqDistDoubleLinked(path + '/' + meta["target_word_freqs"])
else:
    args.tl_word_freqs = None

with open(args.dev_file) as f:
    dev_file_noise = build_noise(f, args)

src_sents = []
trg_sents = []
y_true = []
with open(dev_file_noise) as f:
    for line in f:
        parts = line.strip().split('\t')
        src_sents.append(parts[0])
        trg_sents.append(parts[1])
        y_true.append(parts[2])
os.unlink(dev_file_noise)
y_true = np.array(y_true, dtype=int)

y_pred = clf.predict(src_sents, trg_sents, args.batch_size, calibrated=False)

A, B = calibrate_output(y_true, y_pred)
print(A,B)
meta["calibration_params"] = [A, B]
with open(args.metadata, 'w') as f:
    yaml.dump(meta, f)
