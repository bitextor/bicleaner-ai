from multiprocessing import Queue, Process, Value, cpu_count
from tempfile import TemporaryFile, NamedTemporaryFile, mktemp
from heapq import heappush, heappop
from fuzzywuzzy import process, fuzz
import zstandard
import argparse
import logging
import random
import sys
import os

try:
    from .util import check_positive
    from .tokenizer import Tokenizer
    from .word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
except (SystemError, ImportError):
    from tokenizer import Tokenizer
    from util import check_positive
    from word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked


'''
Add to an argument parser the synthetic noise generation options
'''
def add_noise_options(parser):
    #groupO.add_argument('-f', '--source_word_freqs', type=argparse.FileType('r'), default=None, required=False, help="L language gzipped list of word frequencies")
    parser.add_argument('-F', '--target_word_freqs', type=argparse.FileType('r'), default=None, required=False, help="R language gzipped list of word frequencies (needed for frequence based noise)")
    parser.add_argument('--target_tokenizer_type', choices=['word', 'char'], default='word', help='Type of tokenization for noise generation.')
    parser.add_argument('--block_size', type=check_positive, default=2000, help="Sentence pairs per block when apliying multiprocessing in the noise function")
    parser.add_argument('-p', '--processes', default=None, help="Option no longer available, please set BICLEANER_AI_THREADS environment variable")
    parser.add_argument('--pos_ratio', default=1, type=int, help="Ratio of positive samples used to oversample on validation and test sets")
    parser.add_argument('--rand_ratio', default=3, type=int, help="Ratio of negative samples misaligned randomly")
    parser.add_argument('--womit_ratio', default=3, type=int, help="Ratio of negative samples misaligned by randomly omitting words")
    parser.add_argument('--freq_ratio', default=3, type=int, help="Ratio of negative samples misaligned by replacing words by frequence (needs --target_word_freq)")
    parser.add_argument('--fuzzy_ratio', default=0, type=int, help="Ratio of negative samples misaligned by fuzzy matching")
    parser.add_argument('--neighbour_mix', default=False, type=bool, help="If use negative samples misaligned by neighbourhood")
    parser.add_argument('--min_omit_words', default=1, type=int, help="Minimum words to omit per sentence in omit noise")
    parser.add_argument('--min_freq_words', default=1, type=int, help="Minimum words to replace per sentence in freq noise")


'''
Initialization and setup of arguments related to noise generation
'''
def setup_noise(args):
    # We want to fail if no freq_words is provided and freq noise is requested (which is by default)
    # in case validation and training are pre-generated, noise generation is skipped, so no need to fail
    if args.freq_ratio > 0 and args.target_word_freqs is None \
            and (not args.generated_train and not args.generated_valid):
        logging.error("Frequence based noise needs target language word frequencies. Use '-F'/'--target_word_freqs' or disable with '--freq-ratio 0'")
        sys.exit(1)

    # Warn about args.processes deprecation
    if args.processes is not None:
        logging.warning("--processes option is not available anymore, please use BICLEANER_AI_THREADS environment variable instead.")

    # Set the number of processes from the environment variable
    # or instead use all cores
    if "BICLEANER_AI_THREADS" in os.environ and os.environ["BICLEANER_AI_THREADS"]:
        args.processes = int(os.environ["BICLEANER_AI_THREADS"])
    else:
        args.processes = max(1, cpu_count()-1)

    # Load word frequencies
    #if args.source_word_freqs:
    #    args.sl_word_freqs = WordZipfFreqDist(args.source_word_freqs)
    if args.target_word_freqs:
        args.tl_word_freqs = WordZipfFreqDistDoubleLinked(args.target_word_freqs)
    else:
        args.tl_word_freqs = None


# Generate negative and positive samples for a sentence pair
def sentence_noise(i, src, trg, args, tokenizer):
    size = len(src)
    sts = []
    src_strip = src[i].strip()
    trg_strip = trg[i].strip()

    # When omit or freq noise is enabled, modify sentences
    # starting with lowercase (with 50% probability)
    # This avoids the missing starting capital letter flaw where sentences
    # without it are always scored low
    if args.freq_ratio or args.womit_ratio:
        generate = False

        # Lowercase src only, target only, both or neither randomly
        if src_strip[0].isupper() and random.getrandbits(1):
            src_strip = src_strip[0].lower() + src_strip[1:]

        if trg_strip[0].isupper() and random.getrandbits(1):
            trg_strip = trg_strip[0].lower() + trg_strip[1:]


    # Positive samples
    for j in range(args.pos_ratio):
        sts.append(src_strip + "\t" + trg_strip+ "\t1")

    # Random misalignment
    for j in range(args.rand_ratio):
        sts.append(src[random.randrange(1,size)].strip() + "\t" + trg_strip + "\t0")

    # Frequence based noise
    for j in range(args.freq_ratio):
        t_toks = tokenizer.tokenize(trg[i])
        replaced = replace_freq_words(t_toks, args.tl_word_freqs, args.min_freq_words)
        if replaced is not None:
            sts.append(src_strip + "\t" + tokenizer.detokenize(replaced) + "\t0")

    # Randomly omit words
    for j in range(args.womit_ratio):
        t_toks = tokenizer.tokenize(trg[i])
        omitted = omit_words(t_toks, args.min_omit_words)
        if omitted != []:
            sts.append(src_strip + "\t" + tokenizer.detokenize(omitted) + "\t0")

    # Misalginment by fuzzy matching
    if args.fuzzy_ratio > 0:
        explored = {n:trg[n] for n in random.sample(range(size), min(3000, size))}
        matches = process.extract(trg[i], explored,
                                  scorer=fuzz.token_sort_ratio,
                                  limit=25)
        m_index = [m[2] for m in matches if m[1]<70][:args.fuzzy_ratio]
        for m in m_index:
            sts.append(src_strip + "\t" + trg[m].strip() + "\t0")

    # Misalgniment with neighbour sentences
    if args.neighbour_mix and i <size-2 and i > 1:
        sts.append(src_strip + "\t" + trg[i+1].strip()+ "\t0")
        sts.append(src_strip + "\t" + trg[i-1].strip()+ "\t0")

    return sts

# Take block number from the queue and generate noise for that block
def worker_process(num, src, trg, jobs_queue, output_queue, args):
    nlines = len(src)
    tokenizer = Tokenizer.get_tokenizer(
                    args.target_tokenizer_type,
                    args.target_lang)

    while True:
        job = jobs_queue.get()

        if job is not None:
            logging.debug("Job {0}".format(job.__repr__()))

            # Generate noise for each sentence in the block
            output = []
            for i in range(job, min(job+args.block_size, nlines)):
                output.extend(sentence_noise(i, src, trg, args, tokenizer))

            output_file_name = mktemp()
            output_file = zstandard.open(output_file_name, 'wt')
            for j in output:
                output_file.write(j + '\n')
            output_file.close()
            output_queue.put((job,output_file_name))
        else:
            logging.debug(f"Exiting worker {num}")
            break

# Merges all the temporary files from the workers
def reduce_process(output_queue, output_file, block_size):
    h = []
    last_block = 0
    while True:
        logging.debug("Reduce: heap status {0}".format(h.__str__()))
        while len(h) > 0 and h[0][0] == last_block:
            nblock, filein_name = heappop(h)
            last_block += block_size

            with zstandard.open(filein_name, 'rt') as filein:
                for i in filein:
                    output_file.write(i)
            os.unlink(filein_name)

        job = output_queue.get()
        if job is not None:
            nblock, filein_name = job
            heappush(h, (nblock, filein_name))
        else:
            logging.debug("Exiting reduce loop")
            break

    if len(h) > 0:
        logging.debug(f"Still elements in heap: {h}")

    while len(h) > 0 and h[0][0] == last_block:
        nblock, filein_name = heappop(h)
        last_block += block_size

        with zstdandard.open(filein_name, 'rt') as filein:
            for i in filein:
                output_file.write(i)

        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")

    output_file.close()


# Parallel loop over input sentences to generate noise
def build_noise(input, args):
    src = []
    trg = {}
    # Read sentences into memory
    for i, line in enumerate(input):
        parts = line.rstrip("\n").split("\t")
        src.append(parts[0])
        trg[i] = parts[1]
    size = len(src)

    logging.debug("Running {0} workers at {1} rows per block".format(args.processes, args.block_size))
    process_count = max(1, args.processes)
    maxsize = 1000 * process_count
    output_queue = Queue(maxsize = maxsize)
    worker_count = process_count
    output_file = NamedTemporaryFile('w+', delete=False)

    # Start reducer
    reduce = Process(target = reduce_process,
                     args   = (output_queue, output_file, args.block_size))
    reduce.start()

    # Start workers
    jobs_queue = Queue(maxsize = maxsize)
    workers = []
    for i in range(worker_count):
        worker = Process(target = worker_process,
                         args   = (i, src, trg, jobs_queue, output_queue, args))
        worker.daemon = True # dies with the parent process
        worker.start()
        workers.append(worker)

    # Map jobs
    for i in range(0, size, args.block_size):
        jobs_queue.put(i)

    # Worker termination
    for _ in workers:
        jobs_queue.put(None)

    for w in workers:
        w.join()

    # Reducer termination
    output_queue.put(None)
    reduce.join()

    return output_file.name

# Randomly replace words with other words of same frequency
def replace_freq_words(sentence, double_linked_zipf_freqs, min_words=1):
    if len(sentence) <= min_words:
        return None # Don't apply noise if sent short than minimum words replaced
    count = 0
    sent_orig = sentence[:]
    # Loop until any of the chosen words have an alternative, at most 3 times
    while True:
        # Random number of words that will be replaced
        num_words_replaced = random.randint(min_words, len(sentence))
        # Replacing N words at random positions
        idx_words_to_replace = random.sample(range(len(sentence)), num_words_replaced)

        for wordpos in idx_words_to_replace:
            w = sentence[wordpos]
            wfreq = double_linked_zipf_freqs.get_word_freq(w)
            alternatives = double_linked_zipf_freqs.get_words_for_freq(wfreq)
            if alternatives is not None:
                alternatives = list(sorted(alternatives))

                # Avoid replace with the same word
                if w.lower() in alternatives:
                    alternatives.remove(w.lower())
                if not alternatives == []:
                    sentence[wordpos] = random.choice(alternatives)

                    # Restore starting capital letter with 50% prob
                    if wordpos == 0 and w[0].isupper() and random.getrandbits(1):
                        sentence[wordpos] = sentence[wordpos].capitalize()
        count += 1
        if sentence != sent_orig:
            break
        elif count >= 3:
            return None

    return sentence

# Randomly omit words in a sentence
def omit_words(sentence, min_omit=1):
    if len(sentence) <= min_omit:
        return []
    num_words_deleted = random.randint(min_omit, len(sentence)-1)
    idx_words_to_delete = sorted(random.sample(range(len(sentence)), num_words_deleted), reverse=True)
    for wordpos in idx_words_to_delete:
        del sentence[wordpos]
    # Restore starting capital letter with 50% prob
    if sentence[0][0].isupper() and random.getrandbits(1):
        sentence[0] = sentence[0].capitalize()
    return sentence


