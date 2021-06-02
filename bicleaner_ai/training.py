from multiprocessing import Queue, Process, Value, cpu_count
from heapq import heappush, heappop
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from tempfile import TemporaryFile, NamedTemporaryFile
from fuzzywuzzy import process, fuzz
import logging
import os
import random
import fasttext

try:
    from .tokenizer import Tokenizer
except (SystemError, ImportError):
    from tokenizer import Tokenizer

# Porn removal classifier
# training, compressing, run tests and save model file
def train_porn_removal(args):
    if args.porn_removal_train is None or args.porn_removal_file is None:
        return

    logging.info("Training porn removal classifier.")
    model = fasttext.train_supervised(args.porn_removal_train.name,
                                    thread=args.processes,
                                    lr=1.0,
                                    epoch=25,
                                    minCount=5,
                                    wordNgrams=1,
                                    verbose=0)
    logging.info("Compressing classifier.")
    model.quantize(args.porn_removal_train.name,
                retrain=True,
                thread=args.processes,
                verbose=0)

    if args.porn_removal_test is not None:
        N, p, r = model.test(args.porn_removal_test.name, threshold=0.5)
        logging.info("Precision:\t{:.3f}".format(p))
        logging.info("Recall:\t{:.3f}".format(r))

    logging.info("Saving porn removal classifier.")
    model.save_model(args.porn_removal_file)

# Generate negative and positive samples for a sentence pair
def sentence_noise(i, src, trg, args):
    size = len(src)
    sts = []
    src_strip = src[i].strip()
    trg_strip = trg[i].strip()

    # Positive samples
    for j in range(args.pos_ratio):
        sts.append(src_strip + "\t" + trg_strip+ "\t1")

    # Random misalignment
    for j in range(args.rand_ratio):
        sts.append(src[random.randrange(1,size)].strip() + "\t" + trg_strip + "\t0")

    # Frequence based noise
    tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)
    for j in range(args.freq_ratio):
        t_toks = tokenizer.tokenize(trg[i])
        replaced = replace_freq_words(t_toks, args.tl_word_freqs)
        if replaced is not None:
            sts.append(src_strip + "\t" + tokenizer.detokenize(replaced) + "\t0")

    # Randomly omit words
    tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)
    for j in range(args.womit_ratio):
        t_toks = tokenizer.tokenize(trg[i])
        omitted = omit_words(t_toks)
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

    while True:
        job = jobs_queue.get()

        if job is not None:
            logging.debug("Job {0}".format(job.__repr__()))

            # Generate noise for each sentence in the block
            output = []
            for i in range(job, min(job+args.block_size, nlines)):
                output.extend(sentence_noise(i, src, trg, args))

            output_file = NamedTemporaryFile('w+', delete=False)
            for j in output:
                output_file.write(j + '\n')
            output_file.close()
            output_queue.put((job,output_file.name))
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

            with open(filein_name, 'r') as filein:
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

        with open(filein_name, 'r') as filein:
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
def replace_freq_words(sentence, double_linked_zipf_freqs):
    count = 0
    sent_orig = sentence[:]
    # Loop until any of the chosen words have an alternative, at most 3 times
    while True:
        # Random number of words that will be replaced
        num_words_replaced = random.randint(1, len(sentence))
        # Replacing N words at random positions
        idx_words_to_replace = random.sample(range(len(sentence)), num_words_replaced)

        for wordpos in idx_words_to_replace:
            w = sentence[wordpos]
            wfreq = double_linked_zipf_freqs.get_word_freq(w)
            alternatives = double_linked_zipf_freqs.get_words_for_freq(wfreq)
            if alternatives is not None:
                alternatives = list(alternatives)

                # Avoid replace with the same word
                if w.lower() in alternatives:
                    alternatives.remove(w.lower())
                if not alternatives == []:
                    sentence[wordpos] = random.choice(alternatives)
        count += 1
        if sentence != sent_orig:
            break
        elif count >= 3:
            return None

    return sentence

# Randomly omit words in a sentence
def omit_words(sentence):
    num_words_deleted = random.randint(1, len(sentence))
    idx_words_to_delete = sorted(random.sample(range(len(sentence)), num_words_deleted), reverse=True)
    for wordpos in idx_words_to_delete:
        del sentence[wordpos]
    return sentence

def repr_right(numeric_list, numeric_fmt = "{:1.4f}"):
    result_str = ["["]
    for i in range(len(numeric_list)):
        result_str.append(numeric_fmt.format(numeric_list[i]))
        if i < (len(numeric_list)-1):
            result_str.append(", ")
        else:
            result_str.append("]")
    return "".join(result_str)

# Check if a file path is relative to a path
def check_relative_path(path, filepath):
    file_abs = os.path.abspath(filepath)
    path_abs = os.path.abspath(path.rstrip('/')) # remove trailing / for safety
    return file_abs.replace(path_abs + '/', '').count('/') == 0

# Write YAML with the training parameters and quality estimates
def write_metadata(args, classifier, y_true, y_pred, lm_stats):
    out = args.metadata

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    out.write(f"precision_score: {precision:.3f}\n")
    out.write(f"recall_score: {recall:.3f}\n")
    out.write(f"f1_score: {f1:.3f}\n")
    out.write(f"matthews_corr_coef: {mcc:.3f}\n")


    # Writing it by hand (not using YAML libraries) to preserve the order
    out.write(f"source_lang: {args.source_lang}\n")
    out.write(f"target_lang: {args.target_lang}\n")

    # Save base names only if directories are relative
    if check_relative_path(args.model_dir, args.porn_removal_file):
        porn_removal_file = os.path.basename(args.porn_removal_file)
    else:
        porn_removal_file = args.porn_removal_file
    if check_relative_path(args.model_dir, args.lm_file_sl):
        lm_file_sl = os.path.basename(args.lm_file_sl)
    else:
        lm_file_sl = args.lm_file_sl
    if check_relative_path(args.model_dir, args.lm_file_tl):
        lm_file_tl = os.path.basename(args.lm_file_tl)
    else:
        lm_file_tl = args.lm_file_tl

    if args.porn_removal_file is not None and args.porn_removal_train is not None:
        out.write(f"porn_removal_file: {porn_removal_file}\n")
        out.write(f"porn_removal_side: {args.porn_removal_side}\n")

    if lm_stats is not None and args.lm_file_sl is not None and args.lm_file_tl is not None:
        out.write(f"source_lm: {lm_file_sl}\n")
        out.write(f"target_lm: {lm_file_tl}\n")
        out.write(f"lm_type: CHARACTER\n")
        out.write(f"clean_mean_perp: {lm_stats.clean_mean}\n")
        out.write(f"clean_stddev_perp: {lm_stats.clean_stddev}\n")
        out.write(f"noisy_mean_perp: {lm_stats.noisy_mean}\n")
        out.write(f"noisy_stddev_perp: {lm_stats.noisy_stddev}\n")

    if args.source_tokenizer_command is not None:
        out.write(f"source_tokenizer_command: {args.source_tokenizer_command}\n")
    if args.target_tokenizer_command is not None:
        out.write(f"target_tokenizer_command: {args.target_tokenizer_command}\n")

    # Save classifier
    out.write(f"classifier_type: {args.classifier_type}\n")

    # Save classifier train settings
    out.write("classifier_settings:\n")
    for key in sorted(classifier.settings.keys()):
        # Don't print objects
        if type(classifier.settings[key]) in [int, str, list, tuple]:
            if type(classifier.settings[key]) in [list, tuple]:
                out.write("    " + key + ": " + repr_right(classifier.settings[key], "{:.8f}") + "\n")
            else:
                out.write("    " + key + ": " + str(classifier.settings[key]) + "\n")
