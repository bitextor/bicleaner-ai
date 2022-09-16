from bicleaner_ai.word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
from bicleaner_ai.training import sentence_noise
from tempfile import TemporaryDirectory
from argparse import Namespace

import bicleaner_ai.bicleaner_ai_train as train


def test_sentence_noise():
    with TemporaryDirectory(prefix='bicleaner-ai-test.') as dir_:
        src_lang = 'en'
        trg_lang = 'fr'

        argv = [
            "--model_dir", dir_,
            "--source_lang", src_lang,
            "--target_lang", trg_lang,
            "--parallel_train", './corpus.en-fr',
            "--parallel_valid", './dev.en-fr',
            "--target_word_freqs", 'wordfreq-fr.gz',
            "--classifier_type", 'xlmr',
            "--seed", '42',
        ]

        # Pass args to parser
        args = train.get_arguments(argv)
        train.initialization(args)
        args.tl_word_freqs = WordZipfFreqDistDoubleLinked(args.target_word_freqs)

        src, trg = [], []
        for line in args.parallel_train:
            parts = line.strip().split('\t')
            src.append(parts[0])
            trg.append(parts[1])

        output = [
            "And anyways, isn't it supposed to be man versus machine?\tD'ailleurs, est-ce que ce n'est pas censé être l'homme contre la machine ?\t1",
            "Such in-depth analysis of the diversity of financial needs in different sectors could also be helpful in mobilizing increased political support for ODA.\tD'ailleurs, est-ce que ce n'est pas censé être l'homme contre la machine ?\t0",
            '"Eye for an Eye "\tD\'ailleurs, est-ce que ce n\'est pas censé être l\'homme contre la machine ?\t0', "Switzerland, Sweden, Thailand, Taiwan, Turkey, United States.\tD'ailleurs, est-ce que ce n'est pas censé être l'homme contre la machine ?\t0",
            "And anyways, isn't it supposed to be man versus machine?\tD'annexe, est-ce que ce n'est pas censé être à * contre la machine ?\t0",
            "And anyways, isn't it supposed to be man versus machine?\tD'ailleurs, est-ce que ce n'est pas censé être à homme contre la machine ?\t0",
            "And anyways, isn't it supposed to be man versus machine?\tD'ailleurs, est-ce que ce n'est pas censé être l'ni contre la machine ?\t0",
            "And anyways, isn't it supposed to be man versus machine?\tD', est-ce que ce est pas censé l'homme contre machine\t0",
            "And anyways, isn't it supposed to be man versus machine?\tD'ailleurs, est-ce ce n'est pas être homme contre la machine\t0",
            "And anyways, isn't it supposed to be man versus machine?\tD'ailleurs, que ce n'est pas censé être l'homme contre machine ?\t0"
        ]
        assert sentence_noise(1, src, trg, args) == output
