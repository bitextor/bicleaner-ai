from bicleaner_ai.word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
from bicleaner_ai.training import sentence_noise
from bicleaner_ai.tokenizer import Tokenizer
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

        output = ["and anyways, isn't it supposed to be man versus machine?\tD'ailleurs, est-ce que ce n'est pas censé être l'homme contre la machine ?\t1",
                "Switzerland, Sweden, Thailand, Taiwan, Turkey, United States.\tD'ailleurs, est-ce que ce n'est pas censé être l'homme contre la machine ?\t0",
                "Public expenditure must yield something in return - whether it is food quality, the preservation of the environment and animal welfare, landscapes, cultural heritage, or enhancing social balance and equity.\tD'ailleurs, est-ce que ce n'est pas censé être l'homme contre la machine ?\t0",
                "It has a different threshold to the one we have in Europe.\tD'ailleurs, est-ce que ce n'est pas censé être l'homme contre la machine ?\t0",
                "and anyways, isn't it supposed to be man versus machine?\tD'annexe, est-ce que ce n'est pas censé être à homme contre la machine ?\t0",
                "and anyways, isn't it supposed to be man versus machine?\tD'ailleurs, est-ce que ce n'est pas censé être à homme contre la machine ?\t0",
                "and anyways, isn't it supposed to be man versus machine?\tD'ailleurs, est-ce que ce n'est pas censé être l'ni contre la machine ?\t0",
                "and anyways, isn't it supposed to be man versus machine?\tD', est-ce que ce est pas censé l'homme contre machine\t0",
                "and anyways, isn't it supposed to be man versus machine?\test-ce ce être contre la\t0",
                "and anyways, isn't it supposed to be man versus machine?\tce est pas l'la\t0"
        ]

        out = sentence_noise(1, src, trg, args)
        assert out == output


def test_tokenizer():
    tok = Tokenizer(None, 'fr')
    sent = "D'ailleurs, est-ce que ce n'est pas censé être l'homme contre la machine ?"

    assert tok.tokenize(sent) == ["D'", 'ailleurs', ',',
            'est-ce', 'que', 'ce', "n'", 'est', 'pas',
            'censé', 'être', "l'", 'homme', 'contre',
            'la', 'machine', '?']

