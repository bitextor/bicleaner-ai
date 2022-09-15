import bicleaner_ai.bicleaner_ai_train as train
from tempfile import TemporaryDirectory
from argparse import Namespace

def test_train_full():
    with TemporaryDirectory(prefix='bicleaner-ai-test.') as dir_:
        argv = [
            "--model_dir", dir_,
            "--source_lang", 'en',
            "--target_lang", 'fr',
            "--parallel_train", './corpus.en-fr',
            "--parallel_valid", './dev.en-fr',
            "--model_name", 'bicleaner-ai-test-full-en-fr',
            "--target_word_freqs", 'wordfreq-fr.gz',
            "--save_train", dir_ + '/train.en-fr',
            "--save_valid", dir_ + '/valid.en-fr',
            "--seed", '42',
            "--classifier_type", 'xlmr',
            "--batch_size", '8',
            "--steps_per_epoch", '5',
            "--epochs", '2',
        ]

        args = train.get_arguments(argv)
        train.initialization(args)
        train.main(args)
