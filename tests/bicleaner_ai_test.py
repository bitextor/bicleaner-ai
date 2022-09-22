import bicleaner_ai.bicleaner_ai_train as train
import bicleaner_ai.bicleaner_ai_classifier as classifier
from tempfile import TemporaryDirectory
from argparse import Namespace
from os.path import exists
import requests
import tarfile
import yaml
import os


def test_train_full():
    with TemporaryDirectory(prefix='bicleaner-ai-test.') as dir_:
        steps = 5
        epochs = 2
        batch = 8
        classifier_type = 'xlmr'
        src_lang = 'en'
        trg_lang = 'fr'

        argv = [
            "--model_dir", dir_,
            "--source_lang", src_lang,
            "--target_lang", trg_lang,
            "--parallel_train", './corpus.en-fr',
            "--parallel_valid", './dev.en-fr',
            "--model_name", 'bicleaner-ai-test-full-en-fr',
            "--target_word_freqs", 'wordfreq-fr.gz',
            "--save_train", dir_ + '/train.en-fr',
            "--save_valid", dir_ + '/valid.en-fr',
            "--seed", '42',
            "--classifier_type", classifier_type,
            "--batch_size", str(batch),
            "--steps_per_epoch", str(steps),
            "--epochs", str(epochs),
        ]

        # Pass args to parser
        args = train.get_arguments(argv)
        train.initialization(args)
        # Launch training
        train.main(args)

        # Check produced files
        assert exists(f'{dir_}/metadata.yaml')
        assert exists(f'{dir_}/tf_model.h5')
        assert exists(f'{dir_}/config.json')
        assert exists(f'{dir_}/tokenizer.json')
        assert exists(f'{dir_}/tokenizer_config.json')
        assert exists(f'{dir_}/sentencepiece.bpe.model')
        assert exists(f'{dir_}/special_tokens_map.json')
        assert exists(f'{dir_}/train.en-fr')
        assert exists(f'{dir_}/valid.en-fr')

        # Check metadata fieldsPass args to parser
        with open(f'{dir_}/metadata.yaml') as metadata:
            yml = yaml.safe_load(metadata)

        clf_set = yml['classifier_settings']
        assert type(clf_set['calibration_params']) == list
        assert clf_set['steps_per_epoch'] == steps
        assert clf_set['epochs'] == epochs
        assert clf_set['batch_size'] == batch
        assert yml['classifier_type'] == classifier_type


def test_train_lite():
    with TemporaryDirectory(prefix='bicleaner-ai-test.') as dir_:
        steps = 5
        epochs = 2
        batch = 32
        classifier_type = 'dec_attention'
        src_lang = 'en'
        trg_lang = 'fr'
        vocab_size = 8000

        argv = [
            "--model_dir", dir_,
            "--source_lang", src_lang,
            "--target_lang", trg_lang,
            "--mono_train", './mono.en-fr',
            "--parallel_train", './corpus.en-fr',
            "--parallel_valid", './dev.en-fr',
            "--model_name", 'bicleaner-ai-test-full-en-fr',
            "--target_word_freqs", 'wordfreq-fr.gz',
            "--save_train", dir_ + '/train.en-fr',
            "--save_valid", dir_ + '/valid.en-fr',
            "--seed", '42',
            "--classifier_type", classifier_type,
            "--batch_size", str(batch),
            "--steps_per_epoch", str(steps),
            "--epochs", str(epochs),
        ]

        # Pass args to parser
        args = train.get_arguments(argv)
        train.initialization(args)
        args.vocab_size = vocab_size
        # Launch training
        train.main(args)

        # Check produced files
        assert exists(f'{dir_}/metadata.yaml')
        assert exists(f'{dir_}/model.h5')
        assert exists(f'{dir_}/glove.vectors')
        assert exists(f'{dir_}/spm.model')
        assert exists(f'{dir_}/spm.vocab')
        assert exists(f'{dir_}/train.en-fr')
        assert exists(f'{dir_}/valid.en-fr')

        # Check metadata fields
        with open(f'{dir_}/metadata.yaml') as metadata:
            yml = yaml.safe_load(metadata)

        clf_set = yml['classifier_settings']
        assert type(clf_set['calibration_params']) == list
        assert clf_set['steps_per_epoch'] == steps
        assert clf_set['epochs'] == epochs
        assert clf_set['batch_size'] == batch
        assert clf_set['vocab_size'] == vocab_size
        assert yml['classifier_type'] == classifier_type


def setup_module():
    ''' Download models for classifier test '''
    def download_model(filename, url):
        if not exists(filename):
            download = requests.get(url, stream=True)
            with open(filename, 'wb') as file_:
                file_.writelines(download.iter_content(1024))

    url = 'https://github.com/bitextor/bicleaner-ai-data/releases/download/v1.0/full-en-fr.tgz'
    download_model('./en-fr-full.tgz', url)

    url = 'https://github.com/bitextor/bicleaner-ai-data/releases/download/v1.0/lite-en-fr.tgz'
    download_model('./en-fr-lite.tgz', url)


def test_classify_lite():
    # Create temp dir
    with TemporaryDirectory(prefix='bicleaner-ai-classify-test.') as dir_:
        # Extract model
        with tarfile.open('./en-fr-lite.tgz') as file_:
            file_.extractall(dir_)

        # Define program arguments
        argv = [
            '--disable_hardrules',
            '--scol', '1',
            '--tcol', '2',
            '--score_only',
            './dev.en-fr',
            #'/dev/stdout',
            dir_ + '/scores',
            dir_ + '/en-fr',
        ]

        # Read classifier output scores
        def read_scores(filename):
            scores = []
            with open(filename) as f:
                for line in f:
                    scores.append(float(line.strip()))
            return scores

        # Run classifier
        args = classifier.initialization(argv)
        classifier.main(args)
        args.output.flush()

        # Test normal output
        scores = read_scores(dir_ + '/scores')
        assert scores == [0.856, 1.000, 0.930, 0.140, 1.000, 1.000, 0.051, 0.027, 0.922, 0.855]

        # Run classifier with calibrated option
        argv.insert(0, '--calibrated')
        args = classifier.initialization(argv)
        classifier.main(args)
        args.output.flush()

        # Test calibrated output
        scores = read_scores(dir_ + '/scores')
        assert scores == [0.672, 0.706, 0.690, 0.478, 0.706, 0.706, 0.453, 0.447, 0.688, 0.672]


def test_classify_full():
    # Create temp dir
    with TemporaryDirectory(prefix='bicleaner-ai-classify-test.') as dir_:
        # Extract model
        with tarfile.open('./en-fr-full.tgz') as file_:
            file_.extractall(dir_)

        # Define program arguments
        argv = [
            '--disable_hardrules',
            '--scol', '1',
            '--tcol', '2',
            '--score_only',
            './test.en-fr',
            #'/dev/stdout',
            dir_ + '/scores',
            dir_ + '/en-fr',
        ]

        # Read classifier output scores
        def read_scores(filename, tabs=False):
            scores = []
            with open(filename) as f:
                for line in f:
                    if tabs:
                        parts = line.strip().split('\t')
                        scores.append((float(parts[0]), float(parts[1])))
                    else:
                        scores.append(float(line.strip()))
            return scores

        # Run classifier
        args = classifier.initialization(argv)
        classifier.main(args)
        args.output.flush()

        # Test normal output
        scores = read_scores(dir_ + '/scores')
        assert scores == [0.565, 0.985, 0.018, 0.695, 0.932, 0.928, 0.967, 0.956, 0.747, 0.464]

        # Run classifier with calibrated option
        argv.insert(0, '--calibrated')
        args = classifier.initialization(argv)
        classifier.main(args)
        args.output.flush()

        # Test calibrated output
        scores = read_scores(dir_ + '/scores')
        assert scores == [0.837, 0.993, 0.065, 0.934, 0.989, 0.989, 0.992, 0.991, 0.955, 0.699]

        # Run classifier with calibrated option
        argv[0] = '--raw_output'
        args = classifier.initialization(argv)
        classifier.main(args)
        args.output.flush()

        # Test calibrated output
        scores = read_scores(dir_ + '/scores', tabs=True)
        assert scores == [(-0.302, -0.039),
                          (-2.280, 1.922),
                          (1.915, -2.087),
                          (-0.612, 0.210),
                          (-1.545, 1.079),
                          (-1.415, 1.144),
                          (-1.917, 1.472),
                          (-1.773, 1.312),
                          (-0.743, 0.340),
                          (-0.079, -0.222),]
