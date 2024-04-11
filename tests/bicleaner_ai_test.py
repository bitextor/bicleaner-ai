import bicleaner_ai.bicleaner_ai_train as train
import bicleaner_ai.bicleaner_ai_classifier as classifier
from tempfile import TemporaryDirectory
from argparse import Namespace
from os.path import exists
import tensorflow as tf
import requests
import tarfile
import yaml
import os
tf.config.run_functions_eagerly(True)


def test_train_full():
    with TemporaryDirectory(prefix='bicleaner-ai-test.') as dir_:
        steps = 5
        epochs = 2
        batch = 2
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
            "--generated_train", dir_ + '/train.en-fr',
            "--generated_valid", dir_ + '/valid.en-fr',
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
        train.perform_training(args)

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

        del args


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
            "--generated_train", dir_ + '/train.en-fr',
            "--generated_valid", dir_ + '/valid.en-fr',
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
        train.perform_training(args)

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

        del args


def download_model(filename, url):
    ''' Download models for classifier test '''
    if not exists(filename):
        download = requests.get(url, stream=True)
        with open(filename, 'wb') as file_:
            file_.writelines(download.iter_content(1024))


def test_classify_lite():
    url = 'https://github.com/bitextor/bicleaner-ai-data/releases/download/v1.0/lite-en-fr.tgz'
    modelfile = './en-fr-lite.tgz'
    download_model(modelfile, url)

    # Create temp dir
    with TemporaryDirectory(prefix='bicleaner-ai-classify-test.') as dir_:
        # Extract model
        with tarfile.open(modelfile) as file_:
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
        classifier.perform_classification(args)
        args.output.flush()

        # Test normal output
        scores = read_scores(dir_ + '/scores')
        assert scores == [0.856, 1.000, 0.930, 0.140, 1.000, 1.000, 0.051, 0.027, 0.922, 0.855]

        # Run classifier with calibrated option
        argv.insert(0, '--calibrated')
        args = classifier.initialization(argv)
        classifier.perform_classification(args)
        args.output.flush()

        # Test calibrated output
        scores = read_scores(dir_ + '/scores')
        assert scores == [0.672, 0.706, 0.690, 0.478, 0.706, 0.706, 0.453, 0.447, 0.688, 0.672]


def test_classify_full():
    model = 'bitextor/bicleaner-ai-full-en-fr'

    # Create temp dir
    with TemporaryDirectory(prefix='bicleaner-ai-classify-test.') as dir_:
        # Define program arguments
        argv = [
            '--disable_hardrules',
            '--scol', '1',
            '--tcol', '2',
            '--score_only',
            './dev.en-fr',
            #'/dev/stdout',
            dir_ + '/scores',
            model,
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
        classifier.perform_classification(args)
        args.output.flush()

        # Test normal output
        scores = read_scores(dir_ + '/scores')
        assert scores == [0.675, 0.999, 0.005, 0.863, 0.918, 0.999, 0.940, 0.950, 0.999, 0.723]

        # Run classifier with calibrated option
        argv.insert(0, '--calibrated')
        args = classifier.initialization(argv)
        classifier.perform_classification(args)
        args.output.flush()

        # Test calibrated output
        scores = read_scores(dir_ + '/scores')
        assert scores == [0.595, 0.666, 0.440, 0.637, 0.649, 0.666, 0.653, 0.655, 0.666, 0.606]

        # Run classifier with calibrated option
        argv[0] = '--raw_output'
        args = classifier.initialization(argv)
        classifier.perform_classification(args)
        args.output.flush()

        # Test calibrated output
        scores = read_scores(dir_ + '/scores', tabs=True)
        assert scores == [(-0.381, 0.351),
                          (-3.303, 3.428),
                          (2.714, -2.618),
                          (-0.885, 0.953),
                          (-1.167, 1.243),
                          (-3.524, 3.697),
                          (-1.355, 1.401),
                          (-1.445, 1.492),
                          (-3.293, 3.596),
                          (-0.432, 0.530),
                         ]

