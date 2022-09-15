import bicleaner_ai.bicleaner_ai_train as train
from tempfile import TemporaryDirectory
from argparse import Namespace
from os.path import exists
import yaml


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
