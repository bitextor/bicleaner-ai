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

        assert type(yml['calibration_params']) == list
        assert yml['steps_per_epoch'] == steps
        assert yml['epochs'] == epochs
        assert yml['batch_size'] == batch_size
        assert yml['classifier_type'] == classifier_type
