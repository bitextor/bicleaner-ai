from transformers import (
    TFXLMRobertaForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast
)
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput
from transformers.optimization_tf import create_optimizer
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from contextlib import redirect_stdout
from glove import Corpus, Glove
from abc import ABC, abstractmethod
import tensorflow.keras.backend as K
import sentencepiece as sp
import tensorflow as tf
import numpy as np
import logging
import h5py
import sys
import os

try:
    from . import decomposable_attention
    from .models_util import (
            load_attributes_from_hdf5_group,
            calibrate_output)
    from .metrics import FScore, MatthewsCorrCoef
    from .datagen import (
            TupleSentenceGenerator,
            ConcatSentenceGenerator,
            SentenceEncoder)
    from .layers import (
            TransformerBlock,
            TokenAndPositionEmbedding,
            BicleanerAIClassificationHead)
except (SystemError, ImportError):
    import decomposable_attention
    from models_util import (
            load_attributes_from_hdf5_group,
            calibrate_output)
    from metrics import FScore, MatthewsCorrCoef
    from datagen import (
            TupleSentenceGenerator,
            ConcatSentenceGenerator,
            SentenceEncoder)
    from layers import (
            TransformerBlock,
            TokenAndPositionEmbedding,
            BicleanerAIClassificationHead)


class ModelInterface(ABC):
    '''
    Interface for model classes that gathers the essential
    model methods: init, load, predict and train
    '''
    @abstractmethod
    def __init__(self, directory, settings):
        pass

    @abstractmethod
    def get_generator(self, batch_size, shuffle):
        pass

    @abstractmethod
    def predict(self, x1, x2, batch_size=None, calibrated=False,
                raw=False, verbose=0):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def train(self, train_set, dev_set):
        pass

    def calibrate(self, y_pred):
        A = self.settings["calibration_params"][0]
        B = self.settings["calibration_params"][1]
        return 1/(1 + np.exp(-(A*y_pred+B)))

class BaseModel(ModelInterface):
    '''Abstract Model class that gathers most of the training logic'''

    def __init__(self, directory, settings, distilled=False):
        self.dir = directory
        self.trained = False
        self.spm = None
        self.vocab = None
        self.model = None
        self.wv = None
        self.spm_prefix = 'spm'

        # Override with user defined settings in derived classes, not here
        self.settings = {
            "spm_file": self.spm_prefix + ".model",
            "vocab_file": self.spm_prefix + ".vocab",
            "model_file": "model.h5",
            "wv_file": "glove.vectors",
            "separator": '',
            "bos_id": -1,
            "eos_id": -1,
            "pad_id": 0,
            "unk_id": 1,
            "add_bos": False,
            "add_eos": False,
            "sampling": False,
            "emb_dim": 300,
            "emb_trainable": True,
            "emb_epochs": 10,
            "window": 15,
            "vocab_size": 32000,
            "batch_size": 1024,
            "maxlen": 100,
            "n_hidden": 200,
            "dropout": 0.2,
            "distilled": distilled,
            "n_classes": 2 if distilled else 1,
            "entail_dir": "both",
            "epochs": 200,
            "steps_per_epoch": 4096,
            "patience": 20,
            "loss": "categorical_crossentropy" if distilled else "binary_crossentropy",
            "lr": 5e-4,
            "clipnorm": None,
            "metrics": self.get_metrics,
        }
        scheduler = InverseTimeDecay(self.settings["lr"],
                         decay_steps=self.settings["steps_per_epoch"]//4,
                         decay_rate=0.2)
        self.settings["scheduler"] = scheduler
        self.settings["optimizer"] = Adam(learning_rate=scheduler,
                                          clipnorm=self.settings["clipnorm"])

    def get_metrics(self):
        '''
        Class method to create metric objects.
        Variables need to be instatiated inside the same
        strategy scope that the model.
        '''
        return [
            #TODO create argmax precision and recall or use categorical acc
            #Precision(name='p'),
            #Recall(name='r'),
            FScore(argmax=self.settings["distilled"]),
            MatthewsCorrCoef(argmax=self.settings["distilled"]),
        ]

    def get_generator(self, batch_size, shuffle):
        ''' Returns a sentence generator instance according to the model input '''
        raise NotImplementedError("Subclass must define its sentence generator")

    def build_model(self, compile=True):
        '''Returns a compiled Keras model instance'''
        raise NotImplementedError("Subclass must implement its model architecture")

    def predict(self, x1, x2, batch_size=None, calibrated=False,
                raw=False, verbose=0):
        '''Predicts from sequence generator'''
        if batch_size is None:
            batch_size = self.settings["batch_size"]
        generator = self.get_generator(batch_size, shuffle=False)
        generator.load((x1, x2, None))

        with redirect_stdout(sys.stderr):
            y_pred = self.model.predict(generator, verbose=verbose)
        # Obtain logits if model returns HF output
        if isinstance(y_pred, TFSequenceClassifierOutput):
            y_pred = y_pred.logits

        if raw:
            return y_pred

        if self.settings["n_classes"] == 1:
            y_pred_probs = y_pred
        else:
            # Compute softmax probability if output is 2-class
            y_pred_probs = self.softmax_pos_prob(y_pred)

        if calibrated and "calibration_params" in self.settings:
            return self.calibrate(y_pred_probs)
        else:
            return y_pred_probs


    def load_spm(self):
        '''Loads SentencePiece model and vocabulary from model directory'''
        self.spm = SentenceEncoder(self.dir+'/'+self.settings["spm_file"],
                                   add_bos=self.settings["add_bos"],
                                   add_eos=self.settings["add_eos"],
                                   enable_sampling=self.settings["sampling"])
        self.vocab = {}
        with open(self.dir + '/' + self.settings["vocab_file"]) as vocab_file:
            for i, line in enumerate(vocab_file):
                token = line.split('\t')[0]
                self.vocab[token] = i
        logging.info("Loaded SentencePiece model")

    def load_embed(self):
        '''Loads embeddings from model directory'''
        glove = Glove().load(self.dir+'/'+self.settings["wv_file"])
        self.wv = glove.word_vectors
        logging.info("Loaded SentenePiece Glove vectors")

    def load(self):
        '''Loads the whole model'''
        self.load_spm()
        logging.info("Loading neural classifier")
        deps = {'FScore': FScore,
                'MatthewsCorrCoef': MatthewsCorrCoef,
                'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                'K': tf.keras.backend,
        }

        # Try loading the whole model
        # If it fails due to bad marshal (saved with different Python version)
        # build a new model and load weights
        try:
            self.model = load_model(self.dir+'/'+self.settings["model_file"],
                                    custom_objects=deps, compile=False)
        except ValueError:
            self.model = self.build_model(compile=False)
            self.model.load_weights(self.dir+'/'+self.settings["model_file"])

    def train_vocab(self, monolingual, threads):
        '''Trains SentencePiece model and embeddings with Glove'''

        logging.info("Training SentencePiece joint vocabulary")
        trainer = sp.SentencePieceTrainer
        trainer.train(sentence_iterator=monolingual,
                      model_prefix=self.dir+'/'+self.spm_prefix,
                      vocab_size=self.settings["vocab_size"],
                      input_sentence_size=5000000,
                      shuffle_input_sentence=True,
                      pad_id=self.settings["pad_id"],
                      unk_id=self.settings["unk_id"],
                      bos_id=self.settings["bos_id"],
                      eos_id=self.settings["eos_id"],
                      user_defined_symbols=self.settings["separator"],
                      num_threads=threads,
                      minloglevel=1)
        monolingual.seek(0)
        self.load_spm()

        logging.info("Computing co-occurence matrix")
        # Iterator function that reads and tokenizes file
        # to avoid reading the whole input into memory
        def get_data(input_file):
            for line in input_file:
                yield self.spm.encode(line.rstrip(), out_type=str)
        corpus = Corpus(self.vocab) # Use spm vocab as glove vocab
        corpus.fit(get_data(monolingual), window=self.settings["window"],
                   ignore_missing=True)

        logging.info("Training vocabulary embeddings")
        embeddings = Glove(no_components=self.settings["emb_dim"])
        embeddings.fit(corpus.matrix,
                       epochs=self.settings["emb_epochs"],
                       no_threads=threads)
        self.wv = embeddings.word_vectors
        embeddings.save(self.dir + '/' + self.settings["wv_file"])

    def train(self, train_set, dev_set):
        '''Trains the neural classifier'''

        if self.wv is None or self.spm is None:
            raise Exception("Vocabulary is not trained")
        settings = self.settings

        logging.info("Loading training set")
        train_generator = self.get_generator(
                                settings["batch_size"],
                                shuffle=True)
        train_generator.load(train_set)
        steps_per_epoch = min(len(train_generator),
                              settings["steps_per_epoch"])

        dev_generator = self.get_generator(
                                settings["batch_size"],
                                shuffle=False)
        dev_generator.load(dev_set)

        model_filename = self.dir + '/' + settings["model_file"]
        earlystop = EarlyStopping(monitor='val_f1',
                                  mode='max',
                                  patience=settings["patience"],
                                  restore_best_weights=True)
        class LRReport(Callback):
            def on_epoch_end(self, epoch, logs={}):
                # Compatibility of TF2.11 with previous versions
                if isinstance(self.model.optimizer.lr,
                              tf.keras.optimizers.schedules.LearningRateSchedule):
                    print(f' - lr: {self.model.optimizer.lr(epoch*steps_per_epoch):.3E}')
                else:
                    print(f' - lr: {float(self.model.optimizer.lr):.3E}')

        logging.info("Training neural classifier")

        strategy = tf.distribute.MirroredStrategy()
        num_devices = strategy.num_replicas_in_sync
        with strategy.scope():
            self.model = self.build_model()
        if logging.getLogger().level == logging.DEBUG:
            self.model.summary()
        if logging.getLogger().level <= logging.INFO:
            verbose = 1
        else:
            verbose = 0

        with redirect_stdout(sys.stderr):
            self.model.fit(train_generator,
                           batch_size=settings["batch_size"],
                           epochs=settings["epochs"],
                           steps_per_epoch=steps_per_epoch,
                           validation_data=dev_generator,
                           callbacks=[earlystop, LRReport()],
                           verbose=verbose)
        self.model.save(model_filename)

        y_true = dev_generator.y
        with redirect_stdout(sys.stderr):
            y_pred_probs = self.model.predict(dev_generator, verbose=verbose)
        y_pred = np.where(y_pred_probs >= 0.5, 1, 0)
        logging.info(f"Dev precision: {precision_score(y_true, y_pred):.3f}")
        logging.info(f"Dev recall: {recall_score(y_true, y_pred):.3f}")
        logging.info(f"Dev f1: {f1_score(y_true, y_pred):.3f}")
        logging.info(f"Dev mcc: {matthews_corrcoef(y_true, y_pred):.3f}")

        A, B = calibrate_output(y_true, y_pred_probs)
        self.settings["calibration_params"] = (A, B)

        return y_true, y_pred

class DecomposableAttention(BaseModel):
    '''Decomposable Attention model (Parikh et. al. 2016)'''

    def __init__(self, directory, settings, **kwargs):
        super(DecomposableAttention, self).__init__(directory, settings, **kwargs)

        self.settings = {
            **self.settings, # Obtain settings from parent
            "self_attention": False,
            **settings, # Override default settings with user-defined
        }

    def get_generator(self, batch_size, shuffle):
        return TupleSentenceGenerator(
                    self.spm, shuffle=shuffle,
                    batch_size=batch_size,
                    maxlen=self.settings["maxlen"])

    def build_model(self, compile=True):
        return decomposable_attention.build_model(self.wv, self.settings, compile)

class Transformer(BaseModel):
    '''Basic Transformer model'''

    def __init__(self, directory, settings, **kwargs):
        super(Transformer, self).__init__(directory, settings, **kwargs)

        self.settings = {
            **self.settings, # Obtain settings from parent
            "separator": '[SEP]',
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "add_bos": True,
            "add_eos": True,
            "maxlen": 200,
            "n_hidden": 200,
            "n_heads": 4,
            "dropout": 0.2,
            "att_dropout": 0.5,
            "lr": 5e-4,
            "clipnorm": 1.0,
            **settings, # Override default settings with user-defined
        }
        scheduler = InverseTimeDecay(self.settings["lr"],
                         decay_steps=self.settings["steps_per_epoch"]//4,
                         decay_rate=0.2)
        self.settings["scheduler"] = scheduler
        self.settings["optimizer"] = Adam(learning_rate=self.settings["scheduler"],
                                          clipnorm=self.settings["clipnorm"])

    def get_generator(self, batch_size, shuffle):
        return ConcatSentenceGenerator(
                    self.spm, shuffle=shuffle,
                    batch_size=batch_size,
                    maxlen=self.settings["maxlen"],
                    separator=self.settings["separator"])

    def build_model(self, compile=True):
        settings = self.settings
        inputs = layers.Input(shape=(settings["maxlen"],), dtype='int32')
        embedding = TokenAndPositionEmbedding(settings['vocab_size'],
                                              settings['emb_dim'],
                                              settings["maxlen"],
                                              self.wv,
                                              trainable=True)
        transformer_block = TransformerBlock(
                                settings["emb_dim"],
                                settings["n_heads"],
                                settings["n_hidden"],
                                settings["att_dropout"])

        x = embedding(inputs)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(settings["dropout"])(x)
        x = layers.Dense(settings["n_hidden"], activation="relu")(x)
        x = layers.Dropout(settings["dropout"])(x)
        if settings['loss'] == 'categorical_crossentropy':
            outputs = layers.Dense(settings["n_classes"], activation='softmax')(x)
        else:
            outputs = layers.Dense(settings["n_classes"], activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        if compile:
            model.compile(optimizer=settings["optimizer"],
                          loss=settings["loss"],
                          metrics=settings["metrics"]())
        return model


class BCXLMRoberta(BaseModel):
    ''' Fine-tuned XLMRoberta model '''

    def __init__(self, directory, settings, **kwargs):
        self.dir = directory
        self.model = None
        self.tokenizer = None

        self.settings = {
            "base_model": 'jplu/tf-xlm-roberta-base',
            "batch_size": 16,
            "maxlen": 150,
            "n_classes": 2,
            "epochs": 10,
            "steps_per_epoch": 40000,
            "patience": 3,
            "dropout": 0.1,
            "n_hidden": 2048,
            "activation": 'relu',
            "loss": "binary_crossentropy",
            "lr": 2e-6,
            "decay_rate": 0.1,
            "warmup_steps": 1000,
            "clipnorm": 1.0,
            **settings,
        }
        scheduler = InverseTimeDecay(self.settings["lr"],
                                     decay_steps=32.0,
                                     decay_rate=0.1)
        self.settings["scheduler"] = scheduler
        optimizer, scheduler = create_optimizer(
                self.settings["lr"],
                self.settings["steps_per_epoch"]*self.settings["epochs"],
                self.settings["warmup_steps"],
                weight_decay_rate=self.settings["decay_rate"])
        self.settings["scheduler"] = scheduler
        self.settings["optimizer"] = optimizer

    def get_generator(self, batch_size, shuffle):
        return ConcatSentenceGenerator(
                self.tokenizer, shuffle=shuffle,
                batch_size=batch_size,
                maxlen=self.settings["maxlen"])

    def load_model(self, model_file):
        settings = self.settings

        tf_model = TFXLMRBicleanerAI.from_pretrained(
                        model_file,
                        num_labels=settings["n_classes"],
                        head_hidden_size=settings["n_hidden"],
                        head_dropout=settings["dropout"],
                        head_activation=settings["activation"])

        return tf_model

    def load(self):
        ''' Load fine-tuned model '''
        # If vocab and model files are in a subdirectory, load from there
        if "vocab_file" in self.settings:
            vocab_file = f'{self.dir}/{self.settings["vocab_file"]}'
        else:
            vocab_file = self.dir

        if "model_file" in self.settings:
            model_file = f'{self.dir}/{self.settings["model_file"]}'
        else:
            model_file = self.dir

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(vocab_file)
        self.model = self.load_model(model_file)

    def softmax_pos_prob(self, x):
        # Compute softmax probability of the second (positive) class
        e_x = np.exp(x - np.max(x))
        # Need transpose to compute for each sample in the batch
        # then slice to return class probability
        return (e_x.T / (np.sum(e_x, axis=1).T)).T[:,1:]

    def build_dataset(self, filename):
        ''' Read a file into a TFDataset '''
        data = [[], [], []]
        with open(filename, 'r') as file_:
            for line in file_:
                fields = line.split('\t')
                data[0].append(fields[0].strip())
                data[1].append(fields[1].strip())
                data[2].append(int(fields[2].strip()))
        # Give the sentences in separate arguments
        # so the tokenizer adds the corresponding special tokens
        sentences = self.tokenizer(data[0], data[1],
                                      padding=True,
                                      truncation=True,
                                      max_length=self.settings["maxlen"])

        ds = tf.data.Dataset.from_tensor_slices((dict(sentences),
                                                 data[2]))
        ds = ds.shuffle(len(sentences))
        return ds.batch(self.settings["batch_size"]).prefetch(5)

    def train_vocab(self, **kwargs):
        pass

    def train(self, train_set, dev_set):
        logging.info("Loading training set")

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
                                                    self.settings["base_model"])
        train_generator = self.get_generator(self.settings["batch_size"],
                                             shuffle=True)
        train_generator.load(train_set)
        steps_per_epoch = min(len(train_generator),
                              self.settings["steps_per_epoch"])

        dev_generator = self.get_generator(self.settings["batch_size"],
                                             shuffle=False)
        dev_generator.load(dev_set)

        earlystop = EarlyStopping(monitor='val_f1',
                                  mode='max',
                                  patience=self.settings["patience"],
                                  restore_best_weights=True)

        logging.info("Training classifier")

        strategy = tf.distribute.MirroredStrategy()
        num_devices = strategy.num_replicas_in_sync
        with strategy.scope():
            self.model = self.load_model(self.settings["base_model"])
            self.model.compile(optimizer=self.settings["optimizer"],
                               loss=SparseCategoricalCrossentropy(
                                        from_logits=True),
                               metrics=[FScore(argmax=True),
                                        MatthewsCorrCoef(argmax=True)])
        self.model.config._name_or_path = self.settings["model_name"]

        if logging.getLogger().level == logging.DEBUG:
            self.model.summary()
        if logging.getLogger().level <= logging.INFO:
            verbose = 1
        else:
            verbose = 0

        with redirect_stdout(sys.stderr):
            self.model.fit(train_generator,
                           epochs=self.settings["epochs"],
                           steps_per_epoch=steps_per_epoch,
                           validation_data=dev_generator,
                           batch_size=self.settings["batch_size"],
                           callbacks=[earlystop],
                           verbose=verbose)
        self.model.save_pretrained(self.dir)
        self.tokenizer.save_pretrained(self.dir)

        y_true = dev_generator.y
        with redirect_stdout(sys.stderr):
            y_pred = self.model.predict(dev_generator, verbose=verbose).logits
        y_pred_probs = self.softmax_pos_prob(y_pred)
        y_pred = np.argmax(y_pred, axis=-1)
        logging.info(f"Dev precision: {precision_score(y_true, y_pred):.3f}")
        logging.info(f"Dev recall: {recall_score(y_true, y_pred):.3f}")
        logging.info(f"Dev f1: {f1_score(y_true, y_pred):.3f}")
        logging.info(f"Dev mcc: {matthews_corrcoef(y_true, y_pred):.3f}")

        A, B = calibrate_output(y_true, y_pred_probs)
        self.settings["calibration_params"] = (A, B)

        return y_true, y_pred

class XLMRBicleanerAIConfig(XLMRobertaConfig):
    '''
    Bicleaner AI XLMR configuration class
    adds the config parameters for the classification layer
    '''
    def __init__(self, head_hidden_size=2048, head_dropout=0.1,
                 head_activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.head_hidden_size = head_hidden_size
        self.head_dropout = head_dropout
        self.head_activation = head_activation

class TFXLMRBicleanerAI(TFXLMRobertaForSequenceClassification):
    '''
    Model for Bicleaner sentence-level classification tasks.
    Overwrites XLMRoberta classifiacion layer.
    '''
    config_class = XLMRBicleanerAIConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # Set the appropiate name for the classifier head layer
        # to avoid old models not being loaded correctly
        name = 'bicleaner_ai_classification_head'
        # If it's not a path, we are loading from hub, use the new name
        if os.path.isdir(config._name_or_path):
            # Inspect model file to guess if it has old head name
            with h5py.File(config._name_or_path + '/tf_model.h5', 'r') as h5:
                layers = set(load_attributes_from_hdf5_group(h5, "layer_names"))
                if 'bc_classification_head' in layers:
                    name = 'bc_classification_head'

        self.classifier = BicleanerAIClassificationHead(config,
                            name=name)

