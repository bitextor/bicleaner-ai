from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from glove import Corpus, Glove
from abc import ABC
import tensorflow.keras.backend as K
import sentencepiece as sp
import tensorflow as tf
import numpy as np
import decomposable_attention
import logging

try:
    from .metrics import FScore
    from .datagen import (
            TupleSentenceGenerator,
            ConcatSentenceGenerator,
            SentenceEncoder)
    from .layers import (
            TransformerBlock,
            TokenAndPositionEmbedding)
except (SystemError, ImportError):
    from metrics import FScore
    from datagen import (
            TupleSentenceGenerator,
            ConcatSentenceGenerator,
            SentenceEncoder)
    from layers import (
            TransformerBlock,
            TokenAndPositionEmbedding)

class BaseModel(ABC):
    '''Abstract Model class that gathers most of the training logic'''

    def __init__(self, directory):
        self.dir = directory
        self.trained = False
        self.spm = None
        self.vocab = None
        self.model = None
        self.wv = None

        self.settings = {
            "separator": None,
            "bos_id": -1,
            "eos_id": -1,
            "pad_id": 0,
            "unk_id": 1,
            "add_bos": False,
            "add_eos": False,
            "enable_sampling": False,
            "emb_dim": 300,
            "emb_trainable": False,
            "emb_epochs": 10,
            "window": 15,
            "vocab_size": 32000,
            "batch_size": 1024,
            "maxlen": 100,
            "n_hidden": 200,
            "dropout": 0.2,
            "n_classes": 1,
            "entail_dir": "both",
            "epochs": 200,
            "steps_per_epoch": 4096,
            "patience": 20,
            "loss": "binary_crossentropy",
            "lr": 5e-4,
            "clipnorm": 0.5,
        }
        scheduler = InverseTimeDecay(self.settings["lr"],
                         decay_steps=self.settings["steps_per_epoch"]//4,
                         decay_rate=0.2)
        # scheduler = tf.keras.experimental.CosineDecayRestarts(
        #         self.settings["lr"],
        #         self.settings["steps_per_epoch"]*4,
        #         t_mul=2.0, m_mul=0.8)
        self.settings["scheduler"] = scheduler

    def get_generator(self, batch_size, shuffle):
        ''' Returns a sentence generator instance according to the model input '''
        raise NotImplementedError("Subclass must define its sentence generator")

    def build_model(self):
        '''Returns a compiled Keras model instance'''
        raise NotImplementedError("Subclass must implement its model architecture")

    def predict(self, x1, x2, batch_size=None):
        '''Predicts from sequence generator'''
        if batch_size is None:
            batch_size = self.settings["batch_size"]
        generator = self.get_generator(batch_size, shuffle=False)
        generator.load((x1, x2, None))
        return self.model.predict(generator)

    def load_spm(self):
        '''Loads SentencePiece model and vocabulary from model directory'''
        logging.info("Loading SentencePiece model")
        self.spm = SentenceEncoder(self.dir+'/spm.model',
                                   add_bos=self.settings["add_bos"],
                                   add_eos=self.settings["add_eos"],
                                   enable_sampling=self.settings["enable_sampling"])
        self.vocab = {}
        with open(self.dir + '/spm.vocab') as vocab_file:
            for i, line in enumerate(vocab_file):
                token = line.split('\t')[0]
                self.vocab[token] = i

    def load_embed(self):
        '''Loads embeddings from model directory'''
        logging.info("Loading SentenePiece Glove vectors")
        self.wv = Glove().load(self.dir + '/glove.vectors').word_vectors

    def load(self):
        '''Loads the whole model'''
        self.load_spm()
        logging.info("Loading neural classifier")
        deps = { 'FScore': FScore }
        self.model = load_model(self.dir + '/model.h5', custom_objects=deps)

    def train_vocab(self, monolingual, threads):
        '''Trains SentencePiece model and embeddings with Glove'''

        logging.info("Training SentencePiece joint vocabulary")
        trainer = sp.SentencePieceTrainer
        trainer.train(sentence_iterator=monolingual,
                      model_prefix=self.dir+'/spm',
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
        embeddings.save(self.dir + '/glove.vectors')

    def train(self, train_set, dev_set):
        '''Trains the neural classifier'''

        if self.wv is None or self.spm is None:
            raise Exception("Vocabulary is not trained")

        logging.info("Vectorizing training set")
        train_generator = self.get_generator(
                                self.settings["batch_size"],
                                shuffle=True)
        train_generator.load(train_set)
        steps_per_epoch = min(len(train_generator),
                              self.settings["steps_per_epoch"])

        dev_generator = self.get_generator(
                                self.settings["batch_size"],
                                shuffle=False)
        dev_generator.load(dev_set)

        model_filename = self.dir + '/model.h5'
        earlystop = EarlyStopping(monitor='val_f1',
                                  mode='max',
                                  patience=self.settings["patience"],
                                  restore_best_weights=True)
        class LRReport(Callback):
            def on_epoch_end(self, epoch, logs={}):
                print(f' - lr: {self.model.optimizer.lr(epoch*steps_per_epoch):.3E}')

        logging.info("Training neural classifier")

        self.model = self.build_model()
        self.model.summary()
        self.model.fit(train_generator,
                       batch_size=self.settings["batch_size"],
                       epochs=self.settings["epochs"],
                       steps_per_epoch=steps_per_epoch,
                       validation_data=dev_generator,
                       callbacks=[earlystop, LRReport()],
                       verbose=1)
        self.model.save(model_filename)

        y_true = dev_generator.y
        y_pred = np.where(self.model.predict(dev_generator) >= 0.5, 1, 0)
        logging.info(f"Dev precision: {precision_score(y_true, y_pred):.3f}")
        logging.info(f"Dev recall: {recall_score(y_true, y_pred):.3f}")
        logging.info(f"Dev f1: {f1_score(y_true, y_pred):.3f}")
        logging.info(f"Dev mcc: {matthews_corrcoef(y_true, y_pred):.3f}")

        return y_true, y_pred

class DecomposableAttention(BaseModel):
    '''Decomposable Attention model (Parikh et. al. 2016)'''

    def __init__(self, directory):
        super(DecomposableAttention, self).__init__(directory)

        self.settings = {
            **self.settings,
            "self_attention": True,
        }

    def get_generator(self, batch_size, shuffle):
        return TupleSentenceGenerator(
                    self.spm, shuffle=shuffle,
                    batch_size=batch_size,
                    maxlen=self.settings["maxlen"])

    def build_model(self):
        return decomposable_attention.build_model(self.wv, self.settings)

class Transformer(BaseModel):
    '''Basic Transformer model'''

    def __init__(self, directory):
        super(Transformer, self).__init__(directory)

        self.separator = '[SEP]'
        self.settings = {
            **self.settings,
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
            "batch_size": 1024,
            "lr": 5e-4,
            "clipnorm": 1.0,
        }
        scheduler = InverseTimeDecay(self.settings["lr"],
                         decay_steps=self.settings["steps_per_epoch"]//4,
                         decay_rate=0.2)
        self.settings["scheduler"] = scheduler

    def get_generator(self, batch_size, shuffle):
        return ConcatSentenceGenerator(
                    self.spm, shuffle=shuffle,
                    batch_size=batch_size,
                    maxlen=self.settings["maxlen"],
                    separator=self.separator)

    def build_model(self):
        settings = self.settings
        inputs = layers.Input(shape=(settings["maxlen"],), dtype='int32')
        embedding = TokenAndPositionEmbedding(self.wv,
                                              settings["maxlen"],
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
        model.compile(
                optimizer=Adam(learning_rate=settings["scheduler"],
                               clipnorm=settings["clipnorm"]),
                loss=settings["loss"],
                metrics=[Precision(name='p'), Recall(name='r'), FScore(name='f1')])
        return model

