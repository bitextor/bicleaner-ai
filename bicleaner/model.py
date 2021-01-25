from keras.optimizers.schedules import InverseTimeDecay
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from keras.models import load_model
from glove import Corpus, Glove
import sentencepiece as sp
import tensorflow as tf
import numpy as np
import logging

try:
    from .decomposable_attention import build_model
    from .datagen import TupleSentenceGenerator
except (SystemError, ImportError):
    from decomposable_attention import build_model
    from datagen import TupleSentenceGenerator

class Model(object):

    def __init__(self, directory):
        self.dir = directory
        self.trained = False
        self.spm = None
        self.vocab = None
        self.model = None
        self.wv = None

        self.settings = {
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
            "epochs": 100,
            "steps_per_epoch": 128,
            "patience": 20,
            "loss": "binary_crossentropy",
            "lr": 0.005,
            "clipnorm": 0.5,
        }
        scheduler = InverseTimeDecay(self.settings["lr"],
                                     decay_steps=32.0,
                                     decay_rate=0.1)
        self.settings["scheduler"] = scheduler

    def predict(self, x1, x2, batch_size=None):
        '''Predicts from sequence generator'''
        if batch_size is None:
            batch_size = self.settings["batch_size"]
        generator = TupleSentenceGenerator(
                        self.spm, shuffle=False,
                        batch_size=batch_size,
                        maxlen=self.settings["maxlen"])
        generator.load((x1, x2, None))
        return self.model.predict(generator)

    def load_spm(self):
        '''Loads SentencePiece model and vocabulary from model directory'''
        logging.info("Loading SentencePiece model")
        self.spm = sp.SentencePieceProcessor(model_file=self.dir+'/spm.model')
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
        self.model = load_model(self.dir + '/model.h5')

    def train_vocab(self, monolingual, threads):
        '''Trains SentencePiece model and embeddings with Glove'''

        logging.info("Training SentencePiece joint vocabulary")
        trainer = sp.SentencePieceTrainer
        trainer.train(sentence_iterator=monolingual,
                      model_prefix=self.dir+'/spm',
                      vocab_size=self.settings["vocab_size"],
                      input_sentence_size=5000000,
                      shuffle_input_sentence=True,
                      pad_id=0,
                      unk_id=1,
                      bos_id=-1,
                      eos_id=-1,
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
        if self.wv is None or self.spm is None:
            raise Exception("Vocabulary is not trained")

        logging.info("Vectorizing training set")
        train_generator = TupleSentenceGenerator(
                              self.spm, shuffle=True,
                              batch_size=self.settings["batch_size"],
                              maxlen=self.settings["maxlen"])
        train_generator.load(train_set)
        steps_per_epoch = min(len(train_generator),
                              self.settings["steps_per_epoch"])

        dev_generator = TupleSentenceGenerator(
                            self.spm, shuffle=False,
                            batch_size=self.settings["batch_size"],
                            maxlen=self.settings["maxlen"])
        dev_generator.load(dev_set)

        model_filename = self.dir + '/model.h5'
        earlystop = EarlyStopping(monitor='val_f1',
                                  mode='max',
                                  patience=self.settings["patience"],
                                  restore_best_weights=True)

        logging.info("Training neural classifier")

        self.model = build_model(self.wv, self.settings)
        self.model.summary()
        self.model.fit(train_generator,
                       batch_size=self.settings["batch_size"],
                       epochs=self.settings["epochs"],
                       steps_per_epoch=steps_per_epoch,
                       validation_data=dev_generator,
                       callbacks=[earlystop],
                       verbose=1)
        self.model.save(model_filename)

        y_true = dev_generator.y
        y_pred = np.where(self.model.predict(dev_generator) >= 0.5, 1, 0)
        logging.info("Dev precision: {:.3f}".format(precision_score(y_true, y_pred)))
        logging.info("Dev recall: {:.3f}".format(precision_score(y_true, y_pred)))
        logging.info("Dev f1: {:.3f}".format(f1_score(y_true, y_pred)))
        logging.info("Dev mcc: {:.3f}".format(matthews_corrcoef(y_true, y_pred)))

        return y_true, y_pred
