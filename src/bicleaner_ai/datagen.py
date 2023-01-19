from tensorflow.keras.preprocessing.sequence import pad_sequences
import sentencepiece as sp
import tensorflow as tf
import numpy as np
import logging

class SentenceEncoder(object):
    '''
    Wrapper of a SentencePiece model
    Ensure that all the encode calls us the same special tokens config
    '''

    def __init__(self, model_file, add_bos=False,
                 add_eos=False, enable_sampling=False):
        self.encoder = sp.SentencePieceProcessor(model_file=model_file)
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.enable_sampling = enable_sampling

    def encode(self, data, out_type=int):
        '''Wrapper function of the SentencePiece encode method'''
        return self.encoder.encode(data,
                        out_type=out_type,
                        add_bos=self.add_bos,
                        add_eos=self.add_eos,
                        enable_sampling=self.enable_sampling,
                        alpha=0.1)

class SentenceGenerator(tf.keras.utils.Sequence):
    '''
    Generates batches of sentences and its labels if they have
    Encoding procedure must be defined by subclasses
    '''

    def __init__(self, encoder,
            batch_size=32, maxlen=100, shuffle=False,
            separator=None):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.shuffle = shuffle
        self.num_samples = 0
        self.index = None
        self.text1 = None
        self.text2 = None
        self.weights = None
        self.y = None
        self.encoder = encoder
        self.separator = separator

    def __len__(self):
        '''
        Length of epochs
        '''
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        '''
        Return a batch of sentences
        '''
        # Avoid out of range when last batch smaller than batch_size
        if len(self)-1 == index:
            end = None
        else:
            end = (index+1)*self.batch_size
        start = index*self.batch_size
        indexes = self.index[start:end]

        x = self.encode_batch(
                    self.text1[indexes].tolist(),
                    self.text2[indexes].tolist())

        if self.weights is not None:
            w = self.weights[indexes]
            return x, self.y[indexes], w
        else:
            return x, self.y[indexes]

    def on_epoch_end(self):
        '''Shuffle indexes after each epoch'''
        if self.shuffle:
            np.random.shuffle(self.index)

    def encode_batch(self, text1, text2):
        raise NotImplementedError("Encoding must be defined by subclasses")

    def load(self, source):
        '''
        Load sentences and encode to index numbers
        If source is a string it is considered a file,
        if it is a list is considered:
            [text1_sentences, text2_sentences, tags, weights]
        Sample weights are optional
        '''

        # Read data from file if input is a filename
        if isinstance(source, str):
            data = [[], [], [], []]
            with open(source, 'r') as file_:
                for line in file_:
                    fields = line.split('\t')
                    data[0].append(fields[0])
                    data[1].append(fields[1])
                    data[2].append(fields[2].strip())
                    if len(fields) == 4:
                        data[3].append(fields[3].strip())
                    elif len(fields) > 4:
                        data[3].append([i.strip() for i in fields[3:]])
        else:
            data = source

        # Make a numpy array of sentences
        # to allow easy arbitrary indexing
        self.text1 = np.array(data[0], dtype=object)
        self.text2 = np.array(data[1], dtype=object)

        # Build array of sample weights
        if len(data) >= 4 and data[3]:
            if data[3][0].replace('.', '', 1).isdigit():
                logging.debug("Loading data weights")
                self.weights = np.array(data[3], dtype=float)
            else:
                logging.debug("Ignoring fourth column as it is not numeric")

        # Index samples
        self.num_samples = len(data[0])
        self.index = np.arange(0, self.num_samples)

        # Parse tags to array of integers
        if data[2] is None:
            self.y = np.zeros(self.num_samples)
        else:
            self.y = np.array(data[2], dtype=int)

        if self.shuffle:
            np.random.shuffle(self.index) # Preventive shuffle in case data comes ordered


class TupleSentenceGenerator(SentenceGenerator):
    '''
    Generates batches of tuples of sentences
    '''

    def encode_batch(self, text1, text2):
        # Vectorize sentences
        x1 = pad_sequences(self.encoder.encode(text1),
                           padding='post',
                           truncating='post',
                           maxlen=self.maxlen)
        x2 = pad_sequences(self.encoder.encode(text2),
                           padding='post',
                           truncating='post',
                           maxlen=self.maxlen)

        return x1, x2

class ConcatSentenceGenerator(SentenceGenerator):
    '''
    Generates batches of concatenated sentences
    '''

    def encode_batch(self, text1, text2):
        if isinstance(self.encoder, SentenceEncoder):
            # Concatenate sentences
            text = []
            for sent1, sent2 in zip(text1, text2):
                text.append(sent1 + self.separator + sent2)
            # Tokenize concatenated sentences with SentencePiece
            input_ids = pad_sequences(self.encoder.encode(text),
                                      padding="post",
                                      truncating="post",
                                      maxlen=self.maxlen)
            return input_ids
        else:
            # Tokenize with Transformers tokenizer that concatenates internally
            dataset = self.encoder(text1, text2,
                                   padding='longest',
                                   truncation=True,
                                   max_length=self.maxlen,
                                   return_tensors='np',
                                   return_attention_mask=True,
                                   return_token_type_ids=False)
            input_ids = dataset["input_ids"]
            att_mask = dataset["attention_mask"]

            return input_ids, att_mask
