from tensorflow.keras.preprocessing.sequence import pad_sequences
import sentencepiece as sp
import tensorflow as tf
import numpy as np

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


class TupleSentenceGenerator(tf.keras.utils.Sequence):
    '''
    Generates batches of tuples of sentences and its labels if they have
    '''

    def __init__(self, encoder: SentenceEncoder,
            batch_size=64, maxlen=50, shuffle=False):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.shuffle = shuffle
        self.num_samples = 0
        self.index = None
        self.x1 = None
        self.x2 = None
        self.y = None
        self.encoder = encoder


    def __len__(self):
        '''
        Length of epochs
        '''
        return int(np.ceil(self.x1.shape[0] / self.batch_size))

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

        if self.weights is not None:
            w = self.weights[indexes]
            return [self.x1[indexes], self.x2[indexes]], self.y[indexes], w
        else:
            return [self.x1[indexes], self.x2[indexes]], self.y[indexes]

    def on_epoch_end(self):
        'Shuffle indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.index)

    def load(self, source):
        '''
        Load sentences and encode to index numbers
        If source is a string it is considered a file,
        if it is a list is considered:
            [text1_sentences, text2_sentences, tags, weights]
        Sample weights are optional
        '''

        if isinstance(source, str):
            data = [[], [], [], []]
            with open(source, 'r') as file_:
                for line in file_:
                    fields = line.split('\t')
                    data[0].append(fields[0].strip())
                    data[1].append(fields[1].strip())
                    data[2].append(fields[2].strip())
                    if len(fields) == 4:
                        data[3].append(fields[3].strip())
                    elif len(fields) > 4:
                        data[3].append([i.strip() for i in fields[3:]])
        else:
            data = source

        # Vectorize input sentences
        self.x1 = pad_sequences(self.encoder.encode(data[0]),
                                padding='post',
                                truncating='post',
                                maxlen=self.maxlen)
        self.x2 = pad_sequences(self.encoder.encode(data[1]),
                                padding='post',
                                truncating='post',
                                maxlen=self.maxlen)
        self.num_samples = self.x1.shape[0]

        # Build array of labels
        if data[2] is None:
            # Set to 0's for prediction
            self.y = np.zeros(self.num_samples)
        else:
            self.y = np.array(data[2], dtype=int)

        # Build array of sample weights
        if len(data) >= 4 and data[3]:
            self.weights = np.array(data[3], dtype=float)
        else:
            self.weights = None

        # Build batch index
        self.index = np.arange(0, self.num_samples)

        if self.shuffle:
            # Preventive shuffle in case data comes ordered
            np.random.shuffle(self.index)


class ConcatSentenceGenerator(tf.keras.utils.Sequence):
    '''
    Generates batches of concatenated sentences and its labels if they have
    This generator is designed to be used with Transformers library
    '''

    def __init__(self, tokenizer,
            batch_size=64, maxlen=100, shuffle=False,
            separator=None):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.shuffle = shuffle
        self.num_samples = 0
        self.index = None
        self.x = None
        self.y = None
        self.tok = tokenizer
        self.separator = separator

    def __len__(self):
        '''
        Length of epochs
        '''
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    #TODO investigate how to return batches reading from stdin
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

        if self.att_mask is None:
            return self.x[indexes], self.y[indexes]
        else:
            return [self.x[indexes], self.att_mask[indexes]], self.y[indexes]

    def on_epoch_end(self):
        'Shuffle indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.index)

    def load(self, source):
        '''
        Load sentences and encode to index numbers
        If source is a string it is considered a file,
        if it is a list is considered [text1_sentences, text2_sentences, tags]
        '''

        if isinstance(source, str):
            data = [[], [], []]
            with open(source, 'r') as file_:
                for line in file_:
                    fields = line.split('\t')
                    # Concatenate sentences if tokenizer is SentencePiece
                    if isinstance(self.tok, SentenceEncoder):
                        data[0].append(fields[0] + self.separator + fields[1])
                        data[2].append(fields[2].strip())
                    else:
                        data[0].append(fields[0])
                        data[1].append(fields[1])
                        data[2].append(fields[2].strip())
        else:
            data = source

        if isinstance(self.tok, SentenceEncoder):
            # Tokenize already concatenated sentences with SentencePiece
            self.x = pad_sequences(self.tok.encode(data[0]),
                                    padding="post",
                                    truncating="post",
                                    maxlen=self.maxlen)
            self.att_mask = None
        else:
            # Tokenize with Transformers tokenizer that concatenates internally
            dataset = self.tok(data[0], data[1],
                               padding='max_length',
                               truncation=True,
                               max_length=self.maxlen,
                               return_tensors='np',
                               return_attention_mask=True,
                               return_token_type_ids=False)
            self.x = dataset["input_ids"]
            self.att_mask = dataset["attention_mask"]

        self.num_samples = self.x.shape[0]
        if data[2] is None:
            self.y = np.zeros(self.num_samples)
        else:
            self.y = np.array(data[2], dtype=int)
        self.index = np.arange(0, self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.index) # Preventive shuffle in case data comes ordered

