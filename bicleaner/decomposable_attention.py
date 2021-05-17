'''
Keras implementation of Decomposable Attention (https://arxiv.org/pdf/1606.01933v1.pdf)
taken from spaCy examples (https://github.com/explosion/spaCy/tree/master/examples)
with modifications.

The MIT License (MIT)
Copyright (C) 2016-2020 ExplosionAI GmbH, 2016 spaCy GmbH, 2015 Matthew Honnibal
'''

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import layers, Model, models
from tensorflow.keras import backend as K
import numpy as np

try:
    from .losses import KDLoss
    from .metrics import MatthewsCorrCoef
    from .layers import TokenAndPositionEmbedding
except (SystemError, ImportError):
    from losses import KDLoss
    from metrics import MatthewsCorrCoef
    from layers import TokenAndPositionEmbedding

def build_model(vectors, settings):
    max_length = settings["maxlen"]
    nr_hidden = settings["n_hidden"]
    nr_class = settings["n_classes"]

    input1 = layers.Input(shape=(max_length,), dtype="int32", name="words1")
    input2 = layers.Input(shape=(max_length,), dtype="int32", name="words2")

    # embeddings (projected)
    embed = create_embedding(vectors, max_length, nr_hidden, settings["emb_trainable"])

    a = embed(input1)
    b = embed(input2)

    # step 1: attend
    # self-attend
    if settings["self_attention"]:
        S_a = create_feedforward(nr_hidden, dropout=settings["dropout"])
        S_b = create_feedforward(nr_hidden, dropout=settings["dropout"])
        a_p = layers.Attention()([S_a(a), S_a(a)])
        b_p = layers.Attention()([S_b(b), S_b(b)])
        # self_att_a = layers.dot([S_a(a), S_a(a)], axes=-1)
        # self_att_b = layers.dot([S_b(b), S_b(b)], axes=-1)
        # self_norm_a = layers.Lambda(normalizer(1))(self_att_a)
        # self_norm_b = layers.Lambda(normalizer(1))(self_att_b)
        # a_p = layers.dot([self_norm_a, a], axes=1)
        # b_p = layers.dot([self_norm_b, b], axes=1)
    else:
        a_p = a
        b_p = b

    # attend
    F = create_feedforward(nr_hidden, dropout=settings["dropout"])
    att_weights = layers.dot([F(a_p), F(b_p)], axes=-1)

    G = create_feedforward(nr_hidden)

    if settings["entail_dir"] == "both":
        norm_weights_a = layers.Lambda(normalizer(1))(att_weights)
        norm_weights_b = layers.Lambda(normalizer(2))(att_weights)
        alpha = layers.dot([norm_weights_a, a_p], axes=1)
        beta = layers.dot([norm_weights_b, b_p], axes=1)

        # step 2: compare
        comp1 = layers.concatenate([a_p, beta])
        comp2 = layers.concatenate([b_p, alpha])
        v1 = layers.TimeDistributed(G)(comp1)
        v2 = layers.TimeDistributed(G)(comp2)

        # step 3: aggregate
        v1_sum = layers.Lambda(sum_word)(v1)
        v2_sum = layers.Lambda(sum_word)(v2)
        concat = layers.concatenate([v1_sum, v2_sum])

    elif settings["entail_dir"] == "left":
        norm_weights_a = layers.Lambda(normalizer(1))(att_weights)
        alpha = layers.dot([norm_weights_a, a], axes=1)
        comp2 = layers.concatenate([b, alpha])
        v2 = layers.TimeDistributed(G)(comp2)
        v2_sum = layers.Lambda(sum_word)(v2)
        concat = v2_sum

    else:
        norm_weights_b = layers.Lambda(normalizer(2))(att_weights)
        beta = layers.dot([norm_weights_b, b], axes=1)
        comp1 = layers.concatenate([a, beta])
        v1 = layers.TimeDistributed(G)(comp1)
        v1_sum = layers.Lambda(sum_word)(v1)
        concat = v1_sum

    H = create_feedforward(nr_hidden, dropout=settings["dropout"])
    out = H(concat)
    if settings['distilled']:
        out = layers.Dense(nr_class)(out)
        loss = KDLoss(settings["batch_size"])
    else:
        out = layers.Dense(nr_class)(out)
        out = layers.Activation('sigmoid', dtype='float32')(out)
        loss = settings["loss"]

    model = Model([input1, input2], out)

    model.compile(optimizer=settings["optimizer"],
                  loss=loss,
                  metrics=settings["metrics"](), # Call get_metrics
                  experimental_run_tf_function=False,)

    return model


def create_embedding(vectors, max_length, projected_dim, trainable=False):
    return models.Sequential(
        [
            # layers.Embedding(
            #     vectors.shape[0],
            #     vectors.shape[1],
            #     input_length=max_length,
            #     weights=[vectors],
            #     trainable=trainable,
            #     mask_zero=True,
            # ),
            TokenAndPositionEmbedding(vectors.shape[0],
                                      vectors.shape[1],
                                      max_length,
                                      vectors,
                                      trainable),
            layers.TimeDistributed(
                layers.Dense(projected_dim, activation=None, use_bias=False,
                    kernel_regularizer=None)
            ),
        ]
    )


def create_feedforward(num_units=200, activation="relu", dropout=0.2):
    return models.Sequential(
        [
            layers.Dense(num_units, activation=activation),
            layers.Dropout(dropout),
            layers.Dense(num_units, activation=activation),
            layers.Dropout(dropout),
        ]
    )


def normalizer(axis):
    def _normalize(att_weights):
        exp_weights = K.exp(att_weights)
        sum_weights = K.sum(exp_weights, axis=axis, keepdims=True)
        return exp_weights / sum_weights

    return _normalize


def sum_word(x):
    return K.sum(x, axis=1)


def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives+K.epsilon())
    recall = true_positives / (possible_positives+K.epsilon())
    f1_val = 2*(precision*recall) / (precision+recall+K.epsilon())
    return f1_val


def test_build_model():
    vectors = np.ndarray((100, 8), dtype="float32")
    shape = (10, 16, 3)
    settings = {"lr": 0.001, "dropout": 0.2, "gru_encode": True, "entail_dir": "both"}
    model = build_model(vectors, shape, settings)


def test_fit_model():
    def _generate_X(nr_example, length, nr_vector):
        X1 = np.ndarray((nr_example, length), dtype="int32")
        X1 *= X1 < nr_vector
        X1 *= 0 <= X1
        X2 = np.ndarray((nr_example, length), dtype="int32")
        X2 *= X2 < nr_vector
        X2 *= 0 <= X2
        return [X1, X2]

    def _generate_Y(nr_example, nr_class):
        ys = np.zeros((nr_example, nr_class), dtype="int32")
        for i in range(nr_example):
            ys[i, i % nr_class] = 1
        return ys

    vectors = np.ndarray((100, 8), dtype="float32")
    shape = (10, 16, 3)
    settings = {"lr": 0.001, "dropout": 0.2, "gru_encode": True, "entail_dir": "both"}
    model = build_model(vectors, shape, settings)

    train_X = _generate_X(20, shape[0], vectors.shape[0])
    train_Y = _generate_Y(20, shape[2])
    dev_X = _generate_X(15, shape[0], vectors.shape[0])
    dev_Y = _generate_Y(15, shape[2])

    model.fit(train_X, train_Y, validation_data=(dev_X, dev_Y), epochs=5, batch_size=4)


__all__ = [build_model]
