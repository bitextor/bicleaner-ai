from transformers.modeling_tf_utils import get_initializer
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

class TokenAndPositionEmbedding(layers.Layer):
    '''Token and positional embeddings layer with pre-trained weights'''
    def __init__(self, input_dim, output_dim, input_length,
            vectors=None, trainable=False, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length

        if vectors is not None:
            vectors = [vectors]
        self.token_emb = layers.Embedding(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            input_length=input_length,
                            weights=vectors,
                            trainable=trainable,
                            mask_zero=True)
        self.pos_emb = layers.Embedding(
                            input_dim=input_length,
                            output_dim=output_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_length": self.input_length,
        }
        base_config = super(TokenAndPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TransformerBlock(layers.Layer):
    '''Transformer block as a Keras layer'''
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Divide key dimension by the number of heads
        # so that each head is projected to a lower dimension
        # then gets concatenated
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim//num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class BCClassificationHead(layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_size, dropout, activation, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(
            hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation=activation,
            name="dense",
        )
        self.dropout = layers.Dropout(dropout)
        self.out_proj = layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="out_proj"
        )

    def call(self, features, training=False):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x
