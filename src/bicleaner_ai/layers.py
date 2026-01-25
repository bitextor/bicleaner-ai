from transformers.modeling_tf_utils import get_initializer
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


# =============================================================================
# XLA-Compatible Custom Layers
# =============================================================================
# OPTIMIZATION: Lambda layers are not compatible with XLA (jit_compile=True).
# These custom layers replace Lambda calls in decomposable_attention.py,
# enabling XLA compilation for 10-30% inference speedup.
# See: https://www.tensorflow.org/xla
# =============================================================================


class SoftmaxNormalizer(layers.Layer):
    """Softmax normalization layer (XLA-compatible replacement for Lambda).

    Applies softmax normalization along specified axis.
    Used in Decomposable Attention for attention weight normalization.

    Why custom layer instead of Lambda:
        Lambda layers cannot be traced by XLA compiler, preventing
        jit_compile=True optimization. Custom layers with get_config()
        are fully serializable and XLA-compatible.
    """

    def __init__(self, axis=1, **kwargs):
        super(SoftmaxNormalizer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.nn.softmax(x, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(SoftmaxNormalizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SumAlong(layers.Layer):
    """Sum along axis layer (XLA-compatible replacement for Lambda).

    Sums tensor values along specified axis.
    Used in Decomposable Attention for aggregating word representations.

    Why custom layer instead of Lambda:
        Same as SoftmaxNormalizer - enables XLA jit_compile optimization.
    """

    def __init__(self, axis=1, **kwargs):
        super(SumAlong, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.reduce_sum(x, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(SumAlong, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

class BicleanerAIClassificationHead(layers.Layer):
    """
    Head for Bicleaner sentence classification tasks.
    It reads BicleanerAIConfig to be able to change
    classifier layer parameters (size, activation and dropout)
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(
            config.head_hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation=config.head_activation,
            name="dense",
        )
        self.dropout = layers.Dropout(config.head_dropout)
        self.out_proj = layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="out_proj"
        )
        self.config = config

    def call(self, features, training=False):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # This was failing because our dense layer last input dimension is
                # actually config.hidden_size and not config.head_hidden_size
                # self.hidden_size is the original parameter and is 768, just like the last XLMR state
                # we use a different hidden size with config.head_hidden_size (typically 2048)
                # so even if we use config.head_hidden_size as the number of neurones in self.dense
                # the shape of dense is (self.head_hidden_size,self.hidden_size) (2048,768)
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.head_hidden_size])
