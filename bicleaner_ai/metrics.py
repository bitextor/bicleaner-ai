from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.keras.initializers import zeros as zeros_initializer
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

class FScore(Metric):
    '''Stateful Keras f-score metric'''
    def __init__(self,
            beta=1,
            thresholds=None,
            top_k=None,
            class_id=None,
            name=None,
            dtype=None,
            argmax=False):
        super(FScore, self).__init__(name=name, dtype=dtype)
        self.beta = beta
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        self.argmax = argmax

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
                thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
                'true_positives',
                shape=(len(self.thresholds),),
                initializer=zeros_initializer)
        self.false_positives = self.add_weight(
                'false_positives',
                shape=(len(self.thresholds),),
                initializer=zeros_initializer)
        self.false_negatives = self.add_weight(
                'false_negatives',
                shape=(len(self.thresholds),),
                initializer=zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''Accumulates true positive and false positive statistics.'''
        if self.argmax:
            y_pred = K.argmax(y_pred)

        return metrics_utils.update_confusion_matrix_variables(
                {
                    metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                    metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                    metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
                    },
                y_true,
                y_pred,
                thresholds=self.thresholds,
                top_k=self.top_k,
                class_id=self.class_id,
                sample_weight=sample_weight)

    def result(self):
        b2 = self.beta**2
        q = (1+b2)*self.true_positives
        d = (1+b2)*self.true_positives + b2*self.false_negatives + self.false_positives
        result = tf.math.divide_no_nan(q, d)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
                'thresholds': self.init_thresholds,
                'top_k': self.top_k,
                'class_id': self.class_id,
                'beta': self.beta,
                }
        base_config = super(FScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MatthewsCorrCoef(Metric):
    '''Stateful Keras Matthews correlation coefficient as a metric'''
    def __init__(self,
            thresholds=None,
            top_k=None,
            class_id=None,
            name=None,
            dtype=None,
            argmax=False):
        super(MatthewsCorrCoef, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        self.argmax = argmax

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
                thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
                'true_positives',
                shape=(len(self.thresholds),),
                initializer=zeros_initializer)
        self.false_positives = self.add_weight(
                'false_positives',
                shape=(len(self.thresholds),),
                initializer=zeros_initializer)
        self.false_negatives = self.add_weight(
                'false_negatives',
                shape=(len(self.thresholds),),
                initializer=zeros_initializer)
        self.true_negatives = self.add_weight(
                'true_negatives',
                shape=(len(self.thresholds),),
                initializer=zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''Accumulates true positive and false positive statistics.'''
        if self.argmax:
            y_pred = K.argmax(y_pred)

        return metrics_utils.update_confusion_matrix_variables(
                {
                    metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                    metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                    metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
                    metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives
                    },
                y_true,
                y_pred,
                thresholds=self.thresholds,
                top_k=self.top_k,
                class_id=self.class_id,
                sample_weight=sample_weight)

    def result(self):
        N = (self.true_negatives + self.true_positives
                + self.false_negatives + self.false_positives)
        S = (self.true_positives + self.false_negatives) / N
        P = (self.true_positives + self.false_positives) / N
        result = tf.math.divide_no_nan(self.true_positives/N - S*P,
                                       tf.math.sqrt(P * S * (1-S) * (1-P)))
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
                'thresholds': self.init_thresholds,
                'top_k': self.top_k,
                'class_id': self.class_id,
                }
        base_config = super(MatthewsCorrCoef, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
