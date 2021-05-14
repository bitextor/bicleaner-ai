import tensorflow as tf

class KDLoss(tf.keras.losses.SparseCategoricalCrossentropy):
    ''' Knowledge Distillation loss
    Computes KD loss from student CE loss and KLD loss between
    student and teacher predictions.
    Assumes teacher predictions are coming as sample weights.
    '''
    def __init__(self, batch_size, temperature=3, alpha=0.1,
            name='knowledge_distillation_loss', **kwargs):
        super(KDLoss, self).__init__(name=name,
                                     **kwargs)
        self.reduction = tf.keras.losses.Reduction.NONE
        self.batch_size = batch_size
        self.name = name
        self.__name__ = name
        self.temperature = temperature
        self.alpha = alpha
        self.kld = tf.keras.losses.KLDivergence(reduction=self.reduction)

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Weight trick, using sample weights to get teacher predictions
        # Manually reduce loss with compute_average_loss
        # assuming mirrored strategy is being used

        # Return cross entropy loss if no weights
        if sample_weight==None:
            self.reduction = tf.keras.losses.Reduction.AUTO
            loss = super(KDLoss, self).__call__(y_true, y_pred)
            self.reduction = tf.keras.losses.Reduction.NONE
            return loss

        student_loss = super(KDLoss, self).__call__(y_true, y_pred)
        student_loss = tf.nn.compute_average_loss(student_loss,
                                      global_batch_size=self.batch_size)
        distillation_loss = self.kld(
                tf.nn.softmax(sample_weight/self.temperature, axis=1),
                tf.nn.softmax(y_pred/self.temperature, axis=1),
            )
        distillation_loss = tf.nn.compute_average_loss(distillation_loss,
                                          global_batch_size=self.batch_size)

        return self.alpha * student_loss + (1 - self.alpha) * distillation_loss

    def get_config(self):
        # Add temperature and alpha to the class config for seliarization
        config = {
            "temperature": self.temperature,
            "apha": self.alpha,
        }
        base_config = super(KDLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
