from contextlib import redirect_stdout
import logging
import sys

from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import matthews_corrcoef
import tensorflow as tf
import numpy as np

def calibrate_output(y_true, y_pred):
    ''' Platt calibration
    Estimate A*f(x)+B sigmoid parameters
    '''
    logging.info("Calibrating classifier output")
    init_mcc = matthews_corrcoef(y_true, np.where(y_pred>=0.5, 1, 0))
    # Define target values
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos < n_neg:
        # Separate pos and neg
        y_true_pos = np.extract(y_true == 1, y_true)
        y_true_neg = np.extract(y_true == 0, y_true)
        y_pred_pos = np.extract(y_true == 1, y_pred)
        y_pred_neg = np.extract(y_true == 0, y_pred)
        # Shuffle by index to shuffle with the same pattern preds and labels
        # and avoid srewing up labels
        idx_neg = np.arange(len(y_true_neg))
        np.random.shuffle(idx_neg)
        # Extract from the shuffle the same amount of neg and pos
        y_true_balanced = np.append(y_true_neg[idx_neg][:len(y_true_pos)], y_true_pos)
        y_pred_balanced = np.append(y_pred_neg[idx_neg][:len(y_pred_pos)], y_pred_pos)
    else:
        y_true_balanced = y_true
        y_pred_balanced = y_pred

    y_target = np.where(y_true_balanced == 1, (n_pos+1)/(n_pos+2), y_true_balanced)
    y_target = np.where(y_target == 0, 1/(n_neg+2), y_target)

    # Parametrized sigmoid is equivalent to
    # dense with single neuron and bias A*x + B
    with tf.device("/cpu:0"):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        loss = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        model.compile(optimizer=Adam(learning_rate=5e-3), loss=loss)

        if logging.getLogger().level == logging.DEBUG:
            verbose = 2
        else:
            verbose = 0
        with redirect_stdout(sys.stderr):
            model.fit(y_pred_balanced, y_target, epochs=5000, verbose=verbose,
                      batch_size=4096,
                      validation_split=0.1,
                      callbacks=[earlystop])

    # Check mcc hasn't been affected
    with redirect_stdout(sys.stderr):
        y_pred_calibrated = model.predict(y_pred, verbose=verbose)
    end_mcc = matthews_corrcoef(y_true, np.where(y_pred_calibrated>=0.5, 1, 0))
    logging.debug(f"MCC with calibrated output: {end_mcc}")
    if (init_mcc - end_mcc) > 0.02:
        logging.warning(f"Calibration has decreased MCC from {init_mcc:.4f} to {end_mcc:.4f}")

    # Obtain scalar values from model weights
    A = float(model.layers[0].weights[0].numpy()[0][0])
    B = float(model.layers[0].weights[1].numpy()[0])
    logging.debug(f"Calibrated parameters: {A} * x + {B}")
    return A, B



# Method imported from keras.saving.legacy
# as it is marked as deprecated and we need it
# will disappear in TF 2.12
def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.
    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.
    Returns:
        data: Attributes data.
    """
    if name in group.attrs:
        data = [
            n.decode("utf8") if hasattr(n, "decode") else n
            for n in group.attrs[name]
        ]
    else:
        data = []
        chunk_id = 0
        while "%s%d" % (name, chunk_id) in group.attrs:
            data.extend(
                [
                    n.decode("utf8") if hasattr(n, "decode") else n
                    for n in group.attrs["%s%d" % (name, chunk_id)]
                ]
            )
            chunk_id += 1
    return data
