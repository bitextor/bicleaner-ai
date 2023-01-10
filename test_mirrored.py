from transformers import TFAutoModelForSequenceClassification
from transformers.optimization_tf import create_optimizer
from transformers import AutoTokenizer
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import tensorflow as tf
import numpy as np

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  # Just take the training split for now


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_data = dict(tokenizer(dataset["sentence"], return_tensors="np", padding=True))

labels = np.array(dataset["label"])  # Label is already an array of 0 and 1

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")
# Lower learning rates are often better for fine-tuning transformers
optimizer, _ = create_optimizer(3e-5, 600, 100, weight_decay_rate=0.3)
model.compile(optimizer=optimizer, loss='binary_crossentropy')

model.fit(tokenized_data, labels)
