from Base import produce_example
from Base import architecture
from Base import loss
from Base import data_loader

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import tensorflow as tf
import os


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# TODO:
# method for configration the data processing like giving dataset,
# shuffling, mapping, data slicing, validation set, callbacks selecting
# and training configurations, then saving the methods in data_loader.

# dataset
data = tf.data.Dataset.list_files('/content/data/t-data/s/*.mpg')

# shuffling
data = data.shuffle(500, reshuffle_each_iteration=False)

# mapping
data = data.map(data_loader.mappable_function)

data = data.padded_batch(1, padded_shapes=([75,None,None],[40]))
data = data.prefetch(tf.data.AUTOTUNE)

# data slicing
train = data.take(400)
test = data.skip(400)

# validation set
val_data = train.take(30)

char_to_num, num_to_char = architecture.char_to_num_vocab()

# callbacks
model_name = 'model'
checkpoint_callback = ModelCheckpoint(os.path.join('Model',f'{model_name}.weights.h5'), monitor='loss', save_weights_only=True)
schedule_callback = LearningRateScheduler(scheduler)
example_callback = produce_example.ProduceExample(train, num_to_char)

# training
model = architecture.model(char_to_num)
model = architecture.compile_model(model, loss.CTCLoss)
model.fit(train, epochs=4, validation_data=val_data, callbacks=[checkpoint_callback, schedule_callback])






