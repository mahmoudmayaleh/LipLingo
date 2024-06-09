from Base import produce_example
from Base import architecture
from Base import loss
from Base import data_loader

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
import argparse
import os


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Define the methods for data processing and training configurations
def load_dataset(dataset_path):
    return tf.data.Dataset.list_files(dataset_path)

def shuffle_dataset(dataset, buffer_size, reshuffle_each_iteration=False):
    return dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration)

def map_dataset(dataset, map_func):
    return dataset.map(map_func)

def batch_and_prefetch_dataset(dataset, batch_size, padded_shapes):
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    return dataset.prefetch(tf.data.AUTOTUNE)

def slice_dataset(data, train_percent, val_percent):
    data_size = sum(1 for _ in data)
    train_size = int(data_size * train_percent)
    val_size = int(data_size * val_percent)
    test_size = data_size - train_size - val_size

    train_data = data.take(train_size)
    remaining_data = data.skip(train_size)
    val_data = remaining_data.take(val_size)
    test_data = remaining_data.skip(val_size)

    return train_data, test_data, val_data

def create_callbacks(model_name, train_data, num_to_char, callback_names):
    callbacks = []
    if 'checkpoint' in callback_names:
        checkpoint_callback = ModelCheckpoint(os.path.join('Model', f'{model_name}.weights.h5'), monitor='loss', save_weights_only=True)
        callbacks.append(checkpoint_callback)
    if 'scheduler' in callback_names:
        schedule_callback = LearningRateScheduler(scheduler)
        callbacks.append(schedule_callback)
    if 'example' in callback_names:
        example_callback = produce_example.ProduceExample(train_data, num_to_char)
        callbacks.append(example_callback)
    return callbacks


def train_model(model, train_data, val_data, callbacks, epochs):
    model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Data processing and training configuration')
    parser.add_argument('--dataset_path', type=str, default='/content/data/t-data/s/*.mpg', help='Path to the dataset')
    parser.add_argument('--buffer_size', type=int, default=500, help='Buffer size for shuffling')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data')
    parser.add_argument('--padded_shapes', type=str, default='[[75, None, None], [40]]', help='Padded shapes for batching')
    parser.add_argument('--train_percent', type=float, default=0.8, help='Percentage of data to use for training')
    parser.add_argument('--val_percent', type=float, default=0.1, help='Percentage of data to use for validation')
    parser.add_argument('--test_percent', type=float, default=0.1, help='Percentage of data to use for testing')
    parser.add_argument('--model_name', type=str, default='model', help='Name of the model')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs for training')
    parser.add_argument('--callbacks', type=str, nargs='+', default=['checkpoint', 'scheduler', 'example'], help='List of callbacks to use')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to the pre-trained model')
    return parser.parse_args()

# Execute the main function
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Convert padded_shapes from string to list
    padded_shapes = eval(args.padded_shapes)

    # Dataset loading and processing
    data = load_dataset(args.dataset_path)
    data = shuffle_dataset(data, buffer_size=args.buffer_size)
    data = map_dataset(data, data_loader.mappable_function)
    data = batch_and_prefetch_dataset(data, batch_size=args.batch_size, padded_shapes=padded_shapes)

    # Data slicing
    train_data, test_data, val_data = slice_dataset(data, args.train_percent, args.val_percent)

    # Callbacks
    char_to_num, num_to_char = architecture.char_to_num_vocab()
    callbacks = create_callbacks(args.model_name, train_data, num_to_char, args.callbacks)

    # Model training
    model = architecture.model(char_to_num)

    if args.pretrained_model_path:
        model.load_weights(args.pretrained_model_path)
        print(f"Loaded pre-trained model from {args.pretrained_model_path}")

    model = architecture.compile_model(model, loss.CTCLoss)
    train_model(model, train_data, val_data, callbacks, epochs=args.epochs)
