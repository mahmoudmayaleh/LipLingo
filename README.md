# LipNet Implementation

<img src="https://github.com/user-attachments/assets/aac974e6-6aae-42bf-8b76-b05ba62300ba" alt="Liplingo Poster" width="550" height="800">




This repository contains an implementation of a lip reading model using TensorFlow and Keras. The model processes video frames to predict the spoken words using a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks.

## Table of Contents

- [LipNet Implementation](#lipnet-implementation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
  - [License](#license)

## Installation
To set up the environment and install the necessary packages, run the following commands:

```bash
pip install opencv-python matplotlib imageio gdown tensorflow mediapipe Levenshtein
```
Dataset
Download the dataset and extract it:
```bash
import gdown
import zipfile

url = 'https://drive.google.com/uc?id=1SSlW9fbuDirHLUlfko28MoFLvxwjprxV'
output = 'data.zip'
gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data')
```
Usage
Load Data
Use the provided functions to load and preprocess video data:

```bash
def load_video(path: str) -> np.ndarray:
    # Implementation details...
    return normalized_frames

def load_alignments(path: str) -> List[str]:
    # Implementation details...
    return alignments

def load_data(path: str):
    # Implementation details...
    return frames, alignments

def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result
```
Data Pipeline
Create a data pipeline using TensorFlow:
```bash
data = tf.data.Dataset.list_files('/content/data/t-data/s/*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(1, padded_shapes=([75, None, None], [40]))
data = data.prefetch(tf.data.AUTOTUNE)
train = data.take(400)
test = data.skip(400)
```
Training
To train the model:
1) Setup the training options:
```bash
def scheduler(epoch, lr):
    # Implementation details...
    return new_lr

def CTCLoss(y_true, y_pred):
    # Implementation details...
    return loss

checkpoint_callback = ModelCheckpoint('checkpoint.weights.h5', monitor='loss', save_weights_only=True)
schedule_callback = LearningRateScheduler(scheduler)

model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
```
Train the model:
```bash
model.fit(train, epochs=4, validation_data=val_data, callbacks=[checkpoint_callback, schedule_callback])
```

Evaluation
To evaluate the model:
Calculate similarity matrix:

```bash
def calculate_similarity_matrix(dataset, model, num_to_char):
    # Implementation details...
    pass

def compute_confusion_matrix(true_labels, predicted_labels):
    # Implementation details...
    pass

def calculate_metrics(true_labels, predicted_labels):
    # Implementation details...
    pass

calculate_similarity_matrix(test, model, num_to_char)
```
Calculate character-level similarity matrix:

```bash
def calculate_chars_similarity_matrix(dataset, model, num_to_char):
    # Implementation details...
    pass

calculate_chars_similarity_matrix(test, model, num_to_char)
```

Prediction
To make predictions:

Load the trained model and make predictions:
```bash
test_data = test.as_numpy_iterator()
sample = data.as_numpy_iterator()

yhat = model.predict(next(sample)[0])

decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

for sentence in decoded:
    print(tf.strings.reduce_join([num_to_char(word) for word in sentence]))
```

License
This project is licensed under the MIT License. See the LICENSE file for details.
