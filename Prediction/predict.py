from Base import architecture
import tensorflow as tf

sample = "data"
model = architecture.model
char_to_num, num_to_char = architecture.char_to_num_vocab()
model.load_weights("checkpoint-98.weights.h5")

yhat = model.predict(next(sample)[0])
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
print('~'*100, 'PREDICTIONS')
[tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]


