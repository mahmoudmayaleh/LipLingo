from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset, num_to_char) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_to_char = num_to_char
        self.iterator = self.dataset.as_numpy_iterator()
        self.true_labels = []
        self.predicted_labels = []

    def on_epoch_end(self, epoch, logs=None) -> None:
        try:
            data = self.iterator.next()
        except StopIteration:
            self.iterator = self.dataset.as_numpy_iterator()
            data = self.iterator.next()
            print("End of dataset reached.")
            return

        if data is not None:
            yhat = self.model.predict(data[0])
            decoded = tf.keras.backend.ctc_decode(yhat, [75] * yhat.shape[0], greedy=False)[0][0].numpy()

            original_text = tf.strings.reduce_join(self.num_to_char(data[1][0])).numpy().decode('utf-8')
            predicted_text = tf.strings.reduce_join(self.num_to_char(decoded[0])).numpy().decode('utf-8')
            print('~' * 100)
            print('Original:', original_text)
            print('Prediction:', predicted_text)
            print('~' * 100)


            min_length = min(len(original_text), len(predicted_text))
            self.true_labels.extend(list(original_text[:min_length]))
            self.predicted_labels.extend(list(predicted_text[:min_length]))

            if epoch == self.params['epochs'] - 1:
                self.calculate_metrics()
                self.compute_confusion_matrix()

    def compute_confusion_matrix(self):
        if not self.true_labels or not self.predicted_labels:
            print("No labels collected for confusion matrix.")
            return

        true_labels_flat = self.true_labels
        predicted_labels_flat = self.predicted_labels

        labels = sorted(set(true_labels_flat + predicted_labels_flat))
        label_to_index = {label: idx for idx, label in enumerate(labels)}

        cm = np.zeros((len(labels), len(labels)), dtype=int)

        for true, pred in zip(true_labels_flat, predicted_labels_flat):
            true_idx = label_to_index.get(true, -1)
            pred_idx = label_to_index.get(pred, -1)
            if true_idx != -1 and pred_idx != -1:
                cm[true_idx, pred_idx] += 1

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def calculate_metrics(self):
        true_labels_flat = self.true_labels
        predicted_labels_flat = self.predicted_labels

        accuracy = accuracy_score(true_labels_flat, predicted_labels_flat)
        precision = precision_score(true_labels_flat, predicted_labels_flat, average='weighted', zero_division=0)
        recall = recall_score(true_labels_flat, predicted_labels_flat, average='weighted', zero_division=0)
        f1 = f1_score(true_labels_flat, predicted_labels_flat, average='weighted', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")